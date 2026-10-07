"""
Utilitas bersama untuk retraining (active learning).

Modul ini sengaja bebas dependensi model: hanya numpy + torch untuk seeding,
sehingga seluruh isinya bisa diuji tanpa mengunduh bobot HuggingFace.

Alasan keberadaan modul ini (CRISP-ML(Q) fase Evaluation & Monitoring):
retraining sebelumnya menimpa model tanpa seed tetap, tanpa stratifikasi, dan
tanpa mengukur apakah model menjadi lebih baik. Ketiganya diperlukan agar angka
pada laporan bisa direproduksi dan regresi model bisa terdeteksi.
"""

import os
import random
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from app.utils.logger import setup_logger

logger = setup_logger(__name__)


# ── Reproducibility ─────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """
    Kunci seluruh sumber keacakan yang dipakai jalur retraining.

    Args:
        seed: Nilai seed yang dipakai python, numpy, dan torch (CPU + CUDA).
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover - torch selalu ada di requirements
        logger.warning("torch tidak tersedia, seed torch dilewati")


# ── Split ───────────────────────────────────────────────────────────────────

def stratified_split(
    labels: Sequence[Any],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    """
    Bagi indeks menjadi train/val dengan proporsi kelas dipertahankan.

    `random_split` bawaan torch mengabaikan label, sehingga pada data koreksi yang
    timpang (mis. 78% negatif) set validasi bisa kehilangan sebuah kelas dan
    metriknya menjadi menyesatkan. Setiap kelas di sini menyumbang minimal satu
    sampel ke validasi selama jumlahnya >= 2.

    Args:
        labels: Label per sampel, sejajar dengan urutan dataset.
        val_ratio: Porsi data yang masuk ke set validasi (0 < ratio < 1).
        seed: Seed untuk pengacakan di dalam tiap kelas.

    Returns:
        Tuple (train_indices, val_indices), keduanya terurut menaik.
    """
    if not labels:
        return [], []

    rng = random.Random(seed)

    by_label: Dict[Any, List[int]] = {}
    for idx, label in enumerate(labels):
        by_label.setdefault(label, []).append(idx)

    train_idx: List[int] = []
    val_idx: List[int] = []

    for label, indices in by_label.items():
        shuffled = indices[:]
        rng.shuffle(shuffled)

        n = len(shuffled)
        if n == 1:
            # Kelas dengan satu sampel tidak bisa dibagi; pakai untuk training
            # supaya informasinya tidak hilang sama sekali.
            train_idx.extend(shuffled)
            continue

        n_val = max(1, int(round(n * val_ratio)))
        n_val = min(n_val, n - 1)  # sisakan minimal satu sampel untuk training

        val_idx.extend(shuffled[:n_val])
        train_idx.extend(shuffled[n_val:])

    return sorted(train_idx), sorted(val_idx)


# ── Ketimpangan kelas ───────────────────────────────────────────────────────

def label_distribution(labels: Sequence[Any]) -> Dict[str, Any]:
    """
    Ringkas komposisi label agar bisa dilaporkan sebelum training dimulai.

    Returns:
        Dict berisi count per label, persentase, total, kelas mayoritas, dan
        imbalance_ratio (mayoritas / minoritas).
    """
    counts = Counter(labels)
    total = sum(counts.values())

    if total == 0:
        return {"total": 0, "counts": {}, "percentages": {}, "imbalance_ratio": 0.0}

    ordered = dict(counts.most_common())
    percentages = {
        str(label): round((count / total) * 100, 2) for label, count in ordered.items()
    }
    majority = max(ordered.values())
    minority = min(ordered.values())

    return {
        "total": total,
        "counts": {str(label): count for label, count in ordered.items()},
        "percentages": percentages,
        "majority_class": str(next(iter(ordered))),
        "imbalance_ratio": round(majority / minority, 2) if minority else 0.0,
    }


def compute_class_weights(
    labels: Sequence[int],
    num_classes: int,
) -> List[float]:
    """
    Bobot kelas berbanding terbalik dengan frekuensi (balanced heuristic sklearn).

    Tanpa pembobotan, fine-tuning pada data timpang cenderung menebak kelas
    mayoritas saja — akurasinya terlihat naik padahal model menjadi tidak berguna.

    Args:
        labels: Label integer (sudah dipetakan ke id kelas).
        num_classes: Jumlah kelas total; kelas yang tidak muncul diberi bobot 1.0.

    Returns:
        List bobot sepanjang num_classes.
    """
    counts = Counter(labels)
    total = sum(counts.values())

    if total == 0:
        return [1.0] * num_classes

    present = [c for c in range(num_classes) if counts.get(c, 0) > 0]
    n_present = len(present) or 1

    weights: List[float] = []
    for c in range(num_classes):
        count = counts.get(c, 0)
        # Kelas yang absen tidak boleh menghasilkan pembagian nol.
        weights.append(round(total / (n_present * count), 4) if count else 1.0)

    return weights


# ── Metrik ──────────────────────────────────────────────────────────────────

def classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    label_names: Optional[Dict[int, str]] = None,
) -> Dict[str, Any]:
    """
    Hitung accuracy, precision/recall/F1 per kelas, rata-rata makro & terbobot,
    serta confusion matrix — cukup untuk memenuhi syarat "minimal 3 metrik"
    pada Rule 2 (CRISP-ML(Q)) di dokumen arsitektur.

    Args:
        y_true: Label sebenarnya.
        y_pred: Label prediksi, panjang sama dengan y_true.
        label_names: Peta id kelas -> nama untuk pelaporan.

    Returns:
        Dict metrik siap diserialisasi ke JSON.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true dan y_pred harus sama panjang")

    total = len(y_true)
    if total == 0:
        return {"accuracy": 0.0, "macro_f1": 0.0, "weighted_f1": 0.0, "per_class": {}, "support": 0}

    classes = sorted(set(y_true) | set(y_pred))
    names = label_names or {}

    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / total

    per_class: Dict[str, Dict[str, float]] = {}
    macro_f1 = 0.0
    weighted_f1 = 0.0

    for c in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        support = sum(1 for t in y_true if t == c)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        per_class[names.get(c, str(c))] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }

        macro_f1 += f1
        weighted_f1 += f1 * support

    macro_f1 = macro_f1 / len(classes) if classes else 0.0
    weighted_f1 = weighted_f1 / total

    # Confusion matrix: baris = label sebenarnya, kolom = prediksi
    confusion = [[0] * len(classes) for _ in classes]
    pos = {c: i for i, c in enumerate(classes)}
    for t, p in zip(y_true, y_pred):
        confusion[pos[t]][pos[p]] += 1

    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class": per_class,
        "confusion_matrix": confusion,
        "confusion_labels": [names.get(c, str(c)) for c in classes],
        "support": total,
    }


def token_f1(
    y_true: Sequence[Sequence[int]],
    y_pred: Sequence[Sequence[int]],
    ignore_index: int = -100,
    positive_labels: Sequence[int] = (1, 2),
) -> Dict[str, Any]:
    """
    F1 tingkat token untuk BIO tagging, menghitung hanya token aspek
    (B-ASPECT/I-ASPECT) dan mengabaikan padding serta token spesial.

    Accuracy penuh menyesatkan di sini karena mayoritas token berlabel 'O';
    model yang memprediksi semua-O bisa terlihat >90% akurat.

    Args:
        y_true: Label BIO per sampel.
        y_pred: Prediksi BIO per sampel.
        ignore_index: Nilai label yang dilewati (padding / token spesial).
        positive_labels: Id label yang dianggap "aspek".

    Returns:
        Dict berisi precision, recall, f1, dan support token aspek.
    """
    positives = set(positive_labels)
    tp = fp = fn = 0
    support = 0

    for true_seq, pred_seq in zip(y_true, y_pred):
        for t, p in zip(true_seq, pred_seq):
            if t == ignore_index:
                continue
            t_pos = t in positives
            p_pos = p in positives
            if t_pos:
                support += 1
            if t_pos and p_pos:
                tp += 1
            elif p_pos and not t_pos:
                fp += 1
            elif t_pos and not p_pos:
                fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "aspect_token_support": support,
    }


# ── Keputusan penerimaan checkpoint ─────────────────────────────────────────

def should_accept_checkpoint(
    metric_before: float,
    metric_after: float,
    min_delta: float = 0.0,
    force: bool = False,
) -> Tuple[bool, float]:
    """
    Tentukan apakah hasil fine-tuning layak menggantikan model sebelumnya.

    Dipisah dari loop training agar aturannya bisa diuji tanpa melatih apa pun,
    dan agar jalur sentimen serta aspek memakai aturan yang persis sama.

    Args:
        metric_before: Metrik utama model lama pada set validasi.
        metric_after: Metrik utama model baru pada set validasi yang sama.
        min_delta: Toleransi penurunan yang masih diterima (>= 0).
        force: Terima checkpoint tanpa memandang metrik.

    Returns:
        Tuple (accept, delta) dengan delta = metric_after - metric_before.
    """
    delta = round(metric_after - metric_before, 4)
    accept = force or delta >= -abs(min_delta)
    return accept, delta


# ── Kalibrasi confidence ────────────────────────────────────────────────────

def expected_calibration_error(
    confidences: Sequence[float],
    correct: Sequence[bool],
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error - selisih rata-rata antara keyakinan dan akurasi.

    Model klasifikasi berbasis transformer terkenal *overconfident*: ia
    menyatakan 0,95 pada kasus yang sebenarnya benar 0,85 (Guo, Pleiss, Sun &
    Weinberger, 2017). Angka itu penting di sini karena confidence dipakai
    untuk memilih baris yang perlu dikoreksi manusia - kalau confidence-nya
    tidak jujur, antrean koreksi ikut salah sasaran.

    ECE membagi prediksi ke `n_bins` pita selebar sama, lalu merata-ratakan
    |akurasi - keyakinan| tiap pita dengan bobot jumlah anggotanya. Nilai 0
    berarti kalibrasi sempurna.

    Args:
        confidences: Keyakinan prediksi (probabilitas kelas terpilih), 0..1.
        correct: True bila prediksi pada posisi itu benar.
        n_bins: Jumlah pita.

    Returns:
        ECE pada rentang 0..1.
    """
    if len(confidences) != len(correct):
        raise ValueError("confidences dan correct harus sama panjang")

    total = len(confidences)
    if total == 0:
        return 0.0

    error = 0.0
    for i in range(n_bins):
        low = i / n_bins
        high = (i + 1) / n_bins

        # Pita pertama inklusif di batas bawah agar confidence 0,0 ikut terhitung;
        # pita berikutnya eksklusif supaya tidak ada prediksi yang dihitung dua kali.
        if i == 0:
            members = [j for j in range(total) if low <= confidences[j] <= high]
        else:
            members = [j for j in range(total) if low < confidences[j] <= high]

        if not members:
            continue

        avg_conf = sum(confidences[j] for j in members) / len(members)
        accuracy = sum(1 for j in members if correct[j]) / len(members)
        error += (len(members) / total) * abs(accuracy - avg_conf)

    return round(error, 4)


def fit_temperature(
    logits: Any,
    labels: Any,
    max_iter: int = 100,
    lr: float = 0.1,
    min_samples: int = 50,
) -> float:
    """
    Cari suhu T yang meminimalkan NLL pada set validasi (temperature scaling).

    Prediksi dihitung ulang sebagai softmax(logits / T). Karena pembagian
    dengan skalar positif tidak mengubah urutan logit, **argmax tidak berubah**
    sama sekali: akurasi, F1, dan confusion matrix tetap persis sama, hanya
    keyakinannya yang menjadi jujur (Guo, Pleiss, Sun & Weinberger, 2017).

    T > 1 berarti model terlalu yakin dan keyakinannya diturunkan; T < 1
    sebaliknya.

    **Kalibrasi tidak selalu terdefinisi, dan itu ditangani di sini.** Bila set
    validasi tidak memuat satu pun kesalahan, NLL diminimalkan dengan membuat
    model se-yakin mungkin, sehingga T meluncur ke nol. Terukur pada jalur
    retraining nyata: 19 sampel validasi yang semuanya benar menghasilkan
    **T = 0,308**, yang bukan kalibrasi melainkan kebalikannya - model dibuat
    maksimal overconfident, dan `review_queue` yang bergantung pada confidence
    ikut lumpuh. Karena itu fungsi ini menolak mengalibrasi ketika:

    - sampelnya terlalu sedikit (`min_samples`; Guo dkk. memakai ribuan),
    - tidak ada kesalahan sama sekali (tidak ada yang bisa dikalibrasi), atau
    - tidak ada satu pun prediksi benar (set terlalu menyimpang untuk dipercaya).

    Dalam semua kasus itu ia mengembalikan 1.0, yaitu "tanpa kalibrasi" -
    pilihan yang aman, karena confidence apa adanya lebih baik daripada
    confidence yang sengaja dirusak.

    Args:
        logits: Tensor torch [n, n_kelas] keluaran mentah model.
        labels: Tensor torch [n] berisi id kelas sebenarnya.
        max_iter: Batas iterasi LBFGS.
        lr: Learning rate LBFGS.
        min_samples: Jumlah minimum sampel validasi agar kalibrasi dijalankan.

    Returns:
        Suhu T pada rentang [0.5, 5.0], atau 1.0 bila kalibrasi tidak layak.
    """
    import torch

    if logits is None or labels is None or len(logits) < min_samples:
        jumlah = 0 if logits is None else len(logits)
        logger.info(
            f"Kalibrasi suhu dilewati: {jumlah} sampel validasi (< {min_samples}). "
            f"T=1.0 dipakai."
        )
        return 1.0

    benar = int((logits.argmax(dim=1) == labels).sum().item())
    total = len(labels)

    if benar == total:
        # Tanpa kesalahan, NLL terus turun selama keyakinan dinaikkan, sehingga
        # optimasi meluncur ke T -> 0. Itu bukan kalibrasi.
        logger.info(
            f"Kalibrasi suhu dilewati: model benar pada seluruh {total} sampel "
            f"validasi, sehingga suhu tidak teridentifikasi. T=1.0 dipakai."
        )
        return 1.0

    if benar == 0:
        logger.warning(
            f"Kalibrasi suhu dilewati: model salah pada seluruh {total} sampel "
            f"validasi. T=1.0 dipakai."
        )
        return 1.0

    # Dioptimasi dalam ruang log agar T dijamin tetap positif.
    log_t = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=lr, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    try:
        optimizer.step(closure)
        temperature = float(log_t.exp().item())
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Optimasi suhu gagal ({exc}); T=1.0 dipakai")
        return 1.0

    if not (temperature == temperature) or temperature <= 0:  # NaN atau non-positif
        logger.warning("Optimasi suhu menghasilkan nilai tidak valid; T=1.0 dipakai")
        return 1.0

    if not (0.5 <= temperature <= 5.0):
        # Di luar rentang ini, penyesuaiannya lebih besar daripada yang bisa
        # dibenarkan set validasi seukuran data koreksi.
        logger.warning(
            f"Suhu hasil optimasi di luar rentang wajar ({temperature:.3f}); T=1.0 dipakai"
        )
        return 1.0

    return round(temperature, 4)


def pool_fingerprint(texts: Sequence[str]) -> str:
    """
    Sidik jari isi kolam pelatihan: 16 heksadesimal dari SHA-256 atas teks
    yang diurutkan.

    Diurutkan supaya urutan pengambilan dari basis data tidak mengubah hasil -
    yang diidentifikasi adalah ISI kolam, bukan urutannya. Kolam yang sama
    selalu memberi sidik jari yang sama, kolam yang berbeda hampir pasti tidak.
    """
    import hashlib

    h = hashlib.sha256()
    for t in sorted(str(t) for t in texts):
        h.update(t.encode('utf-8', 'replace'))
        h.update(b'\x00')
    return h.hexdigest()[:16]


def write_provenance(path: str, meta: Dict[str, Any]) -> None:
    """
    Catat asal-usul sebuah checkpoint di sebelahnya.

    Alasannya terukur, bukan hipotetis. Pada studi kasus ini dua pelatihan
    verifikasi dengan 300 kalimat sintetis tertulis ke `aspect_retrained.pt`,
    dan tidak ada satu pun cara untuk mengetahuinya dari berkas itu sendiri -
    ukuran, nama, dan `/health` semuanya tampak normal. Yang membongkarnya
    adalah membaca log baris per baris berjam-jam kemudian, setelah checkpoint
    antaranya sudah tertimpa dan angkanya tidak bisa direproduksi lagi.

    Dengan berkas ini, pertanyaan "checkpoint ini dilatih atas apa" bisa
    dijawab dari checkpoint itu sendiri. `pool_fingerprint` adalah bagian yang
    paling menentukan: kolam koreksi nyata dan data karangan memberi sidik jari
    berbeda, sehingga percampuran terlihat langsung alih-alih perlu diduga.

    Kegagalan menulis TIDAK boleh menggagalkan pelatihan - ini catatan audit,
    bukan prasyarat. Pelatihan aspek memakan enam menit CPU dan tidak pantas
    hilang karena satu berkas JSON gagal ditulis.
    """
    import json
    from datetime import datetime

    try:
        target = (os.path.join(path, 'provenance.json') if os.path.isdir(path)
                  else f'{path}.provenance.json')
        payload = {'dicatat_pada': datetime.now().isoformat(timespec='seconds'), **meta}
        with open(target, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        logger.info(f"Asal-usul checkpoint dicatat ke {target}")
    except Exception as e:  # noqa: BLE001 - catatan audit tidak boleh fatal
        logger.warning(f"Gagal mencatat asal-usul checkpoint ({e}); pelatihan tetap sah")


# ── Pelepasan memori setelah pelatihan ──────────────────────────────────────

def _rss_mb() -> Optional[float]:
    """RSS proses saat ini dalam MB, atau None di luar Linux."""
    try:
        with open('/proc/self/status', encoding='ascii') as f:
            for baris in f:
                if baris.startswith('VmRSS:'):
                    return round(int(baris.split()[1]) / 1024, 1)
    except OSError:
        return None
    return None


def lepas_memori() -> Dict[str, Any]:
    """
    Kembalikan memori sisa pelatihan ke sistem operasi.

    Setelah `_retrain_sync` selesai, optimizer AdamW (dua salinan seukuran
    bobot), gradien, dan model lama memang sudah tidak dirujuk lagi - Python
    membebaskannya. Tetapi alokator glibc tidak mengembalikan halaman yang
    bebas ke sistem operasi, apalagi dengan puluhan thread PyTorch yang
    masing-masing punya arena sendiri. Bagi kernel prosesnya tetap sebesar
    puncak pelatihan.

    Terukur di VPS produksi setelah dua putaran retraining berturut-turut:
    puncak RSS 5,8 GB, lalu proses menetap di 2,4 GB RAM ditambah 4,1 GB
    swap. `MemoryHigh` systemd menekannya 4,5 juta kali, dan evaluasi
    hold-out sesudahnya gagal karena tiap permintaan analisis aspek melewati
    batas 600 detik. Evaluasinya macet, bukan modelnya salah.

    `gc.collect()` memutus siklus rujukan lebih dulu, lalu `malloc_trim(0)`
    menyerahkan halaman bebas di ujung heap kembali ke kernel. Aman dipanggil
    kapan saja: tanpa glibc (Windows, macOS, musl) langkah kedua dilewati.
    """
    import ctypes
    import gc
    import sys

    sebelum = _rss_mb()
    gc.collect()

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - pelepasan memori tidak boleh fatal
        pass

    dipangkas = False
    if sys.platform.startswith('linux'):
        try:
            dipangkas = bool(ctypes.CDLL('libc.so.6').malloc_trim(0))
        except (OSError, AttributeError):
            dipangkas = False

    sesudah = _rss_mb()
    if sebelum is not None and sesudah is not None:
        logger.info(f"Memori setelah pelatihan dilepas: RSS {sebelum} MB -> {sesudah} MB")

    return {'rss_sebelum_mb': sebelum, 'rss_sesudah_mb': sesudah, 'malloc_trim': dipangkas}
