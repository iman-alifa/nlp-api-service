"""
Test untuk app/utils/training.py — seluruhnya berjalan tanpa memuat model,
sehingga bisa dipakai sebagai gerbang cepat sebelum menjalankan fine-tuning.
"""

import pytest

from app.utils.training import (
    classification_metrics,
    compute_class_weights,
    expected_calibration_error,
    fit_temperature,
    label_distribution,
    set_seed,
    should_accept_checkpoint,
    stratified_split,
    token_f1,
)


# ── Reproducibility ─────────────────────────────────────────────────────────

def test_set_seed_membuat_split_dapat_direproduksi():
    labels = ['negative'] * 30 + ['positive'] * 10 + ['neutral'] * 5

    set_seed(42)
    a_train, a_val = stratified_split(labels, 0.2, seed=42)
    set_seed(99)  # seed global berubah, tapi split memakai seed argumennya
    b_train, b_val = stratified_split(labels, 0.2, seed=42)

    assert a_train == b_train
    assert a_val == b_val


def test_seed_berbeda_menghasilkan_split_berbeda():
    labels = ['negative'] * 30 + ['positive'] * 20

    _, val_a = stratified_split(labels, 0.2, seed=1)
    _, val_b = stratified_split(labels, 0.2, seed=2)

    assert val_a != val_b


# ── Stratifikasi ────────────────────────────────────────────────────────────

def test_stratified_split_mempertahankan_semua_kelas_di_validasi():
    # Kasus nyata: data koreksi timpang, 78% negatif.
    labels = ['negative'] * 78 + ['positive'] * 15 + ['neutral'] * 7

    train_idx, val_idx = stratified_split(labels, 0.2, seed=42)

    val_labels = {labels[i] for i in val_idx}
    assert val_labels == {'negative', 'positive', 'neutral'}, \
        "setiap kelas harus terwakili di set validasi"


def test_stratified_split_tidak_tumpang_tindih_dan_lengkap():
    labels = ['a'] * 10 + ['b'] * 10

    train_idx, val_idx = stratified_split(labels, 0.2, seed=42)

    assert set(train_idx) & set(val_idx) == set()
    assert sorted(train_idx + val_idx) == list(range(20))


def test_stratified_split_kelas_satu_sampel_masuk_training():
    labels = ['a'] * 10 + ['langka']

    train_idx, val_idx = stratified_split(labels, 0.2, seed=42)

    assert 10 in train_idx, "kelas dengan satu sampel tidak boleh hilang"
    assert 10 not in val_idx


def test_stratified_split_data_kosong():
    assert stratified_split([], 0.2, 42) == ([], [])


# ── Distribusi label ────────────────────────────────────────────────────────

def test_label_distribution_menghitung_rasio_ketimpangan():
    labels = ['negative'] * 78 + ['positive'] * 15 + ['neutral'] * 7

    dist = label_distribution(labels)

    assert dist['total'] == 100
    assert dist['counts']['negative'] == 78
    assert dist['percentages']['negative'] == 78.0
    assert dist['majority_class'] == 'negative'
    assert dist['imbalance_ratio'] == pytest.approx(78 / 7, rel=1e-2)


def test_label_distribution_kosong():
    assert label_distribution([])['total'] == 0


# ── Bobot kelas ─────────────────────────────────────────────────────────────

def test_class_weights_lebih_besar_untuk_kelas_minoritas():
    # id: 0=positive, 1=neutral, 2=negative
    labels = [2] * 78 + [0] * 15 + [1] * 7

    weights = compute_class_weights(labels, num_classes=3)

    assert weights[1] > weights[0] > weights[2], \
        "kelas paling langka harus mendapat bobot terbesar"


def test_class_weights_kelas_absen_tidak_membagi_nol():
    weights = compute_class_weights([0, 0, 1], num_classes=3)

    assert weights[2] == 1.0
    assert all(w > 0 for w in weights)


def test_class_weights_data_kosong():
    assert compute_class_weights([], 3) == [1.0, 1.0, 1.0]


# ── Metrik klasifikasi ──────────────────────────────────────────────────────

def test_classification_metrics_prediksi_sempurna():
    y = [0, 1, 2, 0, 1, 2]

    m = classification_metrics(y, y, {0: 'positive', 1: 'neutral', 2: 'negative'})

    assert m['accuracy'] == 1.0
    assert m['macro_f1'] == 1.0
    assert m['weighted_f1'] == 1.0
    assert m['per_class']['positive']['recall'] == 1.0


def test_classification_metrics_menangkap_model_yang_menebak_satu_kelas():
    """
    Inilah gejala fine-tuning pada data timpang: akurasi tinggi tapi model
    tidak berguna. Accuracy 0.8 sementara macro F1 hanya ~0.44.
    """
    y_true = [2] * 8 + [0, 1]
    y_pred = [2] * 10

    m = classification_metrics(y_true, y_pred, {0: 'positive', 1: 'neutral', 2: 'negative'})

    assert m['accuracy'] == 0.8
    assert m['macro_f1'] < 0.5
    assert m['per_class']['positive']['recall'] == 0.0
    assert m['per_class']['neutral']['recall'] == 0.0


def test_classification_metrics_confusion_matrix():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]

    m = classification_metrics(y_true, y_pred, {0: 'positive', 1: 'neutral'})

    # baris = label sebenarnya, kolom = prediksi
    assert m['confusion_matrix'] == [[1, 1], [0, 2]]
    assert m['confusion_labels'] == ['positive', 'neutral']


def test_classification_metrics_panjang_tidak_sama_ditolak():
    with pytest.raises(ValueError):
        classification_metrics([0, 1], [0])


def test_classification_metrics_kosong():
    m = classification_metrics([], [])
    assert m['accuracy'] == 0.0
    assert m['support'] == 0


# ── F1 token BIO ────────────────────────────────────────────────────────────

def test_token_f1_mengabaikan_padding():
    y_true = [[-100, 1, 2, 0, -100]]
    y_pred = [[0, 1, 2, 0, 1]]  # posisi -100 tidak boleh dihitung

    m = token_f1(y_true, y_pred)

    assert m['precision'] == 1.0
    assert m['recall'] == 1.0
    assert m['f1'] == 1.0
    assert m['aspect_token_support'] == 2


def test_token_f1_model_semua_O_mendapat_nol():
    """
    Model yang memprediksi seluruh token sebagai 'O' terlihat >80% akurat,
    tapi F1 aspeknya harus 0 — inilah alasan metrik ini dipakai.
    """
    y_true = [[0, 0, 0, 0, 1, 2]]
    y_pred = [[0, 0, 0, 0, 0, 0]]

    m = token_f1(y_true, y_pred)

    assert m['f1'] == 0.0
    assert m['recall'] == 0.0
    assert m['aspect_token_support'] == 2


# ── Aturan penerimaan checkpoint ────────────────────────────────────────────

def test_checkpoint_diterima_saat_metrik_naik():
    accept, delta = should_accept_checkpoint(0.70, 0.82)

    assert accept is True
    assert delta == 0.12


def test_checkpoint_diterima_saat_metrik_sama():
    accept, delta = should_accept_checkpoint(0.80, 0.80)

    assert accept is True
    assert delta == 0.0


def test_checkpoint_ditolak_saat_metrik_turun():
    """Inti perlindungan: model tidak boleh ditimpa oleh hasil yang lebih buruk."""
    accept, delta = should_accept_checkpoint(0.85, 0.60)

    assert accept is False
    assert delta == -0.25


def test_penurunan_dalam_toleransi_diterima():
    accept, _ = should_accept_checkpoint(0.85, 0.83, min_delta=0.05)

    assert accept is True


def test_penurunan_melebihi_toleransi_ditolak():
    accept, _ = should_accept_checkpoint(0.85, 0.70, min_delta=0.05)

    assert accept is False


def test_force_menerima_walau_menurun():
    accept, delta = should_accept_checkpoint(0.90, 0.10, force=True)

    assert accept is True
    assert delta == -0.8


# ── Kalibrasi confidence ────────────────────────────────────────────────────

def test_ece_nol_saat_kalibrasi_sempurna():
    """Keyakinan 100% pada prediksi yang semuanya benar = tidak ada selisih."""
    assert expected_calibration_error([1.0, 1.0, 1.0], [True, True, True]) == 0.0


def test_ece_maksimum_saat_yakin_tapi_selalu_salah():
    assert expected_calibration_error([1.0, 1.0], [False, False]) == 1.0


def test_ece_menghukum_model_yang_terlalu_yakin():
    """Menyatakan 0,9 pada kasus yang benar separuhnya: selisih 0,4."""
    nilai = expected_calibration_error([0.9] * 10, [True] * 5 + [False] * 5)

    assert nilai == pytest.approx(0.4, abs=0.01)


def test_ece_data_kosong_tidak_membagi_nol():
    assert expected_calibration_error([], []) == 0.0


def test_ece_menolak_panjang_berbeda():
    with pytest.raises(ValueError):
        expected_calibration_error([0.9, 0.8], [True])


def test_ece_menghitung_confidence_nol():
    """
    Pita pertama harus inklusif di batas bawah, atau baris bertanda `empty`
    (confidence 0,0) hilang diam-diam dari perhitungan.
    """
    assert expected_calibration_error([0.0], [False]) == 0.0
    assert expected_calibration_error([0.0], [True]) == 1.0


def test_suhu_menurunkan_keyakinan_model_yang_terlalu_yakin():
    """
    Logit yang lebih besar dari yang dibenarkan datanya harus menghasilkan
    T > 1, yaitu instruksi "turunkan keyakinan".
    """
    torch = pytest.importorskip('torch')

    # 100 sampel, model selalu memilih kelas 0 dengan keyakinan ~0,95,
    # tetapi 15% di antaranya sebenarnya kelas 1.
    logits = torch.tensor([[3.0, 0.0]] * 100)
    labels = torch.tensor([0] * 85 + [1] * 15)

    suhu = fit_temperature(logits, labels)
    assert 1.0 < suhu <= 5.0


def test_suhu_dilewati_saat_validasi_terlalu_kecil():
    """
    Guo dkk. mengkalibrasi pada ribuan sampel. Data koreksi bisa jauh lebih
    kecil, dan mengalibrasi di situ hanya menyalin derau.
    """
    torch = pytest.importorskip('torch')

    logits = torch.tensor([[3.0, 0.0]] * 20)
    labels = torch.tensor([0] * 17 + [1] * 3)

    assert fit_temperature(logits, labels) == 1.0
    assert fit_temperature(logits, labels, min_samples=10) != 1.0


def test_suhu_dilewati_saat_validasi_benar_semua():
    """
    Regresi dari jalur retraining nyata: 19 sampel validasi yang semuanya benar
    menghasilkan T = 0,308 - bukan kalibrasi, melainkan model dibuat maksimal
    overconfident, yang justru melumpuhkan antrean peninjauan.

    Tanpa satu pun kesalahan, NLL terus turun selama keyakinan dinaikkan,
    sehingga optimasi meluncur ke T -> 0. Suhu tidak teridentifikasi di sana.
    """
    torch = pytest.importorskip('torch')

    logits = torch.tensor([[5.0, 0.0]] * 100)
    labels = torch.tensor([0] * 100)

    assert fit_temperature(logits, labels) == 1.0


def test_suhu_dilewati_saat_validasi_salah_semua():
    torch = pytest.importorskip('torch')

    logits = torch.tensor([[5.0, 0.0]] * 100)
    labels = torch.tensor([1] * 100)

    assert fit_temperature(logits, labels) == 1.0


def test_suhu_ekstrem_ditolak():
    """
    Model yang menyatakan 0,9997 pada kasus yang benar 70% memerlukan T ~ 9,4.
    Penyesuaian sebesar itu melampaui apa yang bisa dibenarkan set validasi
    seukuran data koreksi, jadi lebih aman tidak mengalibrasi sama sekali.
    """
    torch = pytest.importorskip('torch')

    logits = torch.tensor([[8.0, 0.0]] * 100)
    labels = torch.tensor([0] * 70 + [1] * 30)

    assert fit_temperature(logits, labels) == 1.0


def test_suhu_tidak_mengubah_argmax():
    """
    Sifat yang membuat temperature scaling aman dipakai: pembagian skalar
    positif tidak mengubah urutan logit, sehingga akurasi tidak bergeser.
    """
    torch = pytest.importorskip('torch')

    logits = torch.randn(200, 3) * 2
    labels = (logits.argmax(1) + torch.randint(0, 2, (200,))) % 3

    suhu = fit_temperature(logits, labels)

    assert torch.equal((logits / suhu).argmax(1), logits.argmax(1))


def test_suhu_kembali_satu_saat_data_kosong():
    """1.0 = tanpa kalibrasi, pilihan aman saat tidak ada yang bisa diukur."""
    torch = pytest.importorskip('torch')

    assert fit_temperature(torch.randn(1, 3), torch.tensor([0])) == 1.0
    assert fit_temperature(None, None) == 1.0


