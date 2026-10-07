"""
Metrik mutu topic modeling: coherence, diversity, dan cakupan.

Modul ini murni fungsi - tidak memuat model, tidak menyentuh jaringan - supaya
bisa diuji unit dan dipakai baik oleh service maupun skrip eksperimen.

**Kenapa diimplementasikan sendiri, bukan memakai gensim.** `gensim==4.3.2` ada
di requirements.txt tetapi TIDAK BISA diimpor pada environment ini:
`cannot import name 'triu' from 'scipy.linalg'` - gensim 4.3.2 belum kompatibel
dengan scipy >= 1.13. Menurunkan versi scipy akan menyeret scikit-learn dan
BERTopic ikut turun, dan menaikkan gensim berarti menambah ~30 MB ke image
Railway untuk dua fungsi saja. Rumusnya sendiri pendek, jadi ditulis langsung.

Rujukan:
- Röder, Both & Hinneburg (2015), "Exploring the Space of Topic Coherence
  Measures" - definisi C_v dan C_NPMI yang dipakai di sini.
- Dieng, Ruiz & Blei (2020), "Topic Modeling in Embedding Spaces" - topic
  diversity.
"""

import math
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np

# Röder dkk. menemukan C_v (jendela 110) paling berkorelasi dengan penilaian
# manusia, sedangkan C_NPMI lazim dilaporkan dengan jendela 10. Keduanya
# dipertahankan supaya angka di skripsi bisa dibandingkan dengan literatur.
CV_WINDOW_SIZE = 110
NPMI_WINDOW_SIZE = 10

# Mencegah log(0) pada pasangan kata yang tidak pernah muncul bersama.
_EPSILON = 1e-12


# ── Statistik kemunculan ────────────────────────────────────────────────────

def _iter_windows(
    tokenized_docs: Sequence[Sequence[str]],
    window_size: int,
) -> Iterable[Set[str]]:
    """
    Hasilkan jendela geser boolean sebagai himpunan token.

    Dokumen yang lebih pendek dari jendela menghasilkan satu jendela berisi
    seluruh dokumen - perilaku baku pada literatur coherence, dan penting di
    sini karena korpus produksi berupa komentar pendek.

    Args:
        tokenized_docs: Dokumen yang sudah ditokenisasi.
        window_size: Lebar jendela dalam token; <= 0 berarti seluruh dokumen.

    Yields:
        Himpunan token unik pada tiap jendela.
    """
    for tokens in tokenized_docs:
        if not tokens:
            continue
        if window_size <= 0 or len(tokens) <= window_size:
            yield set(tokens)
            continue
        for start in range(len(tokens) - window_size + 1):
            yield set(tokens[start:start + window_size])


def cooccurrence_stats(
    tokenized_docs: Sequence[Sequence[str]],
    vocabulary: Set[str],
    window_size: int,
) -> Tuple[Dict[str, int], Dict[Tuple[str, str], int], int]:
    """
    Hitung frekuensi jendela untuk kata tunggal dan pasangan kata.

    Hanya kata di dalam `vocabulary` yang dilacak; pada praktiknya itu berarti
    gabungan kata-kata teratas seluruh topik, sehingga biayanya kecil meski
    korpusnya besar.

    Args:
        tokenized_docs: Dokumen yang sudah ditokenisasi.
        vocabulary: Kata yang perlu dihitung.
        window_size: Lebar jendela geser.

    Returns:
        (jumlah per kata, jumlah per pasangan terurut, total jendela).
    """
    word_counts: Dict[str, int] = {w: 0 for w in vocabulary}
    pair_counts: Dict[Tuple[str, str], int] = {}
    total_windows = 0

    for window in _iter_windows(tokenized_docs, window_size):
        total_windows += 1
        present = sorted(window & vocabulary)
        for i, word in enumerate(present):
            word_counts[word] += 1
            for other in present[i + 1:]:
                key = (word, other)
                pair_counts[key] = pair_counts.get(key, 0) + 1

    return word_counts, pair_counts, total_windows


def _npmi(
    left: str,
    right: str,
    word_counts: Dict[str, int],
    pair_counts: Dict[Tuple[str, str], int],
    total_windows: int,
) -> float:
    """
    NPMI satu pasangan kata, dalam rentang [-1, 1].

    Nilai 1 berarti selalu muncul bersama, 0 berarti independen, -1 berarti
    tidak pernah bersama.
    """
    if total_windows == 0:
        return 0.0
    if left == right:
        return 1.0

    key = (left, right) if left < right else (right, left)
    joint = pair_counts.get(key, 0) / total_windows
    p_left = word_counts.get(left, 0) / total_windows
    p_right = word_counts.get(right, 0) / total_windows

    if p_left <= 0 or p_right <= 0:
        return 0.0

    # Kasus degenerate: kedua kata muncul di SETIAP jendela. Maka -log(p) = 0
    # dan rumusnya menjadi 0/0. Secara limit asosiasinya sempurna, jadi 1,0.
    # Tanpa penanganan ini epsilon membalik tandanya menjadi -1,0 - persis
    # kebalikan dari yang benar, dan senyap karena nilainya tetap "wajar".
    if joint >= 1.0 - _EPSILON:
        return 1.0

    numerator = math.log((joint + _EPSILON) / (p_left * p_right))
    denominator = -math.log(joint + _EPSILON)
    if denominator == 0:
        return 0.0
    return numerator / denominator


# ── Coherence ───────────────────────────────────────────────────────────────

def npmi_coherence(
    topic_words: Sequence[Sequence[str]],
    tokenized_docs: Sequence[Sequence[str]],
    top_n: int = 10,
    window_size: int = NPMI_WINDOW_SIZE,
) -> Tuple[float, List[float]]:
    """
    C_NPMI: rata-rata NPMI seluruh pasangan kata teratas tiap topik.

    Args:
        topic_words: Kata teratas per topik, sudah terurut.
        tokenized_docs: Korpus rujukan yang sudah ditokenisasi.
        top_n: Berapa kata teratas yang dinilai.
        window_size: Lebar jendela geser.

    Returns:
        (rata-rata seluruh topik, skor per topik). Topik dengan kurang dari dua
        kata diberi 0,0 agar panjang keluaran tetap sejajar dengan masukan.
    """
    trimmed = [list(words)[:top_n] for words in topic_words]
    vocabulary = {w for words in trimmed for w in words}
    if not vocabulary:
        return 0.0, [0.0 for _ in trimmed]

    word_counts, pair_counts, total = cooccurrence_stats(
        tokenized_docs, vocabulary, window_size
    )

    per_topic: List[float] = []
    for words in trimmed:
        pairs = [
            _npmi(words[i], words[j], word_counts, pair_counts, total)
            for i in range(len(words))
            for j in range(i + 1, len(words))
        ]
        per_topic.append(float(np.mean(pairs)) if pairs else 0.0)

    overall = float(np.mean(per_topic)) if per_topic else 0.0
    return overall, per_topic


def cv_coherence(
    topic_words: Sequence[Sequence[str]],
    tokenized_docs: Sequence[Sequence[str]],
    top_n: int = 10,
    window_size: int = CV_WINDOW_SIZE,
) -> Tuple[float, List[float]]:
    """
    C_v sesuai Röder dkk. (2015).

    Rangkaiannya: segmentasi S_one_set (tiap kata dilawankan dengan seluruh
    himpunan kata topik), estimasi peluang lewat jendela geser boolean 110,
    konfirmasi memakai kosinus antar-vektor NPMI, lalu dirata-rata.

    Berbeda dari C_NPMI yang menilai pasangan kata secara langsung, C_v menilai
    kemiripan POLA kemunculan: dua kata yang tak pernah bersebelahan tetapi
    muncul pada konteks yang mirip tetap dinilai koheren. Itu sebabnya C_v yang
    paling berkorelasi dengan penilaian manusia.

    Args:
        topic_words: Kata teratas per topik.
        tokenized_docs: Korpus rujukan yang sudah ditokenisasi.
        top_n: Berapa kata teratas yang dinilai.
        window_size: Lebar jendela geser.

    Returns:
        (rata-rata seluruh topik, skor per topik).
    """
    trimmed = [list(words)[:top_n] for words in topic_words]
    vocabulary = {w for words in trimmed for w in words}
    if not vocabulary:
        return 0.0, [0.0 for _ in trimmed]

    word_counts, pair_counts, total = cooccurrence_stats(
        tokenized_docs, vocabulary, window_size
    )

    per_topic: List[float] = []
    for words in trimmed:
        if len(words) < 2:
            per_topic.append(0.0)
            continue

        # Vektor konteks tiap kata terhadap seluruh kata topik.
        vectors = np.array([
            [_npmi(w, other, word_counts, pair_counts, total) for other in words]
            for w in words
        ])
        # Vektor himpunan: jumlah seluruh vektor kata.
        set_vector = vectors.sum(axis=0)

        similarities = []
        for vector in vectors:
            denominator = np.linalg.norm(vector) * np.linalg.norm(set_vector)
            similarities.append(
                float(np.dot(vector, set_vector) / denominator)
                if denominator > 0 else 0.0
            )
        per_topic.append(float(np.mean(similarities)))

    overall = float(np.mean(per_topic)) if per_topic else 0.0
    return overall, per_topic


# ── Diversity & cakupan ─────────────────────────────────────────────────────

def topic_diversity(
    topic_words: Sequence[Sequence[str]],
    top_n: int = 25,
) -> float:
    """
    Proporsi kata unik di antara kata teratas seluruh topik (Dieng dkk. 2020).

    Nilai 1,0 berarti tidak ada kata yang dipakai bersama dua topik; nilai
    rendah menandakan topik-topik saling menyalin kata dan sebenarnya tidak
    terpisah. Coherence saja tidak cukup: model yang mengulang kata yang sama
    di semua topik bisa mendapat coherence tinggi tetapi tak berguna.

    Args:
        topic_words: Kata teratas per topik.
        top_n: Berapa kata teratas per topik yang dihitung.

    Returns:
        Rasio dalam [0, 1]; 0,0 bila tidak ada topik.
    """
    trimmed = [list(words)[:top_n] for words in topic_words]
    total = sum(len(words) for words in trimmed)
    if total == 0:
        return 0.0
    unique = len({w for words in trimmed for w in words})
    return unique / total


def outlier_rate(document_topics: Sequence[int]) -> float:
    """
    Proporsi dokumen yang tidak masuk topik mana pun (ditandai -1).

    Angka ini penting di sistem ini karena dokumen outlier juga hilang dari
    perhitungan PMI aspek-topik: setiap dokumen -1 adalah data yang terbuang.

    Args:
        document_topics: Id topik per dokumen, sejajar dengan teks masukan.

    Returns:
        Rasio dalam [0, 1]; 0,0 bila daftarnya kosong.
    """
    if not document_topics:
        return 0.0
    return sum(1 for t in document_topics if t == -1) / len(document_topics)


def evaluate(
    topic_words: Sequence[Sequence[str]],
    tokenized_docs: Sequence[Sequence[str]],
    document_topics: Sequence[int],
    top_n: int = 10,
) -> Dict[str, float]:
    """
    Kumpulkan seluruh metrik dalam satu panggilan.

    Args:
        topic_words: Kata teratas per topik.
        tokenized_docs: Korpus rujukan yang sudah ditokenisasi.
        document_topics: Id topik per dokumen.
        top_n: Berapa kata teratas yang dinilai untuk coherence.

    Returns:
        Dict berisi c_v, c_npmi, diversity, outlier_rate, dan num_topics.
    """
    cv, _ = cv_coherence(topic_words, tokenized_docs, top_n=top_n)
    npmi_score, _ = npmi_coherence(topic_words, tokenized_docs, top_n=top_n)

    return {
        'c_v': round(cv, 4),
        'c_npmi': round(npmi_score, 4),
        'diversity': round(topic_diversity(topic_words), 4),
        'outlier_rate': round(outlier_rate(document_topics), 4),
        'num_topics': len(topic_words),
    }
