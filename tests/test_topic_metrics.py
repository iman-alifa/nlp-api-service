"""
Metrik mutu topic modeling.

Metrik ini dipakai untuk MEMUTUSKAN parameter mana yang dipakai di produksi dan
angka mana yang masuk skripsi, jadi metriknya sendiri harus diuji lebih dulu.
Semua tes memakai korpus mini buatan yang jawabannya bisa dihitung tangan -
tidak ada bobot model yang dimuat.
"""

import math

import pytest

from app.utils.topic_metrics import (
    cooccurrence_stats,
    cv_coherence,
    evaluate,
    npmi_coherence,
    outlier_rate,
    topic_diversity,
)


# ── Statistik jendela ───────────────────────────────────────────────────────

def test_dokumen_lebih_pendek_dari_jendela_jadi_satu_jendela():
    """Korpus di sini berupa komentar pendek; ini jalur yang paling sering."""
    docs = [['a', 'b', 'c']]

    counts, pairs, total = cooccurrence_stats(docs, {'a', 'b', 'c'}, 110)

    assert total == 1
    assert counts == {'a': 1, 'b': 1, 'c': 1}
    assert pairs[('a', 'b')] == 1


def test_jendela_geser_memecah_dokumen_panjang():
    docs = [['a', 'b', 'c', 'd']]

    _, _, total = cooccurrence_stats(docs, {'a', 'b', 'c', 'd'}, 2)

    assert total == 3, 'jendela lebar 2 atas 4 token menghasilkan 3 posisi'


def test_kata_di_luar_kosakata_diabaikan():
    docs = [['a', 'z', 'b']]

    counts, _, _ = cooccurrence_stats(docs, {'a', 'b'}, 0)

    assert counts == {'a': 1, 'b': 1}
    assert 'z' not in counts


def test_dokumen_kosong_tidak_menambah_jendela():
    _, _, total = cooccurrence_stats([[], ['a']], {'a'}, 0)

    assert total == 1


# ── NPMI ────────────────────────────────────────────────────────────────────

def test_npmi_maksimum_saat_selalu_bersama():
    """Dua kata yang selalu muncul berbarengan harus mendekati 1,0."""
    docs = [['a', 'b'] for _ in range(20)]

    skor, per_topik = npmi_coherence([['a', 'b']], docs, window_size=0)

    assert skor == pytest.approx(1.0, abs=0.02)
    assert per_topik == [pytest.approx(1.0, abs=0.02)]


def test_npmi_minimum_saat_tidak_pernah_bersama():
    docs = [['a', 'x'] for _ in range(10)] + [['b', 'y'] for _ in range(10)]

    skor, _ = npmi_coherence([['a', 'b']], docs, window_size=0)

    assert skor < -0.9, 'kata yang tak pernah bersama harus mendekati -1'


def test_npmi_lebih_tinggi_untuk_topik_yang_benar_benar_berkaitan():
    """Uji pembeda: topik koheren harus mengungguli topik acak."""
    docs = (
        [['pajak', 'bayar', 'negara'] for _ in range(15)]
        + [['jalan', 'rusak', 'jembatan'] for _ in range(15)]
    )

    koheren, _ = npmi_coherence([['pajak', 'bayar', 'negara']], docs, window_size=0)
    campuran, _ = npmi_coherence([['pajak', 'rusak', 'jembatan']], docs, window_size=0)

    assert koheren > campuran


def test_topik_berkata_tunggal_tidak_meledak():
    skor, per_topik = npmi_coherence([['a']], [['a', 'b']], window_size=0)

    assert skor == 0.0
    assert per_topik == [0.0]


def test_korpus_kosong_aman():
    skor, per_topik = npmi_coherence([['a', 'b']], [], window_size=0)

    assert math.isfinite(skor)
    assert len(per_topik) == 1


# ── C_v ─────────────────────────────────────────────────────────────────────

def test_cv_membedakan_topik_koheren_dari_topik_acak():
    docs = (
        [['pajak', 'bayar', 'negara', 'uang'] for _ in range(15)]
        + [['jalan', 'rusak', 'jembatan', 'akses'] for _ in range(15)]
    )

    koheren, _ = cv_coherence([['pajak', 'bayar', 'negara']], docs)
    campuran, _ = cv_coherence([['pajak', 'rusak', 'akses']], docs)

    assert koheren > campuran


def test_cv_berada_dalam_rentang_wajar():
    docs = [['a', 'b', 'c'] for _ in range(10)]

    skor, _ = cv_coherence([['a', 'b', 'c']], docs)

    assert -1.0 <= skor <= 1.0


def test_cv_mengembalikan_satu_skor_per_topik():
    docs = [['a', 'b'], ['c', 'd']]

    _, per_topik = cv_coherence([['a', 'b'], ['c', 'd'], ['a', 'c']], docs)

    assert len(per_topik) == 3


# ── Diversity ───────────────────────────────────────────────────────────────

def test_diversity_satu_saat_tidak_ada_kata_berbagi():
    assert topic_diversity([['a', 'b'], ['c', 'd']]) == 1.0


def test_diversity_setengah_saat_topik_identik():
    assert topic_diversity([['a', 'b'], ['a', 'b']]) == 0.5


def test_diversity_menghukum_topik_yang_menyalin_kata():
    """Coherence tinggi tidak berguna bila semua topik memuat kata yang sama."""
    beragam = topic_diversity([['a', 'b'], ['c', 'd'], ['e', 'f']])
    seragam = topic_diversity([['a', 'b'], ['a', 'b'], ['a', 'b']])

    assert beragam > seragam


def test_diversity_tanpa_topik_bernilai_nol():
    assert topic_diversity([]) == 0.0


# ── Outlier ─────────────────────────────────────────────────────────────────

def test_outlier_rate_menghitung_minus_satu():
    assert outlier_rate([0, 1, -1, -1]) == 0.5


def test_outlier_rate_daftar_kosong_nol():
    assert outlier_rate([]) == 0.0


# ── Agregat ─────────────────────────────────────────────────────────────────

def test_evaluate_mengembalikan_seluruh_metrik():
    docs = [['pajak', 'bayar'], ['jalan', 'rusak']]

    hasil = evaluate([['pajak', 'bayar'], ['jalan', 'rusak']], docs, [0, 1])

    assert set(hasil) == {'c_v', 'c_npmi', 'diversity', 'outlier_rate', 'num_topics'}
    assert hasil['num_topics'] == 2
    assert hasil['outlier_rate'] == 0.0
    assert hasil['diversity'] == 1.0
