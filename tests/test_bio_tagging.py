"""
Test pencocokan span aspek untuk pembuatan label BIO.

AspectService dikonstruksi tanpa memuat bobot model (lazy loading), dan
_find_aspect_spans murni operasi string, sehingga test ini tidak butuh
jaringan maupun GPU.
"""

import pytest

from app.services.aspect_service import AspectService


@pytest.fixture(scope='module')
def service(aspect_service_malas):
    """Alias ke fixture sesi di conftest.py - bobot dimuat sekali
    untuk seluruh suite, bukan ulang di tiap berkas."""
    return aspect_service_malas


def test_aspek_ditemukan_sebagai_kata_utuh(service):
    spans, unmatched = service._find_aspect_spans(
        "Pelayanannya bagus tapi harga mahal", ["harga"]
    )

    assert unmatched == []
    assert len(spans) == 1
    start, end = spans[0]
    assert "Pelayanannya bagus tapi harga mahal"[start:end].lower() == "harga"


def test_kata_berimbuhan_cocok_lewat_bentuk_dasar(service):
    """
    'pelayanan' dikoreksi user untuk kalimat yang menulis 'layanannya'.
    Pencocokan literal melewatkannya sehingga sampel berlabel seluruhnya 'O';
    pencocokan bentuk dasar (Sastrawi) menyelamatkannya.
    """
    spans, unmatched = service._find_aspect_spans(
        "Layanannya lambat sekali", ["pelayanan"]
    )

    assert unmatched == []
    assert len(spans) == 1


def test_substring_tidak_cocok_secara_literal(service):
    """'harga' ada di dalam 'seharganya' tapi bukan kata utuh — literal harus gagal."""
    spans, unmatched = service._find_aspect_spans(
        "Seharganya wajar", ["harga"], use_stemming=False
    )

    assert spans == []
    assert unmatched == ["harga"]


def test_turunan_verbal_ditolak(service):
    """
    'menghargai' berakar sama dengan 'harga' tetapi merupakan predikat.
    Menandainya sebagai aspek akan mengajari model melabeli verba.
    """
    for teks in ("Kita harus menghargai jasa mereka", "Barang ini berharga sekali"):
        spans, unmatched = service._find_aspect_spans(teks, ["harga"])
        assert spans == [], f"tidak boleh cocok pada: {teks}"
        assert unmatched == ["harga"]


def test_kemunculan_berulang_semuanya_tertangkap(service):
    spans, unmatched = service._find_aspect_spans(
        "harga naik, harga turun, harga stabil", ["harga"]
    )

    assert unmatched == []
    assert len(spans) == 3


def test_aspek_multi_kata(service):
    text = "Kualitas produk bagus"
    spans, unmatched = service._find_aspect_spans(text, ["kualitas produk"])

    assert unmatched == []
    start, end = spans[0]
    assert text[start:end].lower() == "kualitas produk"


def test_aspek_terpanjang_diproses_lebih_dulu(service):
    """Aspek tumpang tindih: yang lebih panjang harus menang."""
    text = "Kualitas produk sangat baik"
    spans, unmatched = service._find_aspect_spans(text, ["kualitas", "kualitas produk"])

    assert unmatched == []
    # span terpanjang ada di urutan pertama
    first_start, first_end = spans[0]
    assert text[first_start:first_end].lower() == "kualitas produk"


def test_pencocokan_tidak_peka_huruf_besar(service):
    spans, unmatched = service._find_aspect_spans("HARGA nya mahal", ["harga"])

    assert unmatched == []
    assert len(spans) == 1


def test_aspek_dengan_tanda_baca_dinormalisasi(service):
    # _normalize_aspect_text membuang tanda baca di ujung
    spans, unmatched = service._find_aspect_spans("Soal pajak memang berat", ["pajak."])

    assert unmatched == []
    assert len(spans) == 1


def test_campuran_literal_dan_bentuk_dasar(service):
    """Satu aspek cocok literal, satu lagi lewat bentuk dasar."""
    spans, unmatched = service._find_aspect_spans(
        "harga mahal tapi layanannya oke", ["harga", "pelayanan"]
    )

    assert unmatched == []
    assert len(spans) == 2


def test_daftar_aspek_kosong(service):
    spans, unmatched = service._find_aspect_spans("teks apa pun", [])

    assert spans == []
    assert unmatched == []


# ── Pencocokan toleran-imbuhan (Sastrawi, khusus aspek) ─────────────────────

def test_imbuhan_cocok_lewat_stemming(service):
    """
    'pejabat' vs 'pejabatnya' adalah kasus paling sering di data koreksi nyata;
    pencocokan literal melewatkannya dan membuat sampel berlabel semua-'O'.
    """
    text = "Berikan kami contoh pejabatnya juga harus majak"
    spans, unmatched = service._find_aspect_spans(text, ["pejabat"])

    assert unmatched == []
    start, end = spans[0]
    assert text[start:end].lower() == "pejabatnya"


def test_span_menunjuk_kata_utuh_di_teks_asli(service):
    """Span harus memetakan ke teks mentah, bukan ke bentuk dasarnya."""
    text = "layanannya sangat lambat"
    spans, _ = service._find_aspect_spans(text, ["pelayanan"])

    start, end = spans[0]
    assert text[start:end] == "layanannya"


def test_stemming_bisa_dimatikan(service):
    _, unmatched = service._find_aspect_spans(
        "pejabatnya korupsi", ["pejabat"], use_stemming=False
    )

    assert unmatched == ["pejabat"]


def test_pencocokan_literal_diprioritaskan(service):
    """Bila bentuk literal ada, itu yang dipakai — stemming hanya cadangan."""
    text = "pajak naik lagi"
    spans, unmatched = service._find_aspect_spans(text, ["pajak"])

    assert unmatched == []
    assert len(spans) == 1
    assert text[spans[0][0]:spans[0][1]] == "pajak"


def test_ketidakcocokan_semantik_tetap_dilaporkan(service):
    """
    Stemming tidak boleh menutupi masalah anotasi sungguhan: 'dpr' untuk
    kalimat yang menulis 'komisi xi' memang harus muncul sebagai unmatched.
    """
    _, unmatched = service._find_aspect_spans("komisi xi ini hanya modus", ["dpr"])

    assert unmatched == ["dpr"]


def test_stem_cache_dipakai_ulang(service):
    service._find_aspect_spans("pejabatnya korupsi", ["pejabat"])

    assert len(service._stem_cache) > 0
