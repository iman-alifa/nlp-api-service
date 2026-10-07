"""
Test aturan anotasi aspek otomatis (scripts/annotate_aspects.py).

Aturan-aturan inilah yang memisahkan anotasi setara-manusia dari keluaran LLM
mentah, jadi masing-masing dikunci dengan test.
"""

import importlib.util
import os
import sys

import pytest

_spec = importlib.util.spec_from_file_location(
    "annotate_aspects",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "scripts", "annotate_aspects.py"),
)
ann = importlib.util.module_from_spec(_spec)
sys.modules["annotate_aspects"] = ann
_spec.loader.exec_module(ann)


# ── Normalisasi istilah ─────────────────────────────────────────────────────

def test_kurung_penjelas_dibuang():
    assert ann.normalize_term("kantor akuntan publik ( KAP )") == "kantor akuntan publik"


def test_kurung_tidak_tertutup_dibuang():
    assert ann.normalize_term("layanan data ( broadband") == "layanan data"


def test_tahun_di_ekor_dibuang():
    assert ann.normalize_term("kinerja 2003") == "kinerja"


def test_satuan_waktu_di_ekor_dibuang():
    assert ann.normalize_term("kinerja tahun") == "kinerja"


def test_frasa_kepanjangan_dipangkas():
    hasil = ann.normalize_term("kode akses percakapan Sambungan Langsung Jarak Jauh")
    assert len(hasil.split()) <= 4


def test_tanda_baca_ujung_dibersihkan():
    assert ann.normalize_term("  saham publik , ") == "saham publik"


def test_istilah_kosong_jadi_none():
    assert ann.normalize_term("   ") is None
    assert ann.normalize_term("( )") is None


# ── Deteksi verba ───────────────────────────────────────────────────────────

def test_verba_berimbuhan_dikenali():
    for kata in ("merelokasi", "menghapuskan", "dipilih", "bertanding"):
        assert ann.is_verbal(kata), kata


def test_nomina_pengecualian_tidak_dikira_verba():
    """'Mentari' (merek) jangan dikira men- + 'tari'."""
    for kata in ("mentari", "menteri", "direksi", "terminal", "berita"):
        assert not ann.is_verbal(kata), kata


# ── Penolakan kandidat ──────────────────────────────────────────────────────

VOCAB = {"saham", "kinerja", "perusahaan", "pemerintah", "harga", "layanan"}


def test_frasa_berkepala_verba_ditolak():
    assert ann.reject_reason("merelokasi StarOne", VOCAB) == "berkepala verba"


def test_frasa_berkepala_kata_opini_ditolak():
    """'lolos' dan 'menang' adalah opini/predikat, bukan target opini."""
    assert ann.reject_reason("lolos ke Liga Champions", VOCAB) == "berkepala kata opini"


def test_frasa_berkepala_kata_fungsi_ditolak():
    assert ann.reject_reason("tak terkalahkan", VOCAB) == "berkepala kata fungsi"
    assert ann.reject_reason("yang terbaik", VOCAB) == "berkepala kata fungsi"


def test_istilah_berangka_ditolak():
    assert ann.reject_reason("layanan 3G", VOCAB) == "mengandung angka"


def test_nama_diri_ditolak():
    """NER bukan ekstraksi aspek; melatihnya mengajari model menandai kapital."""
    assert ann.reject_reason("Manchester City", VOCAB) == "nama diri"


def test_akronim_konsep_diterima():
    """RUPS/BUMN nomina umum di domainnya, berbeda dari nama klub/orang."""
    for akronim in ("RUPS", "BUMN", "KAP", "PPN"):
        assert ann.reject_reason(akronim, VOCAB) is None, akronim


def test_nomina_umum_diterima():
    for term in ("saham publik", "kinerja perusahaan", "harga"):
        assert ann.reject_reason(term, VOCAB) is None, term


def test_kata_berkapital_yang_juga_muncul_huruf_kecil_diterima():
    """'Perusahaan' di awal kalimat tetap nomina umum."""
    assert ann.reject_reason("Perusahaan", VOCAB) is None


# ── Proyeksi konsisten ──────────────────────────────────────────────────────

def test_semua_kemunculan_ditandai():
    """
    Cacat utama anotasi lama: str.find() hanya menandai kemunculan pertama,
    sehingga 56% kemunculan terlewat dan supervisinya saling bertentangan.
    """
    pola = ann.compile_patterns(["saham"])
    spans = ann.project("saham naik lalu saham turun, saham lagi", pola)

    assert len(spans) == 3


def test_pencocokan_tidak_peka_huruf_besar():
    pola = ann.compile_patterns(["saham"])
    assert len(ann.project("Saham dan SAHAM", pola)) == 2


def test_batas_kata_dihormati():
    pola = ann.compile_patterns(["harga"])
    assert ann.project("seharganya wajar", pola) == []


def test_span_terpanjang_menang():
    pola = ann.compile_patterns(["saham", "saham publik"])
    spans = ann.project("membeli saham publik hari ini", pola)

    assert len(spans) == 1
    assert spans[0][2] == "saham publik"


def test_frasa_multikata_dengan_spasi_ganda():
    pola = ann.compile_patterns(["kerja sama"])
    assert len(ann.project("melakukan kerja  sama dengan", pola)) == 1
