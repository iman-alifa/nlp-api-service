"""
Kontrak keselarasan indeks: N teks masuk, N hasil keluar.

AssociationService memetakan aspek ke topik berdasarkan POSISI, dan Laravel
memetakan hasil kembali ke raw_data dengan cara yang sama. Bila panjangnya
bergeser, setiap baris setelah teks kosong tersimpan dengan teks yang salah —
tanpa error apa pun.
"""

import pytest

from app.schemas.request_schemas import TextAnalysisRequest
from app.services.aspect_service import AspectService
from app.services.association_service import AssociationService


# ── Schema mempertahankan jumlah elemen ─────────────────────────────────────

def test_teks_kosong_tidak_dibuang_dari_daftar():
    req = TextAnalysisRequest(texts=["Harga mahal", "   ", "Pelayanan bagus"])

    assert len(req.texts) == 3
    assert req.texts[1] == ''


def test_spasi_dirapikan_tanpa_mengubah_jumlah():
    req = TextAnalysisRequest(texts=["  Harga mahal  ", "\tPelayanan\n"])

    assert req.texts == ['Harga mahal', 'Pelayanan']


def test_semua_kosong_tetap_ditolak():
    with pytest.raises(Exception):
        TextAnalysisRequest(texts=["   ", "", "\t"])


def test_daftar_kosong_ditolak():
    with pytest.raises(Exception):
        TextAnalysisRequest(texts=[])


# ── document_aspects sepanjang input ────────────────────────────────────────

@pytest.fixture(scope='module')
def service(aspect_service_malas):
    """Alias ke fixture sesi di conftest.py - bobot dimuat sekali
    untuk seluruh suite, bukan ulang di tiap berkas."""
    return aspect_service_malas


def test_document_aspects_sepanjang_input(service):
    aspects = [{
        'aspect': 'harga',
        'occurrences': [{'text_index': 0}, {'text_index': 2}],
    }]

    hasil = service._build_document_aspects(aspects, num_texts=4)

    assert len(hasil) == 4
    assert hasil == [['harga'], [], ['harga'], []]


def test_aspek_berulang_dalam_satu_dokumen_dicatat_sekali(service):
    """count tetap menyimpan frekuensinya; document_aspects cukup daftar unik."""
    aspects = [{
        'aspect': 'pelayanan',
        'occurrences': [{'text_index': 0}] * 12,
    }]

    assert service._build_document_aspects(aspects, num_texts=1) == [['pelayanan']]


def test_beberapa_aspek_berbeda_dipertahankan(service):
    aspects = [
        {'aspect': 'harga', 'occurrences': [{'text_index': 0}]},
        {'aspect': 'pelayanan', 'occurrences': [{'text_index': 0}]},
    ]

    assert service._build_document_aspects(aspects, num_texts=1) == [['harga', 'pelayanan']]


def test_indeks_di_luar_jangkauan_diabaikan(service):
    aspects = [{'aspect': 'harga', 'occurrences': [{'text_index': 99}, {'text_index': -1}]}]

    assert service._build_document_aspects(aspects, num_texts=2) == [[], []]


# ── Rantai ke AssociationService ────────────────────────────────────────────

def test_panjang_sejajar_membuat_pmi_bisa_dihitung():
    """
    Inilah rantai yang putus dulu: panjang berbeda -> {'error': 'Length mismatch'}.
    """
    doc_aspects = [['harga'], [], ['harga'], ['pelayanan'], ['harga'], ['pelayanan']]
    doc_topics = [0, -1, 0, 1, 0, 1]

    hasil = AssociationService().analyze(doc_aspects, doc_topics, num_topics=2, min_mentions=2)

    assert 'error' not in hasil
    assert hasil['pmi_top_associations']


def test_panjang_berbeda_terdeteksi_sebagai_error():
    hasil = AssociationService().analyze([['harga']], [0, 1], num_topics=2)

    assert hasil == {'error': 'Length mismatch'}


def test_dokumen_kosong_dihitung_sebagai_outlier():
    """Teks kosong mendapat -1 dan tidak boleh mengacaukan perhitungan."""
    doc_aspects = [['harga'], [], ['harga'], ['harga']]
    doc_topics = [0, -1, 0, 0]

    hasil = AssociationService().analyze(doc_aspects, doc_topics, num_topics=1, min_mentions=3)

    assert 'error' not in hasil
