"""
Test AssociationService (PMI aspek-topik).

Tidak memuat model apa pun — murni aritmetika, sehingga cepat dijalankan.
"""

import math

import pytest

from app.services.association_service import AssociationService


@pytest.fixture
def service():
    return AssociationService()


# Dokumen 0,1,4 -> topik 0 ; dokumen 2,3,5 -> topik 1
DOC_TOPICS = [0, 0, 1, 1, 0, 1]
DOC_ASPECTS_STR = [
    ['harga', 'pelayanan'],
    ['harga'],
    ['pelayanan'],
    ['harga'],
    ['harga'],
    ['pelayanan'],
]


def test_menerima_daftar_string(service):
    """
    Bentuk yang benar-benar dikirim AspectService._build_document_aspects.
    Sebelum diperbaiki, jalur ini melempar AttributeError dan seluruh blok
    association berubah menjadi {'error': ...} tanpa terlihat di Laravel.
    """
    result = service.analyze(DOC_ASPECTS_STR, DOC_TOPICS, num_topics=2, min_mentions=2)

    assert 'error' not in result
    assert result['pmi_top_associations'], "harus ada asosiasi bernilai PMI positif"


def test_menerima_daftar_dict(service):
    doc_aspects = [[{'aspect': a} for a in row] for row in DOC_ASPECTS_STR]

    result = service.analyze(doc_aspects, DOC_TOPICS, num_topics=2, min_mentions=2)

    assert 'error' not in result
    assert result['pmi_top_associations']


def test_kedua_bentuk_menghasilkan_angka_identik(service):
    doc_aspects_dict = [[{'aspect': a} for a in row] for row in DOC_ASPECTS_STR]

    a = service.analyze(DOC_ASPECTS_STR, DOC_TOPICS, 2, 2)
    b = service.analyze(doc_aspects_dict, DOC_TOPICS, 2, 2)

    assert a['pmi_top_associations'] == b['pmi_top_associations']
    assert a['heatmap_matrix'] == b['heatmap_matrix']


def test_nilai_pmi_sesuai_rumus(service):
    """PMI(a,t) = log2( P(a,t) / (P(a) * P(t)) )."""
    result = service.analyze(DOC_ASPECTS_STR, DOC_TOPICS, 2, 2)

    harga = next(r for r in result['pmi_top_associations']
                 if r['aspect'] == 'harga' and r['topic_id'] == 0)

    # 'harga' muncul di dokumen 0,1,3,4 -> P(a) = 4/6
    # topik 0 berisi dokumen 0,1,4        -> P(t) = 3/6
    # ko-okurensi: dokumen 0,1,4          -> P(a,t) = 3/6
    expected = math.log2((3 / 6) / ((4 / 6) * (3 / 6)))

    assert harga['pmi'] == pytest.approx(expected, abs=1e-3)
    assert harga['co_occurrences'] == 3


def test_panjang_tidak_sama_dilaporkan(service):
    result = service.analyze([['harga']], [0, 1], num_topics=2)

    assert result == {'error': 'Length mismatch'}


def test_dokumen_outlier_diabaikan(service):
    """Topik -1 (outlier HDBSCAN) tidak boleh ikut dihitung."""
    result = service.analyze(
        [['harga'], ['harga'], ['harga']],
        [-1, -1, -1],
        num_topics=2,
    )

    assert result == {}


def test_aspek_di_bawah_min_mentions_disaring(service):
    result = service.analyze(DOC_ASPECTS_STR, DOC_TOPICS, 2, min_mentions=99)

    assert result == {}


def test_input_kosong(service):
    assert service.analyze([], [], num_topics=0) == {}


def test_aspek_kosong_atau_none_dilewati(service):
    doc_aspects = [['harga', '', None], ['harga'], ['harga'], ['harga']]
    doc_topics = [0, 0, 1, 1]

    result = service.analyze(doc_aspects, doc_topics, 2, min_mentions=2)

    aspek = {r['aspect'] for r in result.get('pmi_top_associations', [])}
    assert '' not in aspek and None not in aspek
