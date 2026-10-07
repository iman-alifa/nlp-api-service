"""
Mode rule-based (predefined_aspects).

Mode ini pernah rusak diam-diam: _extract_rule_based mengembalikan struktur
per-dokumen sementara seluruh hilir mengharapkan struktur per-aspek, sehingga
aspect_sentiments kosong, statistik nol, dan document_aspects kosong — tanpa
satu pun error.
"""

import pytest

from app.services.aspect_service import AspectService


@pytest.fixture(scope='module')
def service(aspect_service_dimuat):
    """Alias ke fixture sesi di conftest.py - bobot dimuat sekali
    untuk seluruh suite, bukan ulang di tiap berkas."""
    return aspect_service_dimuat


TEXTS = [
    "Harga mahal tapi pelayanan bagus",
    "Pelayanan lambat sekali",
    "layanannya ramah, harganya terjangkau",
]
ASPECTS = ["harga", "pelayanan"]


def _butuh_checkpoint(service):
    """Dua tes di bawah mengukur perilaku ekstraktor TERLATIH (tier 2 memakai
    checkpoint aspek). Tanpa `models/` - mis. klon baru atau sesi cloud - jalur
    cadangannya menandai kata tambahan (jumlah 4, bukan 3; varian 'ramah').
    Dilewati dengan alasan tertulis, bukan dilonggarkan."""
    if not getattr(service, '_model_path', None):
        pytest.skip('checkpoint aspek hasil pelatihan tidak ada')


def test_struktur_sama_dengan_mode_automatic(service):
    """Kunci yang sama harus ada, agar hilir tidak melewatinya diam-diam."""
    hasil = service._extract_rule_based(TEXTS, ASPECTS)

    assert hasil, 'tidak boleh kosong'
    for entry in hasil:
        assert {'aspect', 'count', 'occurrences', 'score', 'variants'} <= set(entry)


def test_semua_kemunculan_tercatat(service):
    _butuh_checkpoint(service)
    hasil = service._extract_rule_based(TEXTS, ASPECTS)
    per_aspek = {e['aspect']: e for e in hasil}

    # 'pelayanan' muncul di ketiga teks (satu lewat bentuk dasar 'layanannya')
    assert per_aspek['pelayanan']['count'] == 3
    assert per_aspek['harga']['count'] == 2


def test_occurrence_membawa_offset(service):
    """Offset diperlukan agar sentimen dihitung per klausa."""
    hasil = service._extract_rule_based(TEXTS, ASPECTS)

    for entry in hasil:
        for occ in entry['occurrences']:
            assert occ['start'] is not None and occ['end'] is not None
            potongan = occ['text'][occ['start']:occ['end']]
            assert potongan.strip(), potongan


def test_toleran_imbuhan(service):
    """'pelayanan' harus menemukan 'layanannya'."""
    _butuh_checkpoint(service)
    hasil = service._extract_rule_based(["layanannya sangat ramah"], ["pelayanan"])

    assert len(hasil) == 1
    assert hasil[0]['variants'] == ['layanannya']


def test_text_index_sejajar_dengan_input(service):
    """Kontrak keselarasan dengan AssociationService."""
    hasil = service._extract_rule_based(TEXTS, ASPECTS)

    for entry in hasil:
        for occ in entry['occurrences']:
            assert 0 <= occ['text_index'] < len(TEXTS)
            assert occ['text'] == TEXTS[occ['text_index']]


def test_aspek_tidak_ada_di_teks_dilewati(service):
    hasil = service._extract_rule_based(["harga mahal"], ["harga", "pengiriman"])

    assert [e['aspect'] for e in hasil] == ['harga']


def test_daftar_aspek_kosong(service):
    assert service._extract_rule_based(TEXTS, []) == []
    assert service._extract_rule_based(TEXTS, None) == []


@pytest.mark.asyncio
async def test_analyze_rule_based_mengisi_seluruh_keluaran(service):
    hasil = await service.analyze(TEXTS, mode='rule-based', predefined_aspects=ASPECTS)

    assert hasil['aspect_sentiments'], 'aspect_sentiments tidak boleh kosong'
    assert hasil['statistics']['total_aspects'] == 2
    assert len(hasil['document_aspects']) == len(TEXTS)
    assert any(hasil['document_aspects']), 'document_aspects tidak boleh kosong semua'


# ── Pemetaan aspek semantik (lapis 2) ───────────────────────────────────────
#
# Mode rule-based dulu hanya mencocokkan string, sehingga kalimat seperti
# "Antriannya lama" tidak pernah masuk ke kategori "pelayanan" yang diminta
# pengguna. Lapis kedua memetakan istilah temuan model ke kategori terdekat.
#
# Tes di bawah memakai stub, bukan bobot model: keputusan yang diuji adalah
# aturan penggabungannya (penyaringan tindihan, ambang, margin), bukan mutu
# embedding — itu diukur terpisah dan didokumentasikan di config.py.

import torch

from app.config import settings


class _StubAspectService(AspectService):
    """AspectService dengan model diganti stub yang bisa dikendalikan."""

    def __init__(self, extracted, vectors, protos):
        super().__init__()
        self._stub_extracted = extracted
        self._stub_vectors = vectors
        self._stub_protos = protos
        self._loaded = True

    def _extract_with_model(self, texts):
        return self._stub_extracted

    def _embed_spans(self, items):
        return self._stub_vectors[:len(items)]

    def _aspect_prototypes(self, aspects):
        return self._stub_protos, torch.zeros(1, self._stub_protos.shape[1])

    def _extract_context(self, aspect, text, window=3):
        return text


def _stub(extracted, vectors, protos):
    return _StubAspectService(extracted, vectors, protos)


# Dua kategori ortogonal, plus satu dimensi bebas sebagai arah "bukan kategori
# manapun". _semantic_matches menormalisasi vektornya, jadi yang menentukan skor
# adalah ARAH: memperkecil panjang vektor tidak menurunkan kemiripan kosinus.
PROTOS = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
KATEGORI = ['pelayanan', 'harga']


def test_istilah_dipetakan_ke_kategori_terdekat():
    """'antrian' tidak mengandung kata 'pelayanan' tapi harus masuk ke sana."""
    teks = ['Antriannya lama sekali']
    extracted = [{'occurrences': [{'text_index': 0, 'start': 0, 'end': 10}]}]
    svc = _stub(extracted, torch.tensor([[1.0, 0.0, 0.0]]), PROTOS)

    hasil = svc._semantic_matches(teks, KATEGORI, {})

    assert len(hasil) == 1
    assert hasil[0]['aspect'] == 'pelayanan'
    assert hasil[0]['surface'] == 'Antriannya'


def test_skor_di_bawah_ambang_ditolak():
    teks = ['Cuaca hari ini panas']
    extracted = [{'occurrences': [{'text_index': 0, 'start': 0, 'end': 5}]}]
    # Sebagian besar mengarah ke dimensi ketiga -> kemiripan ke kedua kategori
    # kecil, sementara margin di antara keduanya tetap sehat.
    lemah = settings.aspect_semantic_min_score / 2
    sisa = (1.0 - lemah ** 2) ** 0.5
    vektor = torch.tensor([[lemah, 0.0, sisa]])
    svc = _stub(extracted, vektor, PROTOS)

    assert svc._semantic_matches(teks, KATEGORI, {}) == []


def test_margin_tipis_ditolak():
    """Istilah yang mirip dengan SEMUA kategori tidak boleh dipaksa masuk."""
    teks = ['Sesuatu yang ambigu']
    extracted = [{'occurrences': [{'text_index': 0, 'start': 0, 'end': 7}]}]
    # Jarak sama ke kedua prototipe -> margin nol.
    vektor = torch.tensor([[0.7071, 0.7071, 0.0]])
    svc = _stub(extracted, vektor, PROTOS)

    assert svc._semantic_matches(teks, KATEGORI, {}) == []


def test_span_yang_sudah_cocok_harfiah_tidak_dihitung_dua_kali():
    teks = ['Pelayanannya bagus']
    extracted = [{'occurrences': [{'text_index': 0, 'start': 0, 'end': 12}]}]
    svc = _stub(extracted, torch.tensor([[1.0, 0.0, 0.0]]), PROTOS)

    tanpa_lapis1 = svc._semantic_matches(teks, KATEGORI, {})
    dengan_lapis1 = svc._semantic_matches(teks, KATEGORI, {0: [(0, 12)]})

    assert len(tanpa_lapis1) == 1
    assert dengan_lapis1 == [], 'span yang sudah harfiah tidak boleh diulang'


def test_aspek_lain_di_kalimat_yang_sudah_cocok_tetap_terbaca():
    """Penyaringan per-SPAN, bukan per-dokumen.

    Pada "Pelayanannya bagus tapi antriannya lama", 'pelayanannya' cocok
    harfiah. Kalau seluruh dokumen dilewati, 'antriannya' hilang dan aspek
    'pelayanan' tercatat positif saja — padahal keluhannya di klausa kedua.
    """
    teks = ['Pelayanannya bagus tapi antriannya lama']
    extracted = [{'occurrences': [
        {'text_index': 0, 'start': 0, 'end': 12},    # pelayanannya (harfiah)
        {'text_index': 0, 'start': 24, 'end': 34},   # antriannya   (semantik)
    ]}]
    svc = _stub(extracted, torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), PROTOS)

    hasil = svc._semantic_matches(teks, KATEGORI, {0: [(0, 12)]})

    assert [h['surface'] for h in hasil] == ['antriannya']


def test_pemetaan_semantik_bisa_dimatikan(monkeypatch):
    """Mematikannya harus mengembalikan perilaku harfiah-saja sepenuhnya."""
    teks = ['Antriannya lama sekali']
    extracted = [{'occurrences': [{'text_index': 0, 'start': 0, 'end': 10}]}]
    svc = _stub(extracted, torch.tensor([[1.0, 0.0, 0.0]]), PROTOS)

    monkeypatch.setattr(settings, 'aspect_semantic_mapping', False)
    assert svc._extract_rule_based(teks, KATEGORI) == []

    monkeypatch.setattr(settings, 'aspect_semantic_mapping', True)
    assert svc._extract_rule_based(teks, KATEGORI) != []


def test_kecocokan_dapat_diaudit():
    """match_type dan similarity harus ikut, karena lapis 2 probabilistik."""
    teks = ['Antriannya lama sekali']
    extracted = [{'occurrences': [{'text_index': 0, 'start': 0, 'end': 10}]}]
    svc = _stub(extracted, torch.tensor([[1.0, 0.0, 0.0]]), PROTOS)

    hasil = svc._extract_rule_based(teks, KATEGORI)

    entry = hasil[0]
    assert entry['semantic_matches'] == 1
    assert entry['lexical_matches'] == 0
    occ = entry['occurrences'][0]
    assert occ['match_type'] == 'semantic'
    assert 0.0 <= occ['similarity'] <= 1.0
