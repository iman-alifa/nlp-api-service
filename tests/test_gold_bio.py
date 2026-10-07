"""
Jalur latih dari offset emas (data berlabel manusia seperti IndoNLU TermA).

Penurunan BIO lewat pencocokan string menandai ~5% kemunculan yang justru
sengaja dibiarkan 'O' oleh anotator; offset dipakai apa adanya untuk mencegah itu.
"""

import pytest

from app.services.aspect_service import AspectService


@pytest.fixture(scope='module')
def service(aspect_service_dimuat):
    """Alias ke fixture sesi di conftest.py - bobot dimuat sekali
    untuk seluruh suite, bukan ulang di tiap berkas."""
    return aspect_service_dimuat


def decode(svc, text, labels):
    """Ambil kembali potongan teks yang berlabel aspek dari label token."""
    enc = svc.tokenizer(text, return_offsets_mapping=True, truncation=True, max_length=128)
    out, cur = [], []
    for lab, (s, e) in zip(labels, enc['offset_mapping']):
        if s == 0 and e == 0:
            continue
        if lab == 1:
            if cur:
                out.append((cur[0], cur[-1]))
            cur = [s, e]
        elif lab == 2 and cur:
            cur[-1] = e
        else:
            if cur:
                out.append((cur[0], cur[-1]))
            cur = []
    if cur:
        out.append((cur[0], cur[-1]))
    return [text[a:b] for a, b in out]


def test_offset_emas_dipakai_apa_adanya(service):
    """Hanya kemunculan yang ditandai anotator yang berlabel aspek."""
    text = "kamar mandi kotor tapi kamar tidur bersih"
    # Anotator hanya menandai 'kamar mandi', bukan 'kamar' yang kedua
    spans = [(0, 11)]

    labels = service._generate_bio_tags(text, [], char_spans=spans)

    assert decode(service, text, labels) == ["kamar mandi"]


def test_tanpa_offset_kembali_ke_pencocokan_string(service):
    text = "ac rusak dan ac berisik"
    labels = service._generate_bio_tags(text, ["ac"])

    assert len(decode(service, text, labels)) == 2


def test_pencocokan_string_menandai_lebih_banyak_dari_emas(service):
    """
    Inilah kehilangan yang diukur laporan fidelitas: string cocok di dua tempat
    padahal anotator hanya menandai satu.
    """
    text = "wifi lemot padahal wifi baru dipasang"
    gold = service._generate_bio_tags(text, [], char_spans=[(0, 4)])
    derived = service._generate_bio_tags(text, ["wifi"])

    assert len(decode(service, text, gold)) == 1
    assert len(decode(service, text, derived)) == 2


def test_span_multitoken(service):
    text = "kolam renangnya bersih sekali"
    labels = service._generate_bio_tags(text, [], char_spans=[(0, 15)])

    assert decode(service, text, labels) == ["kolam renangnya"]


def test_dataset_emas_dilaporkan_di_bio_report(service):
    data = [
        {'text': 'ac rusak', 'aspects': ['ac'], 'spans': [{'start': 0, 'end': 2}]},
        {'text': 'wifi lemot', 'aspects': ['wifi'], 'spans': [{'start': 0, 'end': 4}]},
    ]
    _, _, _, report = service._prepare_bio_dataset(data)

    assert report['samples_with_gold_spans'] == 2
    assert report['aspects_unmatched'] == 0


def test_campuran_emas_dan_string(service):
    data = [
        {'text': 'ac rusak', 'spans': [{'start': 0, 'end': 2}]},
        {'text': 'wifi lemot', 'aspects': ['wifi']},
    ]
    _, _, _, report = service._prepare_bio_dataset(data)

    assert report['samples_with_gold_spans'] == 1
    assert report['samples_used'] == 2
