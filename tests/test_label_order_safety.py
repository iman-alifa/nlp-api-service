"""
Test dua kegagalan senyap seputar urutan label dan kalibrasi.

Keduanya sekelas: hasilnya tetap keluar, tidak ada galat, dan angkanya salah.
`_resolve_label_map` sudah ada justru karena urutan label berbeda-beda antar
checkpoint publik — tetapi separuh jalurnya dulu tidak ikut memakainya.

Tidak memuat bobot model: seluruh test memakai label_map buatan.
"""

import pytest
import torch

from app.services.sentiment_service import SentimentService


@pytest.fixture
def service():
    return SentimentService()


# ── scores harus mengikuti label_map, bukan indeks tetap ────────────────────

def test_scores_mengikuti_label_map_yang_diselesaikan(service):
    """`taufiqdp` memakai 0=negatif, kebalikan `crypter70`.

    Dulu `sentiment` memakai label_map sementara `scores` memakai indeks tetap
    0/1/2, sehingga untuk checkpoint semacam itu labelnya benar tetapi vektor
    probabilitasnya TERTUKAR - tanpa galat apa pun.
    """
    service.label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
    row = torch.tensor([0.70, 0.20, 0.10])

    skor = service._scores_dict(row)

    assert skor['negative'] == 0.7
    assert skor['neutral'] == 0.2
    assert skor['positive'] == 0.1


def test_scores_pada_urutan_baku_tetap_benar(service):
    """Urutan `crypter70`/`mdhugol` tidak boleh ikut berubah oleh perbaikan ini."""
    service.label_map = {0: 'positive', 1: 'neutral', 2: 'negative'}
    row = torch.tensor([0.70, 0.20, 0.10])

    assert service._scores_dict(row) == {
        'positive': 0.7, 'neutral': 0.2, 'negative': 0.1
    }


def test_scores_selalu_memuat_ketiga_kunci(service):
    """Laravel membaca ketiganya langsung; kunci yang hilang akan meledak di sana."""
    service.label_map = {0: 'positive', 1: 'negative'}
    skor = service._scores_dict(torch.tensor([0.6, 0.4]))

    assert set(skor) == {'positive', 'neutral', 'negative'}
    assert skor['neutral'] == 0.0


def test_scores_dan_sentiment_menunjuk_kelas_yang_sama(service):
    """Kelas dengan probabilitas tertinggi harus sama dengan label yang dipilih.

    Inilah invarian yang sesungguhnya dilanggar oleh bug lama: argmax dari
    `scores` menunjuk kelas yang berbeda dari `sentiment`.
    """
    service.label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
    row = torch.tensor([0.15, 0.10, 0.75])

    skor = service._scores_dict(row)
    tertinggi = max(skor, key=skor.get)
    terpilih = service.label_map[int(torch.argmax(row).item())]

    assert tertinggi == terpilih == 'positive'


def test_scores_tahan_terhadap_label_di_luar_ketiga_kelas(service):
    service.label_map = {0: 'positive', 1: 'neutral', 2: 'negative', 3: 'mixed'}
    skor = service._scores_dict(torch.tensor([0.4, 0.3, 0.2, 0.1]))

    assert set(skor) == {'positive', 'neutral', 'negative'}
    assert skor['positive'] == 0.4


# ── Suhu kalibrasi tidak boleh diwarisi dari model lain ─────────────────────

def test_kandidat_tanpa_suhu_terukur_gagal_keras():
    """`ayameRushia` punya temperature=None di SENTIMENT_MODELS.

    Pengecekan berbasis truthiness dulu MELEWATI penetapan suhu dan membiarkan
    nilai bawaan kelas bertahan - yaitu suhu milik model lain (mdhugol,
    T=1,3057). Model jadi salah kalibrasi tanpa galat, dan antrean tinjauan
    yang bergantung pada keyakinan ikut bergeser. Suhu itu milik bobot, bukan
    milik modul.
    """
    import importlib
    import os

    import app.config as modul_config

    lama = os.environ.get('SENTIMENT_BASE_MODEL')
    os.environ['SENTIMENT_BASE_MODEL'] = (
        'ayameRushia/bert-base-indonesian-1.5G-sentiment-analysis-smsa'
    )
    try:
        with pytest.raises(ValueError, match='temperature'):
            importlib.reload(modul_config)
    finally:
        if lama is None:
            os.environ.pop('SENTIMENT_BASE_MODEL', None)
        else:
            os.environ['SENTIMENT_BASE_MODEL'] = lama
        importlib.reload(modul_config)


def test_setiap_kandidat_dengan_suhu_juga_punya_ambang_tinjau():
    """Ambang tinjauan juga per bobot dan tidak boleh disalin antar model."""
    from app.config import SENTIMENT_MODELS

    for nama, profil in SENTIMENT_MODELS.items():
        if profil.get('temperature'):
            assert profil.get('review_threshold'), (
                f'{nama} punya suhu terukur tetapi tidak punya review_threshold'
            )


def test_suhu_efektif_berasal_dari_profil_model_aktif():
    from app.config import SENTIMENT_MODELS, settings

    profil = SENTIMENT_MODELS[settings.sentiment_base_model]
    assert settings.sentiment_temperature == profil['temperature']
    assert settings.sentiment_review_threshold == profil['review_threshold']
