"""
Pelepasan memori setelah retraining (`lepas_memori`).

Tanpa ini proses layanan menetap sebesar puncak pelatihan: terukur di VPS
2,4 GB RAM + 4,1 GB swap setelah dua putaran, dan evaluasi hold-out
sesudahnya gagal karena setiap permintaan analisis aspek melewati 600 detik.
Tes di bawah menguji perilaku - malloc_trim benar-benar dipanggil di Linux,
tidak disentuh di luar Linux, dan pembungkus `retrain()` melepas memori
walaupun pelatihannya gagal.
"""
import asyncio
import sys
from unittest import mock

import pytest

from app.utils import training
from app.utils.training import lepas_memori


class _LibcPalsu:
    def __init__(self):
        self.panggilan = []

    def malloc_trim(self, pad):
        self.panggilan.append(pad)
        return 1


def test_linux_memanggil_malloc_trim_nol(monkeypatch):
    libc = _LibcPalsu()
    monkeypatch.setattr(sys, 'platform', 'linux')
    with mock.patch('ctypes.CDLL', return_value=libc) as cdll:
        hasil = lepas_memori()

    cdll.assert_called_once_with('libc.so.6')
    assert libc.panggilan == [0]
    assert hasil['malloc_trim'] is True


def test_di_luar_linux_libc_tidak_disentuh(monkeypatch):
    monkeypatch.setattr(sys, 'platform', 'win32')
    with mock.patch('ctypes.CDLL') as cdll:
        hasil = lepas_memori()

    cdll.assert_not_called()
    assert hasil['malloc_trim'] is False


def test_libc_tidak_tersedia_tidak_fatal(monkeypatch):
    """musl/Alpine tidak punya libc.so.6; pelatihan tidak boleh ikut gagal."""
    monkeypatch.setattr(sys, 'platform', 'linux')
    with mock.patch('ctypes.CDLL', side_effect=OSError('tidak ada')):
        hasil = lepas_memori()

    assert hasil['malloc_trim'] is False


def test_gc_dijalankan(monkeypatch):
    with mock.patch('gc.collect') as kumpul:
        lepas_memori()
    kumpul.assert_called()


def _jalankan_retrain_yang_gagal(svc, modul):
    dicatat = []
    with mock.patch.object(type(svc), '_retrain_sync', side_effect=RuntimeError('gagal di tengah')), \
            mock.patch.object(modul, 'lepas_memori', side_effect=lambda: dicatat.append(True)):
        with pytest.raises(RuntimeError):
            asyncio.run(svc.retrain([{'text': 'bagus', 'label': 'positive'}]))
    return dicatat


def test_retrain_sentimen_melepas_memori_walau_gagal():
    from app.services import sentiment_service as modul

    assert _jalankan_retrain_yang_gagal(modul.SentimentService(), modul) == [True]


def test_retrain_aspek_melepas_memori_walau_gagal(sentiment_service_bersama):
    from app.services import aspect_service as modul

    svc = modul.AspectService(sentiment_service=sentiment_service_bersama)
    assert _jalankan_retrain_yang_gagal(svc, modul) == [True]


def test_retrain_sukses_juga_melepas_memori():
    from app.services import sentiment_service as modul

    dicatat = []
    svc = modul.SentimentService()
    with mock.patch.object(modul.SentimentService, '_retrain_sync', return_value={'saved': True}), \
            mock.patch.object(modul, 'lepas_memori', side_effect=lambda: dicatat.append(True)):
        hasil = asyncio.run(svc.retrain([{'text': 'bagus', 'label': 'positive'}]))

    assert hasil == {'saved': True}
    assert dicatat == [True]


def test_rss_terbaca_atau_none():
    nilai = training._rss_mb()
    assert nilai is None or nilai > 0
