import pytest
"""
Fixture bersama untuk seluruh suite.

Beberapa berkas tes membutuhkan `AspectService` yang bobotnya benar-benar
dimuat. Sebelumnya masing-masing punya fixture `scope='module'` sendiri,
sehingga IndoBERT (~500 MB) dimuat ULANG di tiap berkas dalam satu proses
Python yang sama. Akibatnya suite kadang mati dengan
`Windows fatal exception: access violation` di
`safetensors.torch.load_file` - segfault, bukan kegagalan assert, dan hanya
muncul saat seluruh suite dijalankan sekaligus.

Fixture aspek di sini `scope='module'`: bobotnya dilepas begitu satu berkas
selesai, sehingga model aspek dan model sentimen tidak menumpuk hidup
bersamaan sepanjang suite. `scope='session'` sempat dicoba dan justru
MEMPERBURUK - menahan model aspek hidup sepanjang suite membuat segfault
muncul 3 dari 4 kali.

`SentimentService` tetap `scope='session'` karena produksi juga membagikannya
(main.py dan AspectService memakai satu instance), jadi memuat salinan kedua
justru menyimpang dari perilaku sebenarnya.
"""

import pytest

# torch sebelum nltk - lihat catatan urutan impor di text_cleaner.py
import torch  # noqa: F401

from app.services.aspect_service import AspectService
from app.services.sentiment_service import SentimentService


@pytest.fixture(scope='session')
def sentiment_service_bersama():
    """
    Satu SentimentService untuk seluruh suite.

    Produksi juga membagikan satu instance (lihat lifespan di main.py): membuat
    salinan kedua berarti bobot ~500 MB dimuat dua kali dalam satu proses, yang
    di suite tes berujung segfault saat memuat safetensors.
    """
    return SentimentService()


@pytest.fixture(scope='module')
def aspect_service_dimuat(sentiment_service_bersama):
    """AspectService dengan bobot termuat, dipakai bersama seluruh suite."""
    svc = AspectService(sentiment_service=sentiment_service_bersama)
    svc._ensure_loaded()
    return svc


@pytest.fixture(scope='module')
def aspect_service_malas(sentiment_service_bersama):
    """
    AspectService tanpa memaksa muat bobot.

    Sebagian tes hanya memakai metode murni (penandaan BIO, pemecahan klausa)
    yang tidak menyentuh model sama sekali.
    """
    return AspectService(sentiment_service=sentiment_service_bersama)


@pytest.fixture(autouse=True)
def _reset_pembatas_laju():
    """
    Kosongkan pembatas laju sebelum setiap tes.

    Pembatas hidup di global tingkat modul (benar untuk layanan `--workers 1`,
    lihat `app/security.py`), sehingga tanpa fixture ini kuota bocor antar-tes:
    `test_retrain_concurrency` mengirim puluhan permintaan dan membuat tes
    berikutnya kena 429. Kegagalan seperti itu bergantung pada urutan
    eksekusi - kelas bug yang paling melelahkan untuk dilacak.
    """
    import app.main as main
    main._pembatas.reset()
    yield
    main._pembatas.reset()


@pytest.fixture(scope='session', autouse=True)
def _matikan_auth_bawaan():
    """
    Matikan autentikasi untuk seluruh sesi tes.

    Tanpa ini suite membaca `API_KEY` dari `.env` mesin pengembang: begitu kunci
    itu diisi untuk deployment, 38 tes yang mengirim permintaan tanpa header
    langsung gagal 401. Hasil tes tidak boleh bergantung pada isi berkas
    konfigurasi lokal - itu membuat "hijau di mesin saya" berhenti berarti
    apa-apa.

    **Lingkupnya sesi, bukan fungsi, dan itu bukan kerapian melainkan
    kebenaran.** Versi per-fungsi memulihkan kunci di teardown, dan
    `test_retrain_concurrency` mengirim permintaan dari thread latar: di bawah
    beban suite penuh, permintaan itu mendarat SESUDAH pemulihan dan dibalas
    401. Terukur - 6 kegagalan yang hanya muncul pada jalannya seluruh suite,
    tidak pernah saat berkasnya dijalankan sendiri. Memulihkan sekali di akhir
    sesi menghapus jendela balapan itu.

    Tes yang memang menguji autentikasi menyalakannya sendiri lewat
    `monkeypatch` (lihat `tests/test_security.py`), yang memulihkan ke '' -
    bukan ke nilai .env.
    """
    from app.config import settings

    asli = settings.api_key
    settings.api_key = ''
    yield
    settings.api_key = asli
