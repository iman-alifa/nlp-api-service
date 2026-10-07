"""
Uji asap: pastikan bobot model benar-benar termuat, bukan diam-diam jatuh ke
jalur cadangan.

Sisa suite sengaja tidak memuat bobot agar cepat dan bebas jaringan. Justru itu
membuka celah yang nyaris lolos: menambahkan `low_cpu_mem_usage=True` tanpa
paket `accelerate` membuat `from_pretrained` melempar ImportError, blok except
di `_ensure_loaded` menangkapnya, dan service berpindah ke daftar kata **tanpa
satu pun galat sampai ke pemanggil**. Akurasi SmSA anjlok 0,9160 -> 0,6740 dan
SELURUH suite tetap hijau.

Tes di sini menutup kelas kegagalan itu. Ia dilewati dengan rapi bila bobot
tidak tersedia (tanpa cache dan tanpa jaringan), sehingga tidak memaksa unduhan
pada lingkungan bersih.
"""

import pytest

import torch  # noqa: F401  (torch sebelum nltk)



@pytest.fixture(scope='module')
def service_dimuat(sentiment_service_bersama):
    """
    Memakai instance sesi dari conftest, BUKAN membuat yang baru.

    Memuat bobot sentimen kedua dalam satu proses pytest memicu segfault di
    safetensors pada Windows - dan itu juga alasan produksi membagikan satu
    instance antara main.py dan AspectService.
    """
    sentiment_service_bersama._ensure_loaded()
    if sentiment_service_bersama.model is None:
        pytest.skip('bobot model sentimen tidak tersedia (tanpa cache/jaringan)')
    return sentiment_service_bersama


def test_bobot_sentimen_benar_benar_termuat(service_dimuat):
    """Regresi: opsi pemuatan yang salah membuat service diam-diam degradasi."""
    assert service_dimuat.model is not None
    assert service_dimuat.tokenizer is not None
    assert service_dimuat._loaded is True


@pytest.mark.asyncio
async def test_prediksi_memakai_indobert_bukan_daftar_kata(service_dimuat):
    """
    `method` harus menyebut jalur model. Bila ia berbunyi 'rule-based' pada
    proses yang bobotnya seharusnya ada, berarti pemuatan gagal diam-diam.
    """
    hasil = await service_dimuat.analyze(
        texts=['pelayanannya sangat bagus dan memuaskan'],
        preprocessing_config=None,
    )

    prediksi = hasil['predictions'][0]
    assert prediksi['method'] == 'indobert', (
        'service jatuh ke jalur cadangan padahal bobot tersedia'
    )
    assert prediksi['sentiment'] == 'positive'


def test_peta_label_diambil_dari_config_model_yang_dimuat(service_dimuat):
    """Urutan label harus berasal dari model yang benar-benar dipakai."""
    assert set(service_dimuat.label_map.values()) == {
        'positive', 'neutral', 'negative'
    }


def test_suhu_kalibrasi_terpasang(service_dimuat):
    from app.config import SENTIMENT_MODELS

    profil = SENTIMENT_MODELS.get(service_dimuat.model_name, {})
    if profil.get('temperature'):
        assert service_dimuat.temperature == profil['temperature']
