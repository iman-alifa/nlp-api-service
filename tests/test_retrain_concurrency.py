"""
Test ketahanan endpoint retraining.

Sejak `text-analysis-web` memicu retraining OTOMATIS dari koreksi pengguna
(bukan lagi dari tombol admin), waktunya tidak lagi dipilih manusia dan bisa
jatuh tepat ketika pengguna lain sedang menganalisis. Dua sifat karena itu jadi
wajib, dan keduanya diuji di sini tanpa memuat bobot model sungguhan:

1. Pelatihan tidak boleh memblokir event loop - permintaan lain harus tetap
   dilayani selagi pelatihan berjalan.
2. Dua pelatihan model yang sama tidak boleh berjalan bersamaan; yang kedua
   dibalas 409, bukan diantrekan atau dijalankan paralel.
"""

import asyncio
import inspect
import threading
import time

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.main import app
from app.services.aspect_service import AspectService
from app.services.sentiment_service import SentimentService


@pytest.fixture(scope='module')
def client():
    with TestClient(app) as c:
        yield c


SAMPEL_SENTIMEN = [{'text': f'contoh nomor {i}', 'label': 'positive'} for i in range(10)]
SAMPEL_ASPEK = [{'text': f'pelayanan nomor {i}', 'aspects': ['pelayanan']} for i in range(10)]


# ── 1. Pelatihan berjalan di thread, bukan di event loop ────────────────────

def test_retrain_sentimen_adalah_pembungkus_tipis_di_atas_metode_sinkron():
    """`retrain` harus async, `_retrain_sync` harus TIDAK async.

    Kalau seseorang kelak menggabungkannya kembali jadi satu `async def` berisi
    loop PyTorch, sifat non-blocking hilang tanpa satu pun test lain gagal.
    """
    assert inspect.iscoroutinefunction(SentimentService.retrain)
    assert not inspect.iscoroutinefunction(SentimentService._retrain_sync)
    assert 'to_thread' in inspect.getsource(SentimentService.retrain)


def test_retrain_aspek_adalah_pembungkus_tipis_di_atas_metode_sinkron():
    assert inspect.iscoroutinefunction(AspectService.retrain)
    assert not inspect.iscoroutinefunction(AspectService._retrain_sync)
    assert 'to_thread' in inspect.getsource(AspectService.retrain)


def test_pembungkus_meneruskan_seluruh_argumen_apa_adanya():
    """Pembungkus tidak boleh diam-diam menjatuhkan argumen.

    Persis kelas bug yang pernah membuat /api/analyze/combined membuang
    `predefined_aspects` dan `mode`: tanpa galat, tanpa terlihat di UI.
    """
    diterima = {}

    class Palsu(SentimentService):
        def __init__(self):
            pass

        def _retrain_sync(self, *args, **kwargs):
            diterima['args'] = args
            diterima['kwargs'] = kwargs
            return {'ok': True}

    hasil = asyncio.run(
        Palsu().retrain(SAMPEL_SENTIMEN, epochs=7, learning_rate=1e-4,
                        save_path='/x/y', force=True)
    )

    assert hasil == {'ok': True}
    semua = list(diterima['args']) + list(diterima['kwargs'].values())
    assert 7 in semua
    assert 1e-4 in semua
    assert '/x/y' in semua
    assert True in semua


def test_pembungkus_aspek_meneruskan_seluruh_argumen_apa_adanya():
    diterima = {}

    class Palsu(AspectService):
        def __init__(self):
            pass

        def _retrain_sync(self, **kwargs):
            diterima.update(kwargs)
            return {'ok': True}

    asyncio.run(
        Palsu().retrain(SAMPEL_ASPEK, epochs=5, learning_rate=3e-5,
                        save_path='/a/b', force=True, batch_size=8,
                        warmup_ratio=0.2, use_class_weights=True)
    )

    assert diterima['epochs'] == 5
    assert diterima['learning_rate'] == 3e-5
    assert diterima['save_path'] == '/a/b'
    assert diterima['force'] is True
    assert diterima['batch_size'] == 8
    assert diterima['warmup_ratio'] == 0.2
    assert diterima['use_class_weights'] is True


def test_health_tetap_menjawab_selagi_retraining_berjalan(client, monkeypatch):
    """Ini pengujian sesungguhnya dari nomor 1.

    Sebelum perbaikan, `/health` menggantung sampai pelatihan selesai karena
    loop PyTorch berjalan di dalam event loop. Di sini pelatihan ditiru dengan
    `Event.wait` - blocking sinkron yang sama persis sifatnya - dan `/health`
    harus tetap membalas selagi pelatihan itu berlangsung.
    """
    mulai = threading.Event()
    boleh_selesai = threading.Event()

    def latihan_lambat(*args, **kwargs):
        mulai.set()
        boleh_selesai.wait(timeout=10)
        return {'status': 'success'}

    monkeypatch.setattr(main_module.sentiment_service, '_retrain_sync', latihan_lambat)

    balasan = {}

    def kirim_retrain():
        balasan['retrain'] = client.post(
            '/api/retrain/sentiment', json={'training_data': SAMPEL_SENTIMEN}
        ).status_code

    t = threading.Thread(target=kirim_retrain)
    t.start()
    assert mulai.wait(timeout=10), 'retraining tidak pernah mulai'

    # Selagi "pelatihan" masih menggantung di thread pekerja:
    t0 = time.monotonic()
    health = client.get('/health')
    durasi = time.monotonic() - t0

    boleh_selesai.set()
    t.join(timeout=15)

    assert health.status_code == 200
    assert durasi < 5.0, f'/health menggantung {durasi:.1f}s - event loop terblokir'
    assert balasan['retrain'] == 200


# ── 2. Penjaga pelatihan serentak ───────────────────────────────────────────

def test_retrain_kedua_dibalas_409_bukan_dijalankan_paralel(client, monkeypatch):
    """Dua pelatihan bersamaan akan menulis checkpoint ke lokasi yang sama."""
    mulai = threading.Event()
    boleh_selesai = threading.Event()
    jumlah_jalan = []

    def latihan_lambat(*args, **kwargs):
        jumlah_jalan.append(1)
        mulai.set()
        boleh_selesai.wait(timeout=10)
        return {'status': 'success'}

    monkeypatch.setattr(main_module.sentiment_service, '_retrain_sync', latihan_lambat)

    hasil = {}

    def kirim_pertama():
        hasil['pertama'] = client.post(
            '/api/retrain/sentiment', json={'training_data': SAMPEL_SENTIMEN}
        ).status_code

    t = threading.Thread(target=kirim_pertama)
    t.start()
    assert mulai.wait(timeout=10)

    kedua = client.post('/api/retrain/sentiment', json={'training_data': SAMPEL_SENTIMEN})

    boleh_selesai.set()
    t.join(timeout=15)

    assert kedua.status_code == 409
    assert 'sedang berjalan' in str(kedua.json())
    assert hasil['pertama'] == 200
    assert len(jumlah_jalan) == 1, 'pelatihan kedua tetap dijalankan'


def test_lock_dilepas_setelah_selesai_sehingga_retrain_berikutnya_diterima(client, monkeypatch):
    """Lock yang bocor akan mengunci retraining selamanya sampai restart."""
    monkeypatch.setattr(
        main_module.sentiment_service, '_retrain_sync',
        lambda *a, **k: {'status': 'success'},
    )

    for _ in range(3):
        r = client.post('/api/retrain/sentiment', json={'training_data': SAMPEL_SENTIMEN})
        assert r.status_code == 200

    assert not main_module._model_locks['sentiment'].locked()


def test_lock_dilepas_walaupun_pelatihan_melempar_galat(client, monkeypatch):
    """Pelatihan yang gagal tidak boleh membuat endpoint terkunci permanen."""
    def meledak(*args, **kwargs):
        raise RuntimeError('CUDA out of memory')

    monkeypatch.setattr(main_module.sentiment_service, '_retrain_sync', meledak)

    r = client.post('/api/retrain/sentiment', json={'training_data': SAMPEL_SENTIMEN})

    assert r.status_code == 500
    assert not main_module._model_locks['sentiment'].locked()

    monkeypatch.setattr(
        main_module.sentiment_service, '_retrain_sync',
        lambda *a, **k: {'status': 'success'},
    )
    lanjutan = client.post(
        '/api/retrain/sentiment', json={'training_data': SAMPEL_SENTIMEN}
    )
    assert lanjutan.status_code == 200


def test_lock_sentimen_dan_aspek_terpisah(client, monkeypatch):
    """Melatih sentimen tidak boleh memblokir pelatihan aspek."""
    mulai = threading.Event()
    boleh_selesai = threading.Event()

    def latihan_lambat(*args, **kwargs):
        mulai.set()
        boleh_selesai.wait(timeout=10)
        return {'status': 'success'}

    monkeypatch.setattr(main_module.sentiment_service, '_retrain_sync', latihan_lambat)
    monkeypatch.setattr(
        main_module.aspect_service, '_retrain_sync',
        lambda *a, **k: {'status': 'success'},
    )

    def kirim_sentimen():
        client.post('/api/retrain/sentiment', json={'training_data': SAMPEL_SENTIMEN})

    t = threading.Thread(target=kirim_sentimen)
    t.start()
    assert mulai.wait(timeout=10)

    aspek = client.post('/api/retrain/aspect', json={'training_data': SAMPEL_ASPEK})

    boleh_selesai.set()
    t.join(timeout=15)

    assert aspek.status_code == 200


# ── 3. Status HTTP sampai apa adanya ke Laravel ─────────────────────────────

def test_service_belum_siap_membalas_503_bukan_500(client, monkeypatch):
    """503 dulu tertelan `except Exception` dan berubah jadi 500.

    Laravel memakai kode status untuk membedakan "layanan belum siap" (wajar
    saat cold start, layak dicoba lagi) dari kegagalan sungguhan.
    """
    monkeypatch.setattr(main_module, 'sentiment_service', None)

    r = client.post('/api/retrain/sentiment', json={'training_data': SAMPEL_SENTIMEN})
    assert r.status_code == 503

    r = client.post('/api/analyze/sentiment', json={'texts': ['halo']})
    assert r.status_code == 503


def test_aspek_belum_siap_membalas_503_bukan_500(client, monkeypatch):
    monkeypatch.setattr(main_module, 'aspect_service', None)

    r = client.post('/api/retrain/aspect', json={'training_data': SAMPEL_ASPEK})
    assert r.status_code == 503

    r = client.post('/api/analyze/aspect', json={'texts': ['halo']})
    assert r.status_code == 503


def test_topic_belum_siap_membalas_503_bukan_500(client, monkeypatch):
    monkeypatch.setattr(main_module, 'topic_service', None)

    r = client.post('/api/analyze/topic', json={'texts': ['halo']})
    assert r.status_code == 503


# ── 4. Inference tidak boleh menyelip ke tengah pelatihan ───────────────────
#
# Ini konsekuensi dari perbaikan nomor 1, bukan masalah lama: begitu pelatihan
# pindah ke thread, pelatihan dan inference berjalan di atas objek model yang
# SAMA. `_retrain_sync` memanggil `self.model.train()` (dropout menyala) lalu
# memperbarui bobot batch demi batch, sehingga analisis yang jatuh di tengahnya
# membaca bobot setengah terlatih - dan mengembalikan angka salah tanpa galat.

def test_analisis_menunggu_pelatihan_selesai(client, monkeypatch):
    """Analisis tidak boleh berjalan bersamaan dengan pelatihan model yang sama."""
    urutan = []
    mulai = threading.Event()
    boleh_selesai = threading.Event()

    def latihan_lambat(*args, **kwargs):
        urutan.append('latih-mulai')
        mulai.set()
        boleh_selesai.wait(timeout=10)
        urutan.append('latih-selesai')
        return {'status': 'success'}

    async def analisis_palsu(*args, **kwargs):
        urutan.append('analisis')
        return {'predictions': [], 'summary': ''}

    monkeypatch.setattr(main_module.sentiment_service, '_retrain_sync', latihan_lambat)
    monkeypatch.setattr(main_module.sentiment_service, 'analyze', analisis_palsu)

    hasil = {}

    def kirim_analisis():
        hasil['kode'] = client.post(
            '/api/analyze/sentiment', json={'texts': ['halo dunia']}
        ).status_code

    t_latih = threading.Thread(
        target=lambda: client.post(
            '/api/retrain/sentiment', json={'training_data': SAMPEL_SENTIMEN}
        )
    )
    t_latih.start()
    assert mulai.wait(timeout=10)

    t_analisis = threading.Thread(target=kirim_analisis)
    t_analisis.start()
    time.sleep(1.0)   # beri kesempatan analisis menyelip kalau memang bisa

    assert 'analisis' not in urutan, f'analisis menyelip saat melatih: {urutan}'

    boleh_selesai.set()
    t_latih.join(timeout=15)
    t_analisis.join(timeout=15)

    assert hasil['kode'] == 200
    assert urutan == ['latih-mulai', 'latih-selesai', 'analisis'], urutan


def test_health_tetap_cepat_walau_analisis_menunggu_pelatihan(client, monkeypatch):
    """Lock-nya asyncio, bukan threading.

    Kalau dipakai `threading.Lock`, coroutine analisis akan MEMBLOKIR event loop
    selagi menunggu pelatihan, dan /health menggantung lagi - persis masalah yang
    baru saja diperbaiki.
    """
    mulai = threading.Event()
    boleh_selesai = threading.Event()

    def latihan_lambat(*args, **kwargs):
        mulai.set()
        boleh_selesai.wait(timeout=10)
        return {'status': 'success'}

    async def analisis_palsu(*args, **kwargs):
        return {'predictions': [], 'summary': ''}

    monkeypatch.setattr(main_module.sentiment_service, '_retrain_sync', latihan_lambat)
    monkeypatch.setattr(main_module.sentiment_service, 'analyze', analisis_palsu)

    t_latih = threading.Thread(
        target=lambda: client.post(
            '/api/retrain/sentiment', json={'training_data': SAMPEL_SENTIMEN}
        )
    )
    t_latih.start()
    assert mulai.wait(timeout=10)

    t_analisis = threading.Thread(
        target=lambda: client.post('/api/analyze/sentiment', json={'texts': ['halo']})
    )
    t_analisis.start()
    time.sleep(0.5)

    t0 = time.monotonic()
    health = client.get('/health')
    durasi = time.monotonic() - t0

    boleh_selesai.set()
    t_latih.join(timeout=15)
    t_analisis.join(timeout=15)

    assert health.status_code == 200
    assert durasi < 5.0, f'/health menggantung {durasi:.1f}s - event loop terblokir'


def test_analisis_topik_tidak_ikut_terkunci(client, monkeypatch):
    """Model topik tidak pernah dilatih ulang lewat API, jadi tidak perlu antre."""
    mulai = threading.Event()
    boleh_selesai = threading.Event()

    def latihan_lambat(*args, **kwargs):
        mulai.set()
        boleh_selesai.wait(timeout=10)
        return {'status': 'success'}

    async def topik_palsu(*args, **kwargs):
        return {'topics': [], 'document_topics': []}

    monkeypatch.setattr(main_module.sentiment_service, '_retrain_sync', latihan_lambat)
    monkeypatch.setattr(main_module.topic_service, 'analyze', topik_palsu)

    t = threading.Thread(
        target=lambda: client.post(
            '/api/retrain/sentiment', json={'training_data': SAMPEL_SENTIMEN}
        )
    )
    t.start()
    assert mulai.wait(timeout=10)

    t0 = time.monotonic()
    topik = client.post('/api/analyze/topic', json={'texts': ['halo', 'dunia']})
    durasi = time.monotonic() - t0

    boleh_selesai.set()
    t.join(timeout=15)

    assert topik.status_code == 200
    assert durasi < 5.0, f'analisis topik ikut mengantre ({durasi:.1f}s)'


def test_lock_analisis_dilepas_walaupun_analisis_gagal(client, monkeypatch):
    """Analisis yang melempar galat tidak boleh menahan lock selamanya."""
    async def meledak(*args, **kwargs):
        raise RuntimeError('model rusak')

    monkeypatch.setattr(main_module.sentiment_service, 'analyze', meledak)

    r = client.post('/api/analyze/sentiment', json={'texts': ['halo']})
    assert r.status_code == 500

    assert not main_module._model_locks['sentiment'].locked()
    assert not main_module._model_locks['aspect'].locked()


def test_urutan_pengambilan_lock_tetap_sehingga_tidak_deadlock():
    """Analisis aspek butuh dua lock; retraining butuh satu.

    Selama urutannya selalu abjad, tidak ada siklus tunggu-menunggu. Test ini
    mengunci sifat itu supaya penambahan lock baru tidak diam-diam merusaknya.
    """
    import inspect
    sumber = inspect.getsource(main_module._use_models)
    assert 'sorted(' in sumber, 'urutan pengambilan lock tidak lagi dijamin tetap'


# ── Protokol kolam: tiap putaran dari bobot dasar ───────────────────────────

def test_sentimen_dilatih_dari_bobot_dasar_bukan_checkpoint_terakhir():
    """Protokol active learning baku (Settles 2009): kolam berlabel bertambah
    tiap putaran, dan model dilatih ulang pada SELURUH kolam dari titik awal
    yang sama.

    Melanjutkan checkpoint sebelumnya SAMBIL melatih pada kolam kumulatif
    membuat koreksi lama terlatih berkali-kali - koreksi putaran 1 dilihat dua
    kali pada putaran 2, tiga kali pada putaran 3. Selain itu hasil putaran N
    jadi bergantung pada urutan pelatihan, sehingga satu titik pada kurva
    iterasi tidak bisa dihitung ulang tanpa mengulang seluruh rangkaian.
    """
    import inspect

    from app.config import settings
    from app.services.sentiment_service import SentimentService

    sumber = inspect.getsource(SentimentService._retrain_sync)

    assert 'dari_dasar' in sumber
    assert 'settings.retrain_from_baseline' in sumber
    assert 'self.base_model_name' in sumber
    # Peta label WAJIB dibaca ulang: checkpoint hasil retraining dan model
    # dasar tidak dijamin memakai urutan label yang sama.
    assert '_resolve_label_map()' in sumber
    assert settings.retrain_from_baseline is True


def test_aspek_memakai_checkpoint_dasar_bukan_indobert_mentah():
    """Dasar modul aspek adalah checkpoint SEBELUM active learning (K6).

    IndoBERT mentah hanya mencapai F1 0,17-0,35 pada label manusia versus 0,92
    setelah fine-tuning, jadi memakainya sebagai titik awal membuat "sebelum
    active learning" berarti sistem yang tidak pernah dipakai siapa pun.
    """
    import inspect

    from app.services.aspect_service import AspectService

    sumber = inspect.getsource(AspectService._retrain_sync)

    assert 'aspect_baseline_checkpoint' in sumber
    # Tidak boleh diam-diam jatuh ke perilaku lama.
    assert 'logger.warning' in sumber


def test_pelatihan_dari_dasar_dilaporkan_ke_pemanggil():
    """Tanpa penanda ini, riwayat pelatihan tidak bisa membedakan putaran yang
    mengikuti protokol dari yang melanjutkan checkpoint."""
    import inspect

    from app.services.sentiment_service import SentimentService

    assert "'trained_from_baseline'" in inspect.getsource(SentimentService._retrain_sync)


# ── 503 + Retry-After saat pelatihan (retrain_reject_inference) ─────────────
#
# Menunggu lock hanya benar bila tunggunya singkat. Pelatihan memakan
# belasan-puluhan menit sementara Laravel menyerah setelah 600 detik, dan
# menyerahnya klien tidak membatalkan penantian di server: analisisnya tetap
# dijalankan begitu lock lepas tanpa ada yang menunggu. Tes di bawah menjaga
# perilaku penggantinya.

def _latih_di_latar(client, monkeypatch, jalur, modul, sampel):
    mulai = threading.Event()
    boleh_selesai = threading.Event()

    def latihan_lambat(*args, **kwargs):
        mulai.set()
        boleh_selesai.wait(timeout=10)
        return {'status': 'success', 'saved': False}

    svc = getattr(main_module, f'{modul}_service')
    monkeypatch.setattr(svc, '_retrain_sync', latihan_lambat)
    t = threading.Thread(
        target=lambda: client.post(jalur, json={'training_data': sampel})
    )
    t.start()
    assert mulai.wait(timeout=10), 'pelatihan tidak pernah mulai'
    return t, boleh_selesai


def test_analisis_ditolak_503_dengan_retry_after_saat_pelatihan(client, monkeypatch):
    monkeypatch.setattr(main_module.settings, 'retrain_reject_inference', True)
    t, selesai = _latih_di_latar(client, monkeypatch, '/api/retrain/sentiment',
                                 'sentiment', SAMPEL_SENTIMEN)
    try:
        r = client.post('/api/analyze/sentiment', json={'texts': ['halo dunia']})
        assert r.status_code == 503
        tunggu = int(r.headers['Retry-After'])
        assert 30 <= tunggu <= 600
    finally:
        selesai.set()
        t.join(timeout=15)


def test_analisis_aspek_ikut_ditolak_saat_sentimen_dilatih(client, monkeypatch):
    """AspectService memakai SentimentService untuk polaritas per aspek."""
    monkeypatch.setattr(main_module.settings, 'retrain_reject_inference', True)
    t, selesai = _latih_di_latar(client, monkeypatch, '/api/retrain/sentiment',
                                 'sentiment', SAMPEL_SENTIMEN)
    try:
        r = client.post('/api/analyze/aspect', json={'texts': ['pelayanan bagus']})
        assert r.status_code == 503
        assert 'Retry-After' in r.headers
    finally:
        selesai.set()
        t.join(timeout=15)


def test_analisis_topik_tetap_jalan_saat_pelatihan(client, monkeypatch):
    """Model topik tidak pernah dilatih lewat API, jadi tidak boleh ditolak."""
    monkeypatch.setattr(main_module.settings, 'retrain_reject_inference', True)

    async def topik_palsu(*args, **kwargs):
        return {'topics': [], 'document_topics': []}

    monkeypatch.setattr(main_module.topic_service, 'analyze', topik_palsu)
    t, selesai = _latih_di_latar(client, monkeypatch, '/api/retrain/sentiment',
                                 'sentiment', SAMPEL_SENTIMEN)
    try:
        r = client.post('/api/analyze/topic', json={'texts': ['halo dunia']})
        assert r.status_code != 503
    finally:
        selesai.set()
        t.join(timeout=15)


def test_analisis_kembali_normal_setelah_pelatihan_selesai(client, monkeypatch):
    monkeypatch.setattr(main_module.settings, 'retrain_reject_inference', True)

    async def analisis_palsu(*args, **kwargs):
        return {'predictions': [], 'summary': ''}

    monkeypatch.setattr(main_module.sentiment_service, 'analyze', analisis_palsu)
    t, selesai = _latih_di_latar(client, monkeypatch, '/api/retrain/sentiment',
                                 'sentiment', SAMPEL_SENTIMEN)
    selesai.set()
    t.join(timeout=15)
    assert 'sentiment' not in main_module._pelatihan, 'status pelatihan tidak dibersihkan'
    r = client.post('/api/analyze/sentiment', json={'texts': ['halo dunia']})
    assert r.status_code == 200


def test_status_pelatihan_dibersihkan_walau_pelatihan_gagal(client, monkeypatch):
    """Kalau tertinggal, setiap analisis sesudahnya ditolak selamanya."""
    monkeypatch.setattr(main_module.settings, 'retrain_reject_inference', True)

    def latihan_gagal(*args, **kwargs):
        raise RuntimeError('pelatihan gagal')

    monkeypatch.setattr(main_module.sentiment_service, '_retrain_sync', latihan_gagal)
    client.post('/api/retrain/sentiment', json={'training_data': SAMPEL_SENTIMEN})
    assert 'sentiment' not in main_module._pelatihan


def test_bawaan_mati_analisis_tetap_menunggu(client, monkeypatch):
    """Tanpa saklar, perilaku lama dipertahankan: menunggu, bukan ditolak."""
    monkeypatch.setattr(main_module.settings, 'retrain_reject_inference', False)

    async def analisis_palsu(*args, **kwargs):
        return {'predictions': [], 'summary': ''}

    monkeypatch.setattr(main_module.sentiment_service, 'analyze', analisis_palsu)
    t, selesai = _latih_di_latar(client, monkeypatch, '/api/retrain/sentiment',
                                 'sentiment', SAMPEL_SENTIMEN)
    hasil = {}
    t_a = threading.Thread(target=lambda: hasil.setdefault(
        'kode', client.post('/api/analyze/sentiment', json={'texts': ['x']}).status_code))
    t_a.start()
    time.sleep(0.5)
    selesai.set()
    t.join(timeout=15)
    t_a.join(timeout=15)
    assert hasil['kode'] == 200
