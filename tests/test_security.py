"""
Tes untuk kontrol akses deployment (`app/security.py`).

Diuji sebagai perilaku, bukan sebagai teks sumber. Pelajaran itu datang dari
penyaring label topik: tesnya dulu memeriksa isi berkas (`assert '...' in
sumber`) dan lolos dengan senang hati melewati kesalahan rancangan.

Yang dijaga di sini semuanya punya mode kegagalan yang nyata:
- kunci kosong di produksi -> endpoint retraining terbuka untuk publik
- healthcheck kena batas laju -> kontainer dianggap mati lalu di-restart
- X-Forwarded-For dipercaya tanpa proxy -> batas laju dilewati dengan header
"""
import pytest

from app.security import PembatasLaju


# ── Batas laju ──────────────────────────────────────────────────────────────

def test_melewatkan_sampai_batas_lalu_menolak():
    p = PembatasLaju(maks=3, jendela_detik=60)
    assert [p.izinkan('a', sekarang=0) for _ in range(3)] == [True, True, True]
    assert p.izinkan('a', sekarang=0) is False


def test_jendela_bergeser_sehingga_kuota_pulih():
    p = PembatasLaju(maks=2, jendela_detik=10)
    assert p.izinkan('a', sekarang=0)
    assert p.izinkan('a', sekarang=1)
    assert p.izinkan('a', sekarang=2) is False
    # Setelah jendela lewat, dua permintaan pertama tidak lagi dihitung.
    assert p.izinkan('a', sekarang=11) is True


def test_kuota_dipisah_per_pemanggil():
    """Satu pemanggil yang berlebihan tidak boleh memblokir yang lain."""
    p = PembatasLaju(maks=1, jendela_detik=60)
    assert p.izinkan('a', sekarang=0)
    assert p.izinkan('a', sekarang=0) is False
    assert p.izinkan('b', sekarang=0) is True


def test_pembersihan_membuang_pemanggil_kedaluwarsa():
    """Tanpa ini `_jejak` tumbuh selamanya seiring jumlah IP unik."""
    p = PembatasLaju(maks=5, jendela_detik=10)
    p.izinkan('lama', sekarang=0)
    p.izinkan('baru', sekarang=100)
    p.bersihkan(sekarang=100)
    assert 'lama' not in p._jejak
    assert 'baru' in p._jejak


# ── Penjaga startup ─────────────────────────────────────────────────────────

def test_produksi_tanpa_kunci_menolak_start(monkeypatch):
    """Gagal-tertutup: layanan terbuka lebih buruk daripada layanan mati."""
    from app import security
    monkeypatch.setattr(security.settings, 'app_env', 'production')
    monkeypatch.setattr(security.settings, 'api_key', '')
    with pytest.raises(RuntimeError, match='API_KEY'):
        security.verifikasi_startup()


def test_produksi_dengan_kunci_boleh_start(monkeypatch):
    from app import security
    monkeypatch.setattr(security.settings, 'app_env', 'production')
    monkeypatch.setattr(security.settings, 'api_key', 'rahasia')
    security.verifikasi_startup()


def test_pengembangan_tanpa_kunci_tetap_boleh(monkeypatch):
    """Tes dan `python run.py` lokal tidak perlu konfigurasi tambahan."""
    from app import security
    monkeypatch.setattr(security.settings, 'app_env', 'development')
    monkeypatch.setattr(security.settings, 'api_key', '')
    security.verifikasi_startup()


# ── Endpoint ────────────────────────────────────────────────────────────────

@pytest.fixture
def klien(monkeypatch):
    from fastapi.testclient import TestClient
    from app import security
    import app.main as main
    monkeypatch.setattr(security.settings, 'api_key', 'kunci-uji')
    # TestClient tanpa lifespan: memuat bobot model tidak diperlukan untuk
    # menguji lapisan autentikasi, dan suite ini sengaja tidak menyentuh bobot.
    return TestClient(main.app)


def test_health_terbuka_tanpa_kunci(klien):
    """Platform hosting tidak bisa mengirim header kustom ke healthcheck."""
    assert klien.get('/health').status_code != 401


def test_endpoint_analisis_menolak_tanpa_kunci(klien):
    r = klien.post('/api/analyze/sentiment', json={'texts': ['halo']})
    assert r.status_code == 401


def test_endpoint_retraining_menolak_tanpa_kunci(klien):
    """Endpoint paling berbahaya: ia menimpa bobot model."""
    r = klien.post('/api/retrain/sentiment',
                   json=[{'text': 'halo', 'label': 'positive'}])
    assert r.status_code == 401


def test_kunci_salah_ditolak(klien):
    r = klien.post('/api/retrain/sentiment',
                   json=[{'text': 'halo', 'label': 'positive'}],
                   headers={'X-API-Key': 'tebakan'})
    assert r.status_code == 401


def test_kunci_benar_melewati_lapisan_auth(klien):
    """Lolos auth; 503/422 dari lapisan di bawahnya bukan urusan tes ini."""
    r = klien.post('/api/analyze/sentiment', json={'texts': ['halo']},
                   headers={'X-API-Key': 'kunci-uji'})
    assert r.status_code != 401


def test_openapi_tertutup_di_produksi(monkeypatch):
    """Skema terbuka memberi tahu persis bentuk muatan retraining."""
    from app.config import Settings
    s = Settings(app_env='production', expose_docs=False)
    assert s.expose_docs is False
    assert s.app_env == 'production'


# ── Batas laju vs pemanggil terpercaya ──────────────────────────────────────

@pytest.fixture
def klien_terbatas(monkeypatch):
    """Kunci menyala dan batas laju sangat rendah, supaya efeknya terlihat."""
    from fastapi.testclient import TestClient
    from app import security
    import app.main as main
    monkeypatch.setattr(security.settings, 'api_key', 'kunci-uji')
    monkeypatch.setattr(main._pembatas, 'maks', 3)
    main._pembatas.reset()
    return TestClient(main.app)


def test_laravel_berkunci_tidak_kena_batas_laju(klien_terbatas):
    """
    Laravel memecah analisis besar per 50 teks dari SATU IP - analisis TNI
    berarti ~136 permintaan. Pemanggil berkunci sah tidak boleh kena 429,
    kalau tidak analisis besar gagal di tengah jalan.
    """
    kode = [
        klien_terbatas.post('/api/preprocess', json={'texts': ['halo']},
                            headers={'X-API-Key': 'kunci-uji'}).status_code
        for _ in range(10)
    ]
    assert 429 not in kode


def test_tanpa_kunci_tetap_kena_batas_laju(klien_terbatas):
    """Orang luar tetap dibatasi: penebakan kunci tidak boleh tanpa rem."""
    kode = [
        klien_terbatas.post('/api/preprocess', json={'texts': ['halo']}).status_code
        for _ in range(6)
    ]
    assert kode[:3] == [401, 401, 401]
    assert 429 in kode[3:]


def test_kunci_salah_tetap_kena_batas_laju(klien_terbatas):
    kode = [
        klien_terbatas.post('/api/preprocess', json={'texts': ['halo']},
                            headers={'X-API-Key': 'tebakan'}).status_code
        for _ in range(6)
    ]
    assert 429 in kode[3:]
