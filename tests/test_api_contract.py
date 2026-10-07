"""
Test kontrak API dan konfigurasi.

Hanya menyentuh jalur yang tidak memuat bobot model: /health, validasi schema,
dan resolusi path checkpoint. Endpoint analisis sengaja tidak dipanggil karena
akan mengunduh model IndoBERT.
"""

import os

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import app
from app.schemas.request_schemas import (
    AspectRetrainRequest,
    SentimentRetrainRequest,
    TextAnalysisRequest,
)


@pytest.fixture(scope='module')
def client():
    with TestClient(app) as c:
        yield c


# ── Health ──────────────────────────────────────────────────────────────────

def test_health_melaporkan_semua_service(client):
    body = client.get('/health').json()

    assert body['status'] == 'healthy'
    assert body['services'] == {
        'sentiment': True,
        'aspect': True,
        'topic': True,
        'association': True,
    }


def test_health_melaporkan_asal_bobot_model(client):
    """
    Laravel perlu tahu apakah service memakai checkpoint hasil active learning
    atau diam-diam kembali ke model dasar setelah restart.
    """
    models = client.get('/health').json()['models']

    assert 'sentiment_source' in models
    assert isinstance(models['sentiment_retrained'], bool)
    assert isinstance(models['aspect_retrained'], bool)


def test_root_endpoint(client):
    assert client.get('/').json()['status'] == 'online'


# ── Endpoint yang harus ada (dipakai Laravel) ───────────────────────────────

def test_seluruh_endpoint_terdaftar(client):
    paths = {r.path for r in app.routes if hasattr(r, 'path')}

    for expected in [
        '/health',
        '/api/preprocess',
        '/api/analyze/sentiment',
        '/api/analyze/aspect',
        '/api/analyze/topic',
        '/api/analyze/combined',
        '/api/analyze/association',
        '/api/retrain/sentiment',
        '/api/retrain/aspect',
    ]:
        assert expected in paths, f'{expected} hilang — NLPApiService memanggilnya'


# ── Validasi schema (Pydantic v2) ───────────────────────────────────────────

def test_texts_kosong_ditolak():
    with pytest.raises(Exception):
        TextAnalysisRequest(texts=[])


def test_texts_hanya_spasi_ditolak():
    with pytest.raises(Exception):
        TextAnalysisRequest(texts=['   ', ''])


def test_mode_tidak_dikenal_ditolak():
    with pytest.raises(Exception):
        TextAnalysisRequest(texts=['halo'], mode='ngawur')


def test_mode_valid_diterima():
    for mode in ('automatic', 'rule-based'):
        assert TextAnalysisRequest(texts=['halo'], mode=mode).mode == mode


def test_retrain_menolak_kurang_dari_sepuluh_sampel():
    """Batas ini dicerminkan NLPApiService::MIN_RETRAIN_SAMPLES di sisi Laravel."""
    with pytest.raises(Exception):
        SentimentRetrainRequest(
            training_data=[{'text': 'halo', 'label': 'positive'}] * 9
        )


def test_retrain_menerima_sepuluh_sampel():
    req = SentimentRetrainRequest(
        training_data=[{'text': 'halo', 'label': 'positive'}] * 10
    )

    assert len(req.training_data) == 10
    assert req.epochs == 3


def test_force_default_false():
    """Tanpa force, checkpoint yang menurunkan metrik akan ditolak."""
    req = AspectRetrainRequest(
        training_data=[{'text': 'harga mahal', 'aspects': ['harga']}] * 10
    )

    assert req.force is False


def test_force_bisa_diaktifkan():
    req = AspectRetrainRequest(
        training_data=[{'text': 'harga mahal', 'aspects': ['harga']}] * 10,
        force=True,
    )

    assert req.force is True


def test_endpoint_retrain_menolak_payload_kurang(client):
    resp = client.post('/api/retrain/sentiment', json={
        'training_data': [{'text': 'halo', 'label': 'positive'}]
    })

    assert resp.status_code == 422


# ── Konfigurasi checkpoint ──────────────────────────────────────────────────

def test_path_checkpoint_diturunkan_dari_model_path():
    s = Settings(model_path='./custom_models')

    assert s.sentiment_checkpoint_path == os.path.join('./custom_models', 'sentiment_retrained')
    assert s.aspect_checkpoint_path == os.path.join('./custom_models', 'aspect_retrained.pt')


def test_seed_retraining_punya_default_tetap():
    assert Settings().retrain_seed == 42


def test_model_path_dapat_dioverride_lewat_env(monkeypatch):
    """MODEL_PATH di .env sebelumnya tidak berefek apa pun."""
    monkeypatch.setenv('MODEL_PATH', './volume/models')

    assert Settings().model_path == './volume/models'


# ── Preview kesiapan data ───────────────────────────────────────────────────

def test_preview_melaporkan_ketimpangan_dan_baseline(client):
    """
    Data koreksi nyata pada skripsi ini 89% negatif; admin harus melihat bahwa
    menebak kelas mayoritas saja sudah mencapai akurasi setinggi itu.
    """
    data = (
        [{'text': f'buruk sekali {i}', 'label': 'negative'} for i in range(89)]
        + [{'text': f'lumayan {i}', 'label': 'neutral'} for i in range(8)]
        + [{'text': f'bagus {i}', 'label': 'positive'} for i in range(3)]
    )

    body = client.post('/api/retrain/preview', json={
        'model_type': 'sentiment', 'training_data': data
    }).json()

    assert body['label_distribution']['counts']['negative'] == 89
    assert body['majority_baseline_accuracy'] == pytest.approx(0.89, abs=0.01)
    assert any('timpang' in w for w in body['warnings'])
    assert any("'positive'" in w for w in body['warnings'])


def test_preview_memberi_bobot_terbesar_ke_kelas_langka(client):
    data = (
        [{'text': f'buruk {i}', 'label': 'negative'} for i in range(50)]
        + [{'text': f'bagus {i}', 'label': 'positive'} for i in range(5)]
    )

    body = client.post('/api/retrain/preview', json={
        'model_type': 'sentiment', 'training_data': data
    }).json()

    assert body['class_weights']['positive'] > body['class_weights']['negative']


def test_preview_data_seimbang_tanpa_peringatan_ketimpangan(client):
    data = (
        [{'text': f'buruk {i}', 'label': 'negative'} for i in range(30)]
        + [{'text': f'lumayan {i}', 'label': 'neutral'} for i in range(30)]
        + [{'text': f'bagus {i}', 'label': 'positive'} for i in range(30)]
    )

    body = client.post('/api/retrain/preview', json={
        'model_type': 'sentiment', 'training_data': data
    }).json()

    assert body['warnings'] == []


def test_preview_menghitung_sampel_tidak_valid(client):
    data = [
        {'text': 'valid', 'label': 'negative'},
        {'text': '', 'label': 'negative'},
        {'text': 'label ngawur', 'label': 'entahlah'},
    ]

    body = client.post('/api/retrain/preview', json={
        'model_type': 'sentiment', 'training_data': data
    }).json()

    assert body['samples_valid'] == 1
    assert body['samples_invalid'] == 2


def test_preview_model_type_tidak_dikenal_ditolak(client):
    resp = client.post('/api/retrain/preview', json={
        'model_type': 'topic', 'training_data': []
    })

    assert resp.status_code == 422


def test_preview_tidak_menyentuh_batas_sepuluh_sampel(client):
    """Preview harus tetap bisa dipanggil justru ketika data belum cukup."""
    resp = client.post('/api/retrain/preview', json={
        'model_type': 'sentiment',
        'training_data': [{'text': 'halo', 'label': 'positive'}],
    })

    assert resp.status_code == 200


# ── Endpoint combined meneruskan mode & daftar aspek ────────────────────────
#
# Skema TextAnalysisRequest sudah lama punya predefined_aspects dan mode, tapi
# endpoint /api/analyze/combined tidak pernah meneruskannya ke aspect_service.
# Akibatnya analisis gabungan SELALU memakai mode automatic dan daftar aspek
# yang dideklarasikan pengguna dibuang diam-diam - tanpa error, dan tanpa
# terlihat di antarmuka. Formulir Laravel menampilkan pilihan mode aspek untuk
# tipe 'combined' juga, jadi ini jalur yang benar-benar dipakai.

def test_combined_meneruskan_mode_dan_predefined_aspects(client, monkeypatch):
    import app.main as main

    tercatat = {}

    async def aspect_palsu(texts, preprocessing_config=None,
                           predefined_aspects=None, mode='automatic'):
        tercatat['predefined_aspects'] = predefined_aspects
        tercatat['mode'] = mode
        return {'aspects': [], 'aspect_sentiments': [], 'statistics': {},
                'summary': '', 'document_aspects': [[] for _ in texts]}

    async def sentimen_palsu(texts, preprocessing_config=None):
        return {'predictions': [], 'distribution': {}, 'summary': ''}

    async def topik_palsu(texts, preprocessing_config=None, num_topics=5):
        return {'topics': [], 'document_topics': [-1] * len(texts),
                'num_topics': 0, 'summary': ''}

    monkeypatch.setattr(main.aspect_service, 'analyze', aspect_palsu)
    monkeypatch.setattr(main.sentiment_service, 'analyze', sentimen_palsu)
    monkeypatch.setattr(main.topic_service, 'analyze', topik_palsu)

    respons = client.post('/api/analyze/combined', json={
        'texts': ['Antriannya lama sekali'],
        'mode': 'rule-based',
        'predefined_aspects': ['pelayanan', 'harga'],
    })

    assert respons.status_code == 200
    assert tercatat['mode'] == 'rule-based'
    assert tercatat['predefined_aspects'] == ['pelayanan', 'harga']


def test_combined_tanpa_mode_tetap_automatic(client, monkeypatch):
    import app.main as main

    tercatat = {}

    async def aspect_palsu(texts, preprocessing_config=None,
                           predefined_aspects=None, mode='automatic'):
        tercatat['mode'] = mode
        return {'aspects': [], 'aspect_sentiments': [], 'statistics': {},
                'summary': '', 'document_aspects': [[] for _ in texts]}

    async def sentimen_palsu(texts, preprocessing_config=None):
        return {'predictions': [], 'distribution': {}, 'summary': ''}

    async def topik_palsu(texts, preprocessing_config=None, num_topics=5):
        return {'topics': [], 'document_topics': [-1] * len(texts),
                'num_topics': 0, 'summary': ''}

    monkeypatch.setattr(main.aspect_service, 'analyze', aspect_palsu)
    monkeypatch.setattr(main.sentiment_service, 'analyze', sentimen_palsu)
    monkeypatch.setattr(main.topic_service, 'analyze', topik_palsu)

    respons = client.post('/api/analyze/combined', json={'texts': ['Halo dunia']})

    assert respons.status_code == 200
    assert tercatat['mode'] == 'automatic'


def test_num_topics_nol_diteruskan_apa_adanya(client, monkeypatch):
    """
    `request.num_topics or 5` menelan nilai 0.

    Nol berarti "pilih jumlah topik otomatis". Dengan `or`, permintaan itu
    diam-diam dijalankan sebagai 5 topik - dan bug ini tidak terlihat dari luar
    karena hasilnya identik persis dengan permintaan num_topics=5.
    """
    import app.main as main

    tercatat = {}

    async def topik_palsu(texts, preprocessing_config=None, num_topics=5):
        tercatat['num_topics'] = num_topics
        return {'topics': [], 'document_topics': [-1] * len(texts),
                'num_topics': 0, 'summary': '', 'word_frequencies': []}

    monkeypatch.setattr(main.topic_service, 'analyze', topik_palsu)

    respons = client.post('/api/analyze/topic',
                          json={'texts': ['pajak naik'], 'num_topics': 0})

    assert respons.status_code == 200
    assert tercatat['num_topics'] == 0


def test_num_topics_tidak_dikirim_tetap_lima(client, monkeypatch):
    import app.main as main

    tercatat = {}

    async def topik_palsu(texts, preprocessing_config=None, num_topics=5):
        tercatat['num_topics'] = num_topics
        return {'topics': [], 'document_topics': [-1] * len(texts),
                'num_topics': 0, 'summary': '', 'word_frequencies': []}

    monkeypatch.setattr(main.topic_service, 'analyze', topik_palsu)
    client.post('/api/analyze/topic', json={'texts': ['pajak naik']})

    assert tercatat['num_topics'] == 5


def test_satu_topik_ditolak(client):
    """1 topik tidak bermakna, dan 0 sudah dipakai untuk mode otomatis."""
    respons = client.post('/api/analyze/topic',
                          json={'texts': ['pajak naik'], 'num_topics': 1})

    assert respons.status_code == 422


# ── Kontrak keluaran sentimen lewat endpoint ────────────────────────────────

def test_endpoint_sentimen_menjaga_penjajaran_dan_menandai_baris_kosong(
    client, monkeypatch
):
    """
    Baris kosong harus kembali sebagai netral bertanda `empty`, bukan dibuang
    dan bukan pula diprediksi. Model asli mengembalikan `positive` 0,9429 untuk
    string kosong, sehingga setiap baris kosong pada CSV dulu menaikkan
    persentase positif yang ditampilkan Laravel.

    Jalur cadangan dipakai agar tes tidak mengunduh bobot IndoBERT.
    """
    import app.main as main

    monkeypatch.setattr(main.sentiment_service, '_ensure_loaded', lambda: None)
    monkeypatch.setattr(main.sentiment_service, 'model', None)
    monkeypatch.setattr(main.sentiment_service, 'tokenizer', None)

    texts = ['pelayanannya bagus sekali', '', 'harganya mahal', '   ']
    resp = client.post('/api/analyze/sentiment', json={'texts': texts})

    assert resp.status_code == 200
    hasil = resp.json()['results']

    assert len(hasil['predictions']) == len(texts)
    assert hasil['predictions'][1]['method'] == 'empty'
    assert hasil['predictions'][3]['method'] == 'empty'
    assert hasil['metrics']['total_texts'] == 4
    assert hasil['metrics']['total_empty'] == 2


def test_endpoint_sentimen_mengembalikan_teks_asli_dan_hasil_pembersihan(
    client, monkeypatch
):
    import app.main as main

    monkeypatch.setattr(main.sentiment_service, '_ensure_loaded', lambda: None)
    monkeypatch.setattr(main.sentiment_service, 'model', None)
    monkeypatch.setattr(main.sentiment_service, 'tokenizer', None)

    asli = 'Pelayanannya BAGUS sekali!!!'
    resp = client.post('/api/analyze/sentiment', json={
        'texts': [asli],
        'preprocessing_config': {'case_folding': True, 'remove_punctuation': True},
    })

    prediksi = resp.json()['results']['predictions'][0]
    assert prediksi['text'] == asli
    assert prediksi['processed_text'] == 'pelayanannya bagus sekali'


def test_endpoint_sentimen_menyertakan_antrean_peninjauan(client, monkeypatch):
    """Laravel memakai blok ini untuk mengurutkan baris yang perlu dikoreksi."""
    import app.main as main

    monkeypatch.setattr(main.sentiment_service, '_ensure_loaded', lambda: None)
    monkeypatch.setattr(main.sentiment_service, 'model', None)
    monkeypatch.setattr(main.sentiment_service, 'tokenizer', None)

    resp = client.post('/api/analyze/sentiment', json={'texts': ['pelayanannya bagus']})

    antrean = resp.json()['results']['review_queue']
    assert set(antrean) == {'threshold', 'count', 'share', 'indices'}


def test_health_melaporkan_provenance_model_sentimen(client):
    """
    Hasil analisis lama harus bisa ditelusuri ke model yang menghasilkannya.
    Karena kandidat model bisa ditukar lewat konfigurasi setelah evaluasi studi
    kasus, /health menyebut bobot dasar, suhu kalibrasi, dan urutan labelnya.
    """
    models = client.get('/health').json()['models']

    assert models['sentiment_base'] == 'indobenchmark/indobert-base-p1'
    assert models['sentiment_temperature'] > 1.0
    assert 0.5 <= models['sentiment_review_threshold'] <= 0.999
    assert set(models['sentiment_label_map'].values()) == {
        'positive', 'neutral', 'negative'
    }


# ── Pratinjau preprocessing per profil ──────────────────────────────────────

AGRESIF = {'case_folding': True, 'remove_punctuation': True,
           'remove_stopwords': True, 'stemming': True}


def test_preprocess_tanpa_task_memakai_config_apa_adanya(client):
    resp = client.post('/api/preprocess', json={
        'texts': ['Pelayanannya tidak bagus'], 'config': AGRESIF,
    })

    assert resp.status_code == 200
    assert resp.json()['task'] is None


def test_preprocess_profil_transformer_menolak_stemming(client):
    """
    Tanpa parameter ini pratinjau menyesatkan: pengguna mengaktifkan stemming,
    melihat teks ter-stem, lalu menjalankan analisis sentimen yang diam-diam
    mematikannya.
    """
    resp = client.post('/api/preprocess', json={
        'texts': ['Pelayanannya tidak bagus'],
        'config': AGRESIF, 'task': 'transformer',
    })

    body = resp.json()
    assert 'tidak' in body['preprocessed'][0]
    assert 'pelayanannya' in body['preprocessed'][0]
    assert body['applied_policy']['stemming'] is False


def test_preprocess_profil_bag_of_words_memaksa_stemming(client):
    resp = client.post('/api/preprocess', json={
        'texts': ['Pelayanannya tidak bagus'],
        'config': {'stemming': False, 'remove_stopwords': False},
        'task': 'bag_of_words',
    })

    body = resp.json()
    assert body['applied_policy']['stemming'] is True
    assert 'layan' in body['preprocessed'][0]


def test_preprocess_profil_span_mengembalikan_teks_utuh(client):
    """Ekstraksi aspek memakai offset karakter; teks tidak boleh berubah."""
    asli = 'Pelayanannya tidak bagus!'
    resp = client.post('/api/preprocess', json={
        'texts': [asli], 'config': AGRESIF, 'task': 'span',
    })

    assert resp.json()['preprocessed'] == [asli]


def test_preprocess_task_tidak_dikenal_ditolak(client):
    resp = client.post('/api/preprocess', json={
        'texts': ['halo'], 'task': 'entah-apa',
    })

    assert resp.status_code == 422


def test_preprocess_menjaga_jumlah_teks(client):
    resp = client.post('/api/preprocess', json={
        'texts': ['satu dua', '', 'tiga empat'], 'task': 'bag_of_words',
    })

    body = resp.json()
    assert len(body['preprocessed']) == 3
    assert body['original_count'] == body['processed_count'] == 3


# ── Kesiapan model & pemanasan ──────────────────────────────────────────────

def test_health_melaporkan_bobot_sudah_dimuat_atau_belum(client):
    """
    Bobot dimuat malas, jadi "service tersedia" tidak sama dengan "model siap".
    Tanpa blok ini Laravel tidak bisa membedakan cold start - yang di kontainer
    baru bisa memakan menit - dari analisis yang benar-benar menggantung.
    """
    body = client.get('/health').json()

    assert set(body['weights_loaded']) == {'sentiment', 'aspect', 'topic'}
    for nilai in body['weights_loaded'].values():
        assert isinstance(nilai, bool)


def test_warmup_terdaftar_dan_melaporkan_per_model(client, monkeypatch):
    """Pemanasan memindahkan biaya muat model keluar dari permintaan pengguna."""
    import app.main as main

    for service in (main.sentiment_service, main.aspect_service, main.topic_service):
        monkeypatch.setattr(service, '_ensure_loaded', lambda: None, raising=False)

    resp = client.post('/api/warmup')

    assert resp.status_code == 200
    body = resp.json()
    assert set(body['models']) == {'sentiment', 'aspect', 'topic'}
    assert 'total_seconds' in body


def test_warmup_tidak_menjatuhkan_service_saat_model_gagal(client, monkeypatch):
    """
    Jalur analisis punya rantai cadangan sendiri; kegagalan memuat model tidak
    boleh membuat endpoint pemanasan melempar 500.
    """
    import app.main as main

    def gagal():
        raise RuntimeError('simulasi gagal unduh bobot')

    monkeypatch.setattr(main.sentiment_service, '_ensure_loaded', gagal)

    resp = client.post('/api/warmup')

    assert resp.status_code == 200
    assert 'error' in resp.json()['models']['sentiment']


def test_batas_jumlah_teks_diambil_dari_settings():
    """
    Batas ini dulu ditulis langsung sebagai 10000 di skema sementara
    `max_batch_size` tidak pernah dibaca - knob yang tampak ada tapi mati.
    """
    from app.config import settings

    with pytest.raises(Exception):
        TextAnalysisRequest(texts=['x'] * (settings.max_batch_size + 1))

    assert TextAnalysisRequest(texts=['x'] * 10).texts == ['x'] * 10
