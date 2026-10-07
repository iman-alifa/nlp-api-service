"""
Kontrak keluaran dan jalur cadangan modul analisis sentimen.

Modul ini sebelumnya nyaris tanpa tes langsung, dan akibatnya sebuah cacat
lolos cukup lama: teks kosong diteruskan ke IndoBERT, yang mengklasifikasi
urutan `[CLS] [SEP]` sebagai `positive` dengan keyakinan 0,9429. Setiap baris
kosong pada CSV karena itu menaikkan persentase positif - angka utama yang
ditampilkan Laravel.

Semua tes di sini berjalan TANPA memuat bobot model: `_ensure_loaded`
di-patch supaya service jatuh ke jalur daftar kata, sehingga suite tetap
cepat dan tidak butuh jaringan.
"""

import pytest

from app.services.sentiment_service import LABEL_MAP, LABEL_TO_ID, SentimentService


LARAVEL_CONFIG = {
    'case_folding': True,
    'remove_punctuation': True,
    'remove_numbers': False,
    'remove_stopwords': True,
    'stemming': True,
    'lemmatization': False,
    'custom_stopwords': [],
}


@pytest.fixture
def service(monkeypatch):
    """SentimentService yang dipaksa memakai jalur cadangan (tanpa bobot)."""
    svc = SentimentService.__new__(SentimentService)
    svc.model = None
    svc.tokenizer = None
    svc.device = 'cpu'
    svc._loaded = True
    svc.model_name = 'dummy'
    svc.base_model_name = 'dummy'
    svc.using_retrained = False
    svc.temperature = 1.0
    svc.label_map = dict(LABEL_MAP)
    monkeypatch.setattr(svc, '_ensure_loaded', lambda: None)
    return svc


async def analyze(svc, texts, config=LARAVEL_CONFIG):
    return await svc.analyze(texts=texts, preprocessing_config=config)


# ── Peta label ──────────────────────────────────────────────────────────────

def test_peta_label_sesuai_model_dasar():
    """
    mdhugol/indonesia-bert-sentiment-classification memakai urutan
    LABEL_0=positive, LABEL_1=neutral, LABEL_2=negative. Mengurutkannya ulang
    membalik seluruh hasil tanpa galat apa pun.
    """
    assert LABEL_MAP == {0: 'positive', 1: 'neutral', 2: 'negative'}
    assert LABEL_TO_ID['positive'] == 0
    assert LABEL_TO_ID['negative'] == 2


# ── Kontrak penjajaran: N masuk = N keluar ──────────────────────────────────

@pytest.mark.asyncio
async def test_jumlah_prediksi_sama_dengan_jumlah_teks(service):
    texts = ['pelayanannya bagus', '', 'harganya mahal sekali', '   ']
    hasil = await analyze(service, texts)

    assert len(hasil['predictions']) == len(texts)


@pytest.mark.asyncio
async def test_urutan_prediksi_mengikuti_urutan_masukan(service):
    texts = ['pelayanannya bagus sekali', 'harganya mahal dan mengecewakan']
    hasil = await analyze(service, texts)

    assert hasil['predictions'][0]['sentiment'] == 'positive'
    assert hasil['predictions'][1]['sentiment'] == 'negative'


# ── Teks kosong ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_teks_kosong_jadi_netral_bukan_positif(service):
    """Regresi: model lama mengembalikan positive 0.9429 untuk string kosong."""
    hasil = await analyze(service, ['', '   '])

    for prediksi in hasil['predictions']:
        assert prediksi['sentiment'] == 'neutral'
        assert prediksi['confidence'] == 0.0
        assert prediksi['method'] == 'empty'


@pytest.mark.asyncio
async def test_teks_yang_menjadi_kosong_setelah_dibersihkan_juga_ditandai(service):
    """
    Pembersihan bisa MENGHASILKAN teks kosong dari masukan yang tidak kosong,
    misalnya emoji atau tanda baca saja. Pemeriksaan karena itu harus terjadi
    setelah preprocessing, bukan sebelumnya.
    """
    hasil = await analyze(service, ['😀', '!!!'])

    for prediksi in hasil['predictions']:
        assert prediksi['method'] == 'empty'
    assert hasil['predictions'][0]['text'] == '😀'


@pytest.mark.asyncio
async def test_token_pendek_tidak_dibuang_pada_jalur_transformer(service):
    """
    Profil 'transformer' memakai min_token_length=1. Dulu ambangnya 2 untuk
    semua jalur, sehingga "a", angka, dan kata sependek "di" lenyap - padahal
    IndoBERT justru memakai kata fungsi dan angka sebagai sinyal.
    """
    hasil = await analyze(service, ['nilai a naik 5 persen di kota'])

    diproses = hasil['predictions'][0]['processed_text'].split()
    assert 'a' in diproses
    assert '5' in diproses
    assert 'di' in diproses


@pytest.mark.asyncio
async def test_baris_kosong_tidak_mencemari_distribusi(service):
    """Baris kosong tidak boleh ikut menjadi penyebut persentase."""
    hasil = await analyze(service, ['pelayanannya bagus sekali', '', '', ''])

    assert hasil['distribution']['positive'] == 100.0
    assert hasil['metrics']['total_texts'] == 4
    assert hasil['metrics']['total_empty'] == 3
    assert hasil['metrics']['total_analyzed'] == 1


# ── Bentuk keluaran ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_setiap_prediksi_menyebut_metodenya(service):
    """
    Tanpa `method`, hasil IndoBERT dan tebakan daftar kata tidak bisa
    dibedakan pemanggil - degradasi diam-diam.
    """
    hasil = await analyze(service, ['pelayanannya bagus', ''])

    assert hasil['predictions'][0]['method'] == 'rule-based'
    assert hasil['predictions'][1]['method'] == 'empty'


@pytest.mark.asyncio
async def test_semua_jalur_mengembalikan_kunci_scores(service):
    """Blade Laravel membaca `scores` tanpa membedakan jalur mana yang jalan."""
    hasil = await analyze(service, ['pelayanannya bagus', 'harganya mahal', ''])

    for prediksi in hasil['predictions']:
        assert set(prediksi['scores']) == {'positive', 'neutral', 'negative'}


@pytest.mark.asyncio
async def test_text_adalah_teks_asli_bukan_hasil_pembersihan(service):
    """
    Sebelumnya `text` berisi hasil pembersihan, sehingga "wkwkwk anjay bang"
    kembali sebagai "tertawa bang". Laravel menambalnya sendiri; API kini
    mengembalikan keduanya secara eksplisit.
    """
    asli = 'Pelayanannya BAGUS sekali!!!'
    hasil = await analyze(service, [asli])

    assert hasil['predictions'][0]['text'] == asli
    assert hasil['predictions'][0]['processed_text'] != asli


# ── Jalur cadangan: negasi ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_negasi_membalik_polaritas_pada_jalur_cadangan(service):
    """
    Versi lama memasukkan 'tidak' ke daftar kata negatif, sehingga
    "tidak bagus" berakhir seri (1 positif, 1 negatif) lalu dinilai netral.
    """
    hasil = await analyze(service, ['pelayanannya tidak bagus'])

    assert hasil['predictions'][0]['sentiment'] == 'negative'


@pytest.mark.asyncio
async def test_negasi_atas_kata_negatif_menjadi_positif(service):
    hasil = await analyze(service, ['pelayanannya tidak mengecewakan'])

    assert hasil['predictions'][0]['sentiment'] == 'positive'


def test_negasi_tidak_dihitung_sebagai_kata_negatif():
    """Negator membalik polaritas; ia bukan sentimen negatif dengan sendirinya."""
    tumpang_tindih = SentimentService.NEGATION_WORDS & SentimentService.NEGATIVE_WORDS
    assert not tumpang_tindih, f'negator ikut jadi kata negatif: {tumpang_tindih}'


def test_daftar_kata_tidak_saling_tumpang_tindih():
    tumpang_tindih = SentimentService.POSITIVE_WORDS & SentimentService.NEGATIVE_WORDS
    assert not tumpang_tindih, f'kata muncul di dua daftar: {tumpang_tindih}'


@pytest.mark.asyncio
async def test_jangkauan_negasi_tidak_melewati_dua_kata(service):
    """
    "tidak" hanya membalik dua kata sesudahnya; kata jauh setelahnya tetap
    dibaca apa adanya, agar satu negasi tidak membalik seluruh kalimat.
    """
    hasil = await analyze(
        service, ['tidak ada masalah apa pun dan pelayanannya sangat memuaskan']
    )

    assert hasil['predictions'][0]['sentiment'] == 'positive'


# ── Kebijakan preprocessing ─────────────────────────────────────────────────

def test_sanitize_config_selalu_mematikan_stemming_dan_stopword():
    """
    Terukur pada SmSA test (500 dokumen): konfigurasi Laravel yang meminta
    stemming + stopword removal menghasilkan angka yang IDENTIK dengan
    konfigurasi yang tidak memintanya (0,9080 / macro F1 0,8762), yang hanya
    mungkin bila pengaman ini benar-benar bekerja.
    """
    config = SentimentService._sanitize_config(LARAVEL_CONFIG)

    assert config['stemming'] is False
    assert config['remove_stopwords'] is False
    assert config['lemmatization'] is False


def test_sanitize_config_mempertahankan_normalisasi_slang():
    """
    Normalisasi slang/typo justru MENAIKKAN akurasi: terukur 0,9040 -> 0,9080
    di atas case folding + pembuangan tanda baca. Ia tidak boleh ikut dimatikan.
    """
    config = SentimentService._sanitize_config(LARAVEL_CONFIG)

    assert config.get('normalize_slang', True) is not False
    assert config.get('fix_typos', True) is not False


# ── Ringkasan ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_ringkasan_menghitung_hanya_teks_yang_dinilai(service):
    hasil = await analyze(service, ['pelayanannya bagus sekali', '', ''])

    assert 'Dari 1 teks' in hasil['summary']


# ── Antrean peninjauan (jembatan ke active learning) ────────────────────────

def _prediksi(confidence, method='indobert'):
    return {'text': 't', 'sentiment': 'neutral', 'confidence': confidence,
            'scores': {'positive': 0.0, 'neutral': 1.0, 'negative': 0.0},
            'method': method}


def test_antrean_peninjauan_menandai_confidence_rendah():
    """
    Terukur pada SmSA test: akurasi 0,9847 pada confidence >= 0,99 tetapi hanya
    0,4340 di bawah 0,70. Baris berkeyakinan rendah karena itu yang paling
    layak dikoreksi manusia.
    """
    from app.config import settings

    ambang = settings.sentiment_review_threshold
    antrean = SentimentService._build_review_queue([
        _prediksi(ambang + 0.05), _prediksi(0.55),
        _prediksi(ambang + 0.01), _prediksi(0.72),
    ])

    assert antrean['threshold'] == ambang
    assert antrean['count'] == 2
    assert set(antrean['indices']) == {1, 3}
    assert antrean['share'] == 0.5


def test_antrean_peninjauan_terurut_dari_paling_tidak_yakin():
    """Yang paling meragukan harus ikut terbawa walau daftarnya dipotong."""
    antrean = SentimentService._build_review_queue([
        _prediksi(0.85), _prediksi(0.40), _prediksi(0.70),
    ])

    assert antrean['indices'] == [1, 2, 0]


def test_antrean_peninjauan_mengabaikan_baris_kosong():
    """
    Baris kosong berkeyakinan 0,0 tetapi tidak ada yang bisa dikoreksi darinya;
    memasukkannya hanya membanjiri antrean dengan baris tanpa isi.
    """
    from app.config import settings

    antrean = SentimentService._build_review_queue([
        _prediksi(settings.sentiment_review_threshold + 0.05),
        _prediksi(0.0, method='empty'), _prediksi(0.0, method='empty'),
    ])

    assert antrean['count'] == 0
    assert antrean['share'] == 0.0


def test_antrean_peninjauan_aman_saat_semua_baris_kosong():
    """Pembagian dengan nol pernah menjatuhkan jalur serupa di modul lain."""
    from app.config import settings

    antrean = SentimentService._build_review_queue([_prediksi(0.0, method='empty')])

    assert antrean == {'threshold': settings.sentiment_review_threshold,
                       'count': 0, 'share': 0.0, 'indices': []}


@pytest.mark.asyncio
async def test_antrean_peninjauan_ikut_pada_hasil_analyze(service):
    """
    Jalur cadangan memberi confidence paling tinggi 0,95 dan lazimnya 0,7,
    sehingga hampir semua barisnya masuk antrean - memang seperti itu yang
    diinginkan: mutunya terukur macro F1 0,65 dan pantas ditinjau manusia.
    """
    hasil = await analyze(service, ['pelayanannya bagus sekali', ''])

    assert hasil['review_queue']['count'] == 1
    assert hasil['review_queue']['indices'] == [0]


# ── Kalibrasi confidence ────────────────────────────────────────────────────

def test_suhu_default_diambil_dari_settings(tmp_path):
    """Model dasar HuggingFace tidak membawa calibration.json."""
    from app.config import settings

    svc = SentimentService.__new__(SentimentService)
    assert svc._load_calibration(str(tmp_path)) == settings.sentiment_temperature


def test_suhu_checkpoint_menang_atas_default(tmp_path):
    """
    Suhu adalah milik BOBOT tertentu. Checkpoint hasil retraining punya
    distribusi logit sendiri, sehingga suhu model dasar tidak berlaku untuknya.
    """
    import json

    (tmp_path / 'calibration.json').write_text(
        json.dumps({'temperature': 1.75}), encoding='utf-8'
    )

    svc = SentimentService.__new__(SentimentService)
    assert svc._load_calibration(str(tmp_path)) == 1.75


def test_suhu_checkpoint_rusak_jatuh_ke_default(tmp_path):
    """File rusak tidak boleh menjatuhkan service."""
    from app.config import settings

    (tmp_path / 'calibration.json').write_text('bukan json', encoding='utf-8')

    svc = SentimentService.__new__(SentimentService)
    assert svc._load_calibration(str(tmp_path)) == settings.sentiment_temperature


def test_suhu_tidak_wajar_ditolak(tmp_path):
    """Nilai ekstrem menandakan kalibrasi gagal, bukan kalibrasi yang benar."""
    import json
    from app.config import settings

    (tmp_path / 'calibration.json').write_text(
        json.dumps({'temperature': 0.0}), encoding='utf-8'
    )

    svc = SentimentService.__new__(SentimentService)
    assert svc._load_calibration(str(tmp_path)) == settings.sentiment_temperature


def test_menyimpan_dan_memuat_kalibrasi_bolak_balik(tmp_path):
    svc = SentimentService.__new__(SentimentService)
    SentimentService._save_calibration(str(tmp_path), 1.42, {'val_size': 30})

    assert svc._load_calibration(str(tmp_path)) == 1.42
    assert (tmp_path / 'calibration.json').exists()


def test_suhu_default_lebih_dari_satu():
    """
    Setiap kandidat terukur terlalu yakin, jadi suhunya selalu > 1. Untuk model
    yang dipakai (crypter70) T = 2,7748 dan ECE test turun 0,0827 -> 0,0240
    tanpa mengubah akurasi sama sekali.
    """
    from app.config import settings

    assert settings.sentiment_temperature > 1.0


# ── Ketahanan inferensi ─────────────────────────────────────────────────────

class _BatchPalsu(dict):
    """Tiruan BatchEncoding: dict yang juga punya .to(device)."""

    def to(self, device):  # noqa: D102
        return self


class _TokenizerPalsu:
    """Tokenizer minimal: satu token per kata, tanpa memuat bobot apa pun."""

    def __init__(self, max_tokens):
        self.max_tokens = max_tokens

    def __call__(self, texts, **kwargs):
        import torch

        limit = kwargs.get('max_length', self.max_tokens)
        lengths = [min(len(t.split()), limit) for t in texts]
        widest = max(lengths) if lengths else 1
        mask = torch.zeros(len(texts), widest, dtype=torch.long)
        for i, n in enumerate(lengths):
            mask[i, :n] = 1
        return _BatchPalsu({
            'input_ids': torch.zeros(len(texts), widest, dtype=torch.long),
            'attention_mask': mask,
        })


class _KeluaranPalsu:
    def __init__(self, logits):
        self.logits = logits


def _pasang_model_palsu(svc, gagal_bila_lebih_dari=None):
    """Pasang model tiruan; opsional selalu gagal saat batch lebih besar dari n."""
    import torch

    hitungan = {'panggil': 0}

    class ModelPalsu:
        def __call__(self, **kwargs):
            hitungan['panggil'] += 1
            n = kwargs['input_ids'].shape[0]
            if gagal_bila_lebih_dari is not None and n > gagal_bila_lebih_dari:
                raise RuntimeError('simulasi kehabisan memori')
            return _KeluaranPalsu(torch.tensor([[4.0, 0.0, 0.0]] * n))

    svc.model = ModelPalsu()
    svc.tokenizer = _TokenizerPalsu(svc.MAX_MODEL_TOKENS)
    svc.temperature = 1.0
    return hitungan


@pytest.mark.asyncio
async def test_teks_terpotong_ditandai_dan_dilaporkan(service):
    """
    Tidak satu pun korpus proyek melebihi 512 token, tetapi API menerima teks
    sampai 10 000 karakter. Ekor dokumen panjang tidak boleh hilang tanpa jejak
    - pada ulasan, justru bagian akhir yang memuat kesimpulan penilaian.
    """
    _pasang_model_palsu(service)

    panjang = ' '.join(['kata'] * (service.MAX_MODEL_TOKENS + 50))
    hasil = await service.analyze(texts=['pendek saja', panjang], preprocessing_config=None)

    assert hasil['predictions'][0].get('truncated') is None
    assert hasil['predictions'][1]['truncated'] is True
    assert hasil['metrics']['total_truncated'] == 1


@pytest.mark.asyncio
async def test_chunk_gagal_dibelah_bukan_menjatuhkan_seluruh_batch(service):
    """
    Regresi: satu kegagalan dulu menjatuhkan seluruh 256 teks di dalam chunk
    menjadi netral berkeyakinan 0, padahal lazimnya hanya satu teks bermasalah.
    Chunk kini dibelah dua sampai ukuran satu.
    """
    hitungan = _pasang_model_palsu(service, gagal_bila_lebih_dari=1)

    hasil = await service.analyze(
        texts=['satu dua', 'tiga empat', 'lima enam', 'tujuh delapan'],
        preprocessing_config=None,
    )

    assert all(p['method'] == 'indobert' for p in hasil['predictions'])
    assert hasil['metrics']['total_failed'] == 0
    assert hitungan['panggil'] > 1, 'chunk seharusnya dibelah dan dicoba ulang'


@pytest.mark.asyncio
async def test_teks_yang_benar_benar_rusak_ditandai_error(service):
    """Bila pembelahan sampai satu teks pun tetap gagal, barulah ditandai."""
    _pasang_model_palsu(service, gagal_bila_lebih_dari=0)

    hasil = await service.analyze(texts=['satu dua', 'tiga empat'],
                                  preprocessing_config=None)

    assert all(p['method'] == 'error' for p in hasil['predictions'])
    assert hasil['metrics']['total_failed'] == 2
    # Kontrak penjajaran tetap terjaga walaupun semuanya gagal.
    assert len(hasil['predictions']) == 2


@pytest.mark.asyncio
async def test_urutan_terjaga_walau_chunk_dibelah(service):
    """
    Pembelahan chunk terjadi pada urutan yang SUDAH diurutkan panjang, jadi
    pemetaan balik ke posisi asli harus tetap benar.
    """
    _pasang_model_palsu(service, gagal_bila_lebih_dari=1)

    texts = ['a b c d e', 'f', 'g h i', 'j k']
    hasil = await service.analyze(texts=texts, preprocessing_config=None)

    assert [p['text'] for p in hasil['predictions']] == texts


# ── Registry model & keamanan urutan label ──────────────────────────────────

class _ConfigPalsu:
    def __init__(self, id2label):
        self.id2label = id2label


class _ModelPalsuBerlabel:
    def __init__(self, id2label):
        self.config = _ConfigPalsu(id2label)


def _svc_dengan_config(id2label):
    svc = SentimentService.__new__(SentimentService)
    svc.model = _ModelPalsuBerlabel(id2label)
    svc.model_name = 'uji'
    return svc


def test_urutan_label_dibaca_dari_config_model():
    """Model menyebutkan labelnya sendiri; itu yang dipakai."""
    svc = _svc_dengan_config({0: 'positive', 1: 'neutral', 2: 'negative'})

    assert svc._resolve_label_map() == {0: 'positive', 1: 'neutral', 2: 'negative'}


def test_urutan_label_terbalik_dihormati_bukan_ditimpa():
    """
    `taufiqdp/indonesian-sentiment` memakai 0=negatif, kebalikan dari bawaan.
    Memaksakan peta bawaan akan membalik SELURUH hasil tanpa galat apa pun -
    kesalahan yang hanya terlihat dari mutu prediksi yang anjlok.
    """
    svc = _svc_dengan_config({0: 'negatif', 1: 'netral', 2: 'positif'})

    assert svc._resolve_label_map() == {0: 'negative', 1: 'neutral', 2: 'positive'}


def test_label_kapital_dan_bahasa_indonesia_dikenali():
    """`ayameRushia` memakai 'Positive', `crypter70` memakai 'POSITIVE'."""
    svc = _svc_dengan_config({0: 'POSITIVE', 1: 'Neutral', 2: 'negatif'})

    assert svc._resolve_label_map() == {0: 'positive', 1: 'neutral', 2: 'negative'}


def test_config_tidak_informatif_jatuh_ke_peta_bawaan():
    """`mdhugol` masih memakai LABEL_0/1/2; peta bawaan yang benar untuknya."""
    svc = _svc_dengan_config({0: 'LABEL_0', 1: 'LABEL_1', 2: 'LABEL_2'})

    assert svc._resolve_label_map() == LABEL_MAP


def test_config_dengan_label_duplikat_ditolak():
    """Dua id memetakan ke label yang sama = config rusak, jangan dipercaya."""
    svc = _svc_dengan_config({0: 'positive', 1: 'positive', 2: 'negative'})

    assert svc._resolve_label_map() == LABEL_MAP


def test_label_to_id_konsisten_dengan_label_map():
    """
    Jalur retraining memakai `label_to_id` untuk mengubah label koreksi menjadi
    id kelas. Kalau tidak konsisten, model dilatih dengan label tertukar -
    loss tetap turun dan tidak ada galat, tetapi hasilnya kacau.
    """
    svc = SentimentService.__new__(SentimentService)
    svc.label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}

    assert svc.label_to_id == {'negative': 0, 'neutral': 1, 'positive': 2}


# ── Registry kandidat ───────────────────────────────────────────────────────

def test_model_yang_dipakai_terdaftar_di_registry():
    from app.config import SENTIMENT_MODELS, settings

    assert settings.sentiment_base_model in SENTIMENT_MODELS


def test_model_yang_dipakai_berbasis_indobert():
    """
    Seminar proposal mengajukan IndoBERT, jadi model dasarnya tidak boleh
    berpindah arsitektur diam-diam. `w11wo` sengaja disimpan di registry sebagai
    cadangan meski BUKAN IndoBERT - pilihannya harus disadari, bukan kebetulan.
    """
    from app.config import SENTIMENT_MODELS, settings

    profil = SENTIMENT_MODELS[settings.sentiment_base_model]
    assert profil['base'] == 'indobenchmark/indobert-base-p1'


def test_setiap_kandidat_punya_profil_lengkap():
    from app.config import SENTIMENT_MODELS

    for nama, profil in SENTIMENT_MODELS.items():
        assert profil.get('base'), f'{nama} tanpa model dasar'
        assert profil.get('note'), f'{nama} tanpa keterangan'
        assert 0.5 <= profil['review_threshold'] <= 0.999, nama
        if profil.get('temperature') is not None:
            assert 0.5 <= profil['temperature'] <= 5.0, nama


def test_suhu_dan_ambang_mengikuti_model_terpilih():
    """
    Mengganti model tanpa mengganti kalibrasinya membuat confidence salah
    kalibrasi, dan antrean koreksi ikut salah sasaran - tanpa galat apa pun.
    """
    from app.config import SENTIMENT_MODELS, settings

    profil = SENTIMENT_MODELS[settings.sentiment_base_model]
    assert settings.sentiment_temperature == profil['temperature']
    assert settings.sentiment_review_threshold == profil['review_threshold']


# ── Kalibrasi diwarisi, bukan direset ───────────────────────────────────────

def test_suhu_lama_diwarisi_ketika_validasi_terlalu_kecil():
    """"Tidak bisa mengukur T baru" bukan berarti "tanpa kalibrasi".

    `fit_temperature` mengembalikan 1.0 ketika kalibrasi tidak layak. Menulis
    1.0 apa adanya MEMBUANG suhu yang sudah terukur pada ribuan sampel dan
    menggantinya dengan asumsi model terkalibrasi sempurna - asumsi yang
    diketahui salah untuk keluarga model ini (ECE mentah 0,0827).

    Terukur pada studi kasus: retrain 218 koreksi menyisakan validasi 44 baris,
    penjaga menyala, T 2,7748 -> 1,0, rerata keyakinan melonjak 0,8929 ->
    0,9497, dan antrean tinjauan menyusut 109 -> 58 baris dengan kesalahan
    tertangkap turun 66,7% -> 53,7%.
    """
    import inspect

    from app.config import settings
    from app.services.sentiment_service import SentimentService

    sumber = inspect.getsource(SentimentService._retrain_sync)

    assert 'diwarisi' in sumber, 'pewarisan suhu hilang dari jalur retraining'
    assert 'calibration_min_samples' in sumber
    # Suhu lama HARUS dipakai saat diwarisi, bukan hasil fit yang 1.0.
    assert 'self.temperature if diwarisi else fitted_temperature' in sumber
    assert settings.calibration_min_samples >= 50


def test_pewarisan_hanya_saat_penjaga_menyala():
    """Suhu hasil pengukuran yang sah tidak boleh ditimpa suhu lama.

    Syaratnya tiga sekaligus: hasil fit tepat 1.0, suhu lama bukan 1.0, DAN
    set validasi memang di bawah ambang. Kalau salah satu tidak terpenuhi,
    hasil pengukuranlah yang dipakai.
    """
    import inspect

    from app.services.sentiment_service import SentimentService

    sumber = inspect.getsource(SentimentService._retrain_sync)
    blok = sumber[sumber.index('diwarisi = ('):sumber.index('new_temperature =')]

    assert 'fitted_temperature == 1.0' in blok
    assert 'self.temperature != 1.0' in blok
    assert 'len(val_idx) < settings.calibration_min_samples' in blok
