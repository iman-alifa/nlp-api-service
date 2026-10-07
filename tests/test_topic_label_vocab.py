"""
Test penyaringan kosakata label topik.

Masalah yang ditanganinya terlihat langsung pada korpus produksi: dari 18 topik,
enam berlabel kata yang hanya muncul sekali - `chuaaakksss`, `preetttt`,
`membaaaanguuuuun`, `indonesiiiiaaaa`, `mahpuuudd`, `gatel`. Itu klaster salah
ketik dan kata yang dipanjang-panjangkan, bukan topik.

Penyebabnya struktural, bukan kebetulan: c-TF-IDF memberi bobot TERTINGGI pada
kata yang jarang, sehingga hapax justru menang di topiknya sendiri, dan MMR
memperkuatnya karena kata langka selalu tampak melengkapi kata lain. Menambah
stopword tidak bisa menyelesaikannya - salah ketik tidak mungkin didaftar satu
per satu.

Yang WAJIB dijaga: penyaringan ini hanya menyentuh pemilihan kata label.
Pengelompokan dokumen memakai embedding atas teks penuh dan tidak boleh berubah.

Tidak memuat bobot model.
"""

import pytest

from app.config import settings
from app.services.topic_service import TOPIC_KEYWORD_STOPWORDS, TopicService


@pytest.fixture(scope='module')
def svc():
    return TopicService()


def korpus(n_isi=40):
    """Korpus sintetis: kata isi berulang, plus hapax berupa salah ketik."""
    return (
        ['pajak naik terus rakyat susah bayar'] * n_isi
        + ['korupsi merajalela negara rugi besar'] * n_isi
        + ['utang negara triliunan bengkak parah'] * n_isi
        + [
            'chuaaakksss preetttt', 'membaaaanguuuuun naaaaah',
            'indonesiiiiaaaa gatel', 'mahpuuudd lhooo', 'maksdnya samgat',
        ]
    )


@pytest.fixture
def longgar(monkeypatch):
    """Nyalakan penyaringan dan longgarkan ambangnya untuk korpus sintetis.

    Bawaannya MATI (topic_label_min_freq = 1), jadi mekanismenya harus
    dinyalakan sendiri di sini - lihat test_bawaan_penyaringan_mati untuk
    alasan kenapa mati.
    """
    monkeypatch.setattr(settings, 'topic_label_min_freq', 3)
    monkeypatch.setattr(settings, 'topic_label_min_vocab', 5)


# ── Inti: hapax tidak boleh menjadi label ───────────────────────────────────

def test_kata_yang_muncul_sekali_dibuang(svc, longgar):
    v = svc._label_vocabulary(korpus())

    for salah_ketik in ('chuaaakksss', 'preetttt', 'membaaaanguuuuun',
                        'indonesiiiiaaaa', 'gatel', 'mahpuuudd', 'samgat'):
        assert salah_ketik not in v, f'{salah_ketik} masih boleh jadi label'


def test_kata_isi_yang_berulang_tetap_lolos(svc, longgar):
    """Penyaringan tidak boleh ikut membuang kata yang justru paling penting."""
    v = svc._label_vocabulary(korpus())

    for kata in ('pajak', 'korupsi', 'negara', 'utang', 'triliunan', 'rakyat'):
        assert kata in v


def test_stopword_label_tidak_masuk_kosakata(svc, longgar):
    v = svc._label_vocabulary(korpus())

    assert 'terus' not in v
    assert 'susah' not in v


def test_ambang_dihormati(svc, monkeypatch):
    monkeypatch.setattr(settings, 'topic_label_min_vocab', 1)
    # Harus melewati batas topic_label_min_docs, kalau tidak penyaringan mati
    # lebih dulu dan yang teruji bukan ambang frekuensinya.
    docs = ['alpha beta'] * 60 + ['gamma'] * 2 + ['delta']

    monkeypatch.setattr(settings, 'topic_label_min_freq', 2)
    v2 = svc._label_vocabulary(docs)
    assert 'gamma' in v2 and 'delta' not in v2

    monkeypatch.setattr(settings, 'topic_label_min_freq', 3)
    v3 = svc._label_vocabulary(docs)
    assert 'gamma' not in v3 and 'alpha' in v3


# ── Penjaga: kapan penyaringan harus MATI ──────────────────────────────────

def test_mati_pada_korpus_kecil(svc):
    """Pada korpus kecil hampir seluruh kosakata memang hapax; ambang berapa pun
    akan membuang kata terpentingnya. Alasan yang sama dengan min_df."""
    assert svc._label_vocabulary(['pajak naik terus'] * 10) is None


def test_mati_bila_ambang_satu(svc, monkeypatch):
    monkeypatch.setattr(settings, 'topic_label_min_freq', 1)
    assert svc._label_vocabulary(korpus()) is None


def test_mati_bila_kosakata_sisanya_terlalu_tipis(svc):
    """Label dari kosakata yang terlalu sedikit lebih buruk daripada tidak
    disaring sama sekali - jadi penyaringan dibatalkan, bukan dipaksakan."""
    # Nilai bawaan topic_label_min_vocab=50; korpus ini hanya punya belasan kata.
    assert svc._label_vocabulary(korpus()) is None


# ── Vectorizer memakainya hanya di jalur label ─────────────────────────────

def test_vectorizer_tidak_pernah_dibatasi_kosakata(svc, longgar):
    """Penyaringan TIDAK boleh dipasang di vectorizer.

    Versi sebelumnya memasang `vocabulary=` pada vectorizer c-TF-IDF dengan
    alasan "hanya menyentuh label". Alasan itu salah: BERTopic memakai c-TF-IDF
    untuk menggabungkan topik yang mirip dan menarik outlier, sehingga kosakata
    yang dipersempit membuat topik tampak lebih mirip dan lebih banyak dilebur.
    Terukur pada korpus berita n=250: 9 topik (terbesar 22,4%) runtuh menjadi
    2 topik (terbesar 59,6%), dan robustness turun 14/14 -> 13/14.
    """
    docs = korpus()

    for for_ctfidf in (True, False):
        vec = svc._build_vectorizer(len(docs), for_ctfidf=for_ctfidf, docs=docs)
        assert vec.vocabulary is None
        assert vec.stop_words


def test_tanpa_dokumen_vectorizer_tetap_berjalan(svc):
    """Pemanggil lama yang tidak mengirim docs tidak boleh rusak."""
    vec = svc._build_vectorizer(100, for_ctfidf=True)

    assert vec.vocabulary is None
    assert vec.stop_words


def test_penyaring_label_membuang_kata_langka(svc):
    """Penyaringan dilakukan atas keluaran model, bukan lewat vectorizer."""
    pasangan = [('pajak', 0.9), ('chuaaakksss', 0.8), ('korupsi', 0.7)]
    kosakata = {'pajak', 'korupsi'}

    hasil = svc._filter_label_words(pasangan, kosakata, 10)

    assert [w for w, _ in hasil] == ['pajak', 'korupsi']


def test_penyaring_label_membuang_kata_kosong(svc):
    """BERTopic membantali daftarnya sampai top_n_words dengan string kosong.

    Terlihat pada studi kasus Polri: satu topik berlabel
    `keberatan, cepat, , , , , ,` tampil apa adanya di dasbor.
    """
    pasangan = [('pajak', 0.9), ('', 0.5), ('korupsi', 0.3), ('   ', 0.1)]

    hasil = svc._filter_label_words(pasangan, None, 10)

    assert [w for w, _ in hasil] == ['pajak', 'korupsi']


def test_penyaring_label_mempertahankan_pasangan_bobot(svc):
    """Kata dan bobotnya harus dibuang BERSAMAAN; kalau tidak, tiap kata
    memakai bobot milik kata lain."""
    pasangan = [('pajak', 0.9), ('', 0.5), ('korupsi', 0.3)]

    hasil = svc._filter_label_words(pasangan, None, 10)

    assert hasil == [('pajak', 0.9), ('korupsi', 0.3)]


def test_penyaring_label_tidak_mengosongkan_topik(svc):
    """Kalau semua kata sebuah topik langka, label langka masih lebih berguna
    daripada topik tanpa label sama sekali."""
    pasangan = [('langka1', 0.9), ('langka2', 0.8)]

    hasil = svc._filter_label_words(pasangan, {'pajak'}, 10)

    assert [w for w, _ in hasil] == ['langka1', 'langka2']


def test_penyaring_label_menghormati_batas(svc):
    pasangan = [(f'kata{i}', 1.0 - i / 100) for i in range(30)]

    assert len(svc._filter_label_words(pasangan, None, 10)) == 10


# ── Stopword singkatan ─────────────────────────────────────────────────────

def test_singkatan_chat_TIDAK_dipasang_di_stopword_vectorizer():
    """Perluasan stopword sempat ditambahkan, lalu DICABUT setelah diukur.

    Niatnya benar - `dgn`, `utk`, `tsb` memang kata fungsi yang menyamar dan
    sempat mengisi satu topik utuh. Tetapi TOPIC_KEYWORD_STOPWORDS dipasang
    pada vectorizer c-TF-IDF, dan apa pun di sana BUKAN sekadar soal label:
    BERTopic memakai c-TF-IDF untuk menggabungkan topik dan menarik outlier.
    Kelas kesalahan yang sama persis dengan `vocabulary=` yang lebih dulu
    dicabut - dan yang kedua ini sempat luput ketika yang pertama diperbaiki.

    Terukur pada studi kasus: Polri 8 topik -> 5 dengan c_v 0,4736 -> 0,2081,
    DPR 15 -> 17 dengan c_v 0,4858 -> 0,3606.
    """
    for singkatan in ('dgn', 'utk', 'tsb', 'sdh', 'klo', 'gmn', 'bgt', 'smpe'):
        assert singkatan not in TOPIC_KEYWORD_STOPWORDS, (
            f'{singkatan} kembali masuk stopword vectorizer; ukur ulang '
            'robustness DAN keenam korpus studi kasus sebelum mempertahankannya'
        )


def test_bawaan_penyaringan_kosakata_label_mati():
    """Diukur dan ditolak sebagai bawaan.

    Manfaatnya nyata pada korpus pajak (14 dari 20 topik berlabel salah ketik),
    tetapi pada korpus lembaga kata label langka hanya 4-12% sementara biayanya
    terukur mahal. Dinyalakan hanya bila salah ketik benar-benar mendominasi,
    dan robustness diukur ulang sesudahnya.
    """
    assert settings.topic_label_min_freq == 1


def test_kata_isi_tidak_ikut_masuk_stopword():
    """Perluasan stopword tidak boleh menelan kata bermakna."""
    for kata in ('pajak', 'korupsi', 'utang', 'gaji', 'negara', 'pemerintah'):
        assert kata not in TOPIC_KEYWORD_STOPWORDS


# ── Pembersihan wajib, tidak bergantung pemanggil ──────────────────────────

def test_pembersihan_dijalankan_walau_tanpa_preprocessing_config(svc, monkeypatch):
    """Kebijakan 'bag_of_words' harus berlaku SELALU untuk jalur topik.

    Sebelumnya seluruh blok pembersihan dilewati bila `preprocessing_config`
    kosong, sehingga pemodelan topik berjalan di atas teks MENTAH - tanpa
    stopword, tanpa stemming, tanpa normalisasi slang. Laravel selalu mengirim
    config (resolvernya punya FALLBACK) jadi produksi tidak terkena, tetapi
    setiap pemanggil lain terkena: skrip evaluasi, panggilan API tanpa config,
    dan SETIAP pengukuran yang dijalankan dari Python - artinya angka yang
    diukur lewat jalur itu tidak menggambarkan yang dilihat pengguna.
    """
    import inspect

    sumber = inspect.getsource(svc.analyze)
    kotor = sumber.index('cleaner = TextCleaner.for_task')

    # Pembersihan tidak boleh berada di dalam cabang `if config_dict:`.
    sebelum = sumber[:kotor]
    assert 'if config_dict:' not in sebelum, (
        'pembersihan kembali dijadikan bersyarat pada preprocessing_config'
    )
    assert 'config_dict or {}' in sumber


# ── Pembersihan wajib, tidak bergantung pemanggil ──────────────────────────

def test_pembersihan_dijalankan_walau_tanpa_preprocessing_config(svc, monkeypatch):
    """Kebijakan 'bag_of_words' harus berlaku SELALU untuk jalur topik.

    Sebelumnya seluruh blok pembersihan dilewati bila `preprocessing_config`
    kosong, sehingga pemodelan topik berjalan di atas teks MENTAH - tanpa
    stopword, tanpa stemming, tanpa normalisasi slang. Laravel selalu mengirim
    config (resolvernya punya FALLBACK) jadi produksi tidak terkena, tetapi
    setiap pemanggil lain terkena: skrip evaluasi, panggilan API tanpa config,
    dan SETIAP pengukuran yang dijalankan dari Python - artinya angka yang
    diukur lewat jalur itu tidak menggambarkan yang dilihat pengguna.
    """
    import inspect

    sumber = inspect.getsource(svc.analyze)
    kotor = sumber.index('cleaner = TextCleaner.for_task')

    # Pembersihan tidak boleh berada di dalam cabang `if config_dict:`.
    sebelum = sumber[:kotor]
    assert 'if config_dict:' not in sebelum, (
        'pembersihan kembali dijadikan bersyarat pada preprocessing_config'
    )
    assert 'config_dict or {}' in sumber


# ── Kata kosong tidak boleh tampil sebagai label ───────────────────────────

