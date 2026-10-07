"""
TopicService: jalur yang tidak memerlukan bobot model.

BERTopic butuh model embedding, jadi tes di sini menyasar bagian yang bisa
dijalankan tanpa unduhan: LDA (murni scikit-learn), fallback frekuensi,
vectorizer, pemulihan bentuk kata, dan penarikan outlier dengan model tiruan.

Modul ini sebelumnya tidak punya tes sama sekali - termasuk jalur fallback
terakhir yang justru mengandung bug pembongkaran tuple.
"""

import pytest

from app.config import settings
from app.services.topic_service import TOPIC_KEYWORD_STOPWORDS, TopicService


@pytest.fixture(scope='module')
def service():
    """Konstruktor murah dan tidak memuat model; lihat _ensure_loaded."""
    return TopicService(use_bertopic=False)


# ── Fallback frekuensi ──────────────────────────────────────────────────────

def test_fallback_selalu_mengembalikan_dua_nilai(service):
    """
    Pemanggilnya membongkar `topics, sub_topics = ...`.

    Versi sebelumnya mengembalikan [] saat tidak ada kata tersisa, sehingga
    korpus yang habis setelah pembersihan memicu ValueError - tepat di jalur
    yang seharusnya menjadi jaring pengaman terakhir.
    """
    topics, document_topics = service._get_simple_topics(['', '   '])

    assert topics == []
    assert document_topics == [-1, -1]


def test_fallback_menyaring_kata_tanpa_makna(service):
    topics, _ = service._get_simple_topics(['tidak susah banget pajak pajak'])

    kata = {w for t in topics for w in t['words']}
    assert 'pajak' in kata
    assert not (kata & TOPIC_KEYWORD_STOPWORDS)


def test_fallback_menjaga_panjang_keluaran(service):
    _, document_topics = service._get_simple_topics(['pajak naik', 'jalan rusak'])

    assert len(document_topics) == 2


# ── Vectorizer ──────────────────────────────────────────────────────────────

def test_vectorizer_membuang_stopword_label(service):
    vectorizer = service._build_vectorizer(n_texts=10)
    vectorizer.fit(['pajak tidak susah', 'pajak naik terus'])

    kosakata = set(vectorizer.get_feature_names_out())
    assert 'pajak' in kosakata
    assert 'tidak' not in kosakata, 'negasi tidak informatif sebagai label topik'
    assert 'susah' not in kosakata


def test_vectorizer_memakai_hitungan_bukan_tfidf(service):
    """LDA adalah model generatif atas hitungan; TF-IDF melanggar asumsinya."""
    vectorizer = service._build_vectorizer(n_texts=10)
    matriks = vectorizer.fit_transform(['pajak pajak pajak', 'jalan rusak'])

    assert matriks[0, vectorizer.vocabulary_['pajak']] == 3


def test_vectorizer_melonggarkan_min_df_pada_korpus_kecil(service):
    """min_df=2 pada korpus kecil bisa menghabiskan seluruh kosakata."""
    vectorizer = service._build_vectorizer(n_texts=5)
    vectorizer.fit(['pajak naik', 'jalan rusak'])

    assert len(vectorizer.get_feature_names_out()) > 0


# ── LDA ─────────────────────────────────────────────────────────────────────

def test_lda_menghasilkan_topik_dan_penugasan_sejajar(service):
    texts = (
        ['pajak naik bayar negara'] * 6
        + ['jalan rusak jembatan akses'] * 6
    )

    topics, document_topics = service._extract_topics_lda(texts, num_topics=2)

    assert len(topics) == 2
    assert len(document_topics) == len(texts)
    assert all(t['method'] == 'lda' for t in topics)


def test_lda_kosakata_kosong_jatuh_ke_frekuensi(service):
    """Semua kata tersaring -> harus fallback, bukan melempar exception."""
    topics, document_topics = service._extract_topics_lda(
        ['tidak tidak', 'susah susah'], num_topics=2
    )

    assert len(document_topics) == 2
    assert all(t['method'] == 'frequency-based' for t in topics)


# ── Pemulihan bentuk kata ───────────────────────────────────────────────────

def test_bentuk_stem_diganti_bentuk_asli(service):
    topics = [{'topic_id': 0, 'words': ['jabat']}]
    asli = ['para pejabat korupsi', 'pejabat itu kaya', 'pejabat lagi']

    hasil = service._make_words_readable(topics, asli)

    assert hasil[0]['words'] == ['pejabat']


def test_bentuk_stem_tetap_disimpan_untuk_coherence(service):
    """Coherence harus dinilai pada kosakata korpus pemodelan yang ter-stem."""
    topics = [{'topic_id': 0, 'words': ['jabat']}]

    hasil = service._make_words_readable(topics, ['pejabat korupsi'])

    assert hasil[0]['words_stemmed'] == ['jabat']


def test_peluluhan_huruf_awal_tetap_tertangkap(service):
    """'pimpin' -> 'pemimpin' kehilangan p; uji substring penuh akan meleset."""
    topics = [{'topic_id': 0, 'words': ['pimpin']}]

    hasil = service._make_words_readable(topics, ['pemimpin negara', 'pemimpin kita'])

    assert hasil[0]['words'] == ['pemimpin']


def test_kata_tanpa_padanan_dibiarkan_apa_adanya(service):
    topics = [{'topic_id': 0, 'words': ['xyzabc']}]

    hasil = service._make_words_readable(topics, ['pajak naik'])

    assert hasil[0]['words'] == ['xyzabc']


def test_tanpa_teks_asli_topik_tidak_berubah(service):
    topics = [{'topic_id': 0, 'words': ['jabat']}]

    assert service._make_words_readable(topics, []) == topics


# ── Penarikan outlier ───────────────────────────────────────────────────────

class _ModelPalsu:
    def __init__(self, hasil=None, meledak=False):
        self._hasil = hasil
        self._meledak = meledak

    def reduce_outliers(self, texts, topics, strategy=None, threshold=None):
        if self._meledak:
            raise RuntimeError('representasi belum siap')
        return self._hasil


def test_outlier_ditarik_ke_topik_terdekat(service):
    hasil = service._reduce_outliers(_ModelPalsu([0, 0, 1]), ['a', 'b', 'c'], [0, -1, 1])

    assert hasil == [0, 0, 1]


def test_kegagalan_penarikan_tidak_menggagalkan_analisis(service):
    semula = [0, -1, 1]

    hasil = service._reduce_outliers(_ModelPalsu(meledak=True), ['a', 'b', 'c'], semula)

    assert hasil == semula


def test_panjang_berubah_ditolak(service):
    """Panjang yang berubah akan merusak penjajaran indeks ke pemanggil."""
    semula = [0, -1, 1]

    hasil = service._reduce_outliers(_ModelPalsu([0, 1]), ['a', 'b', 'c'], semula)

    assert hasil == semula


def test_tanpa_outlier_model_tidak_dipanggil(service):
    hasil = service._reduce_outliers(_ModelPalsu(meledak=True), ['a', 'b'], [0, 1])

    assert hasil == [0, 1]


# ── Metrik mutu ─────────────────────────────────────────────────────────────

def test_metrik_mutu_ikut_dikembalikan(service):
    topics = [
        {'topic_id': 0, 'words': ['pajak', 'bayar'], 'words_stemmed': ['pajak', 'bayar']},
        {'topic_id': 1, 'words': ['jalan', 'rusak'], 'words_stemmed': ['jalan', 'rusak']},
    ]

    metrik = service._quality_metrics(topics, ['pajak bayar', 'jalan rusak'], [0, 1])

    assert set(metrik) >= {'c_v', 'c_npmi', 'diversity', 'outlier_rate'}


def test_metrik_mutu_tanpa_topik_kosong(service):
    assert service._quality_metrics([], ['pajak'], [-1]) == {}


def test_max_df_tidak_membuang_kata_utama_pada_korpus_kecil(service):
    """
    max_df sebagai pecahan tidak stabil pada korpus kecil.

    Dengan 10 dokumen, ambang 0,85 membuang kata yang muncul di 9 dokumen -
    pada korpus bertema tunggal itu justru kata paling pentingnya.
    """
    vectorizer = service._build_vectorizer(n_texts=10)
    vectorizer.fit(['pajak tidak susah', 'pajak naik terus'])

    assert 'pajak' in set(vectorizer.get_feature_names_out())


def test_word_cloud_memakai_bentuk_kata_yang_sama_dengan_label(service):
    """
    Satu halaman hasil tidak boleh menampilkan dua bentuk untuk kata yang sama.

    Sebelumnya daftar topik berbunyi "pejabat" sementara word cloud di halaman
    yang sama berbunyi "jabat".
    """
    hasil = service._get_word_frequencies(
        ['jabat jabat korupsi'],
        original_texts=['pejabat korupsi', 'pejabat lagi'],
    )

    kata = {w['word'] for w in hasil}
    assert 'pejabat' in kata
    assert 'jabat' not in kata


def test_word_cloud_tanpa_teks_asli_tetap_jalan(service):
    hasil = service._get_word_frequencies(['pajak naik pajak'])

    assert hasil[0]['word'] == 'pajak'
    assert hasil[0]['frequency'] == 2


# ── Konsistensi size / proportion ───────────────────────────────────────────

def test_fallback_menugaskan_dokumen_bukan_semua_outlier(service):
    """
    Versi sebelumnya mengembalikan -1 untuk SELURUH dokumen, sehingga pada
    jalur fallback ini asosiasi PMI aspek-topik mati total.
    """
    texts = ['pajak bayar negara'] * 4 + ['jalan rusak jembatan'] * 4

    _, document_topics = service._get_simple_topics(texts)

    assert any(t != -1 for t in document_topics)
    assert len(document_topics) == len(texts)


def test_fallback_dokumen_tanpa_kata_bersama_tetap_outlier(service):
    """
    Bucket dibentuk dari 15 kata terpopuler saja.

    Korpus di bawah punya 15 kata yang sering, sehingga kata langka pada
    dokumen terakhir jatuh di luar bucket mana pun dan dokumennya harus tetap
    ditandai -1 - bukan dipaksa masuk topik.
    """
    sering = 'aaa bbb ccc ddd eee fff ggg hhh iii jjj kkk lll mmm nnn ooo'
    texts = [sering] * 5 + ['zzzlangka']

    _, document_topics = service._get_simple_topics(texts)

    assert document_topics[-1] == -1
    assert all(t != -1 for t in document_topics[:-1])


def test_fallback_size_adalah_jumlah_dokumen(service):
    """`size` ditampilkan sebagai "muncul pada N teks", bukan jumlah kata."""
    texts = ['pajak bayar negara rakyat uang'] * 5

    topics, document_topics = service._get_simple_topics(texts)

    total = sum(t['size'] for t in topics)
    assert total == sum(1 for t in document_topics if t != -1)
    assert total <= len(texts)


def test_lda_size_tidak_melebihi_jumlah_teks(service):
    """Ambang 0,1 yang lama membuat satu dokumen terhitung di banyak topik."""
    texts = ['pajak naik bayar negara'] * 6 + ['jalan rusak jembatan akses'] * 6

    topics, document_topics = service._extract_topics_lda(texts, num_topics=2)

    assert sum(t['size'] for t in topics) == len(texts)
    for topic in topics:
        assert topic['size'] == sum(1 for t in document_topics if t == topic['topic_id'])


# ── Pemilihan jumlah topik otomatis ─────────────────────────────────────────

class _ModelTopik:
    """
    Model BERTopic tiruan yang bisa dipangkas.

    Menirukan sifat yang penting: reduce_topics HANYA bisa mengurangi, dan
    mengubah keadaan model secara destruktif.
    """

    def __init__(self, peta):
        # peta: k -> (assigned, {topic_id: [kata, ...]})
        self._peta = peta
        self._k = max(peta)
        self.topics_ = list(peta[self._k][0])

    def get_topic(self, tid):
        kata = self._peta[self._k][1].get(tid, [])
        return [(w, 1.0) for w in kata]

    def reduce_topics(self, texts, nr_topics):
        if nr_topics not in self._peta:
            raise ValueError('k tidak tersedia')
        self._k = nr_topics
        self.topics_ = list(self._peta[nr_topics][0])


def _peta_dua_pilihan():
    """
    k=3 punya coherence lebih tinggi TAPI satu topik menelan 80% dokumen;
    k=2 seimbang. Ini pola nyata yang terukur pada korpus produksi.
    """
    seimbang = [0] * 5 + [1] * 5
    timpang = [0] * 8 + [1] + [2]
    return {
        3: (timpang, {0: ['pajak', 'bayar'], 1: ['jalan', 'rusak'], 2: ['zzz', 'qqq']}),
        2: (seimbang, {0: ['pajak', 'bayar'], 1: ['jalan', 'rusak']}),
    }


def test_topik_dominan_ditolak_meski_coherence_lebih_tinggi(service, monkeypatch):
    """
    Memaksimalkan C_v saja tidak aman.

    Terukur pada 885 komentar: C_v tertinggi jatuh di k=8, tetapi di sana satu
    topik menelan 58% dokumen - solusi degenerate yang tampak bagus di metrik.
    """
    monkeypatch.setattr(settings, 'topic_auto_min_k', 2)
    monkeypatch.setattr(settings, 'topic_auto_max_imbalance', 3.5)
    model = _ModelTopik(_peta_dua_pilihan())
    texts = ['pajak bayar'] * 5 + ['jalan rusak'] * 5

    k = service._select_num_topics(model, texts, model.topics_)

    assert k == 2, 'k=3 harus ditolak karena satu topik menelan 80%'


def test_klaster_alami_dipakai_bila_sudah_kecil(service, monkeypatch):
    monkeypatch.setattr(settings, 'topic_auto_min_k', 4)
    model = _ModelTopik(_peta_dua_pilihan())

    k = service._select_num_topics(model, ['a b'] * 10, [0, 1, 2] + [0] * 7)

    assert k == 3, 'tidak perlu mencari bila klaster alami sudah di batas bawah'


def test_batas_mutlak_menolak_topik_yang_menelan_korpus(service, monkeypatch):
    """
    Batas relatif saja tidak cukup pada k kecil.

    Terukur pada 200 dokumen: k=4 dengan satu topik 82,5% tetap lolos batas
    relatif (0,825 x 4 = 3,3 <= 3,5), padahal jelas degenerate.
    """
    monkeypatch.setattr(settings, 'topic_auto_min_k', 2)
    monkeypatch.setattr(settings, 'topic_auto_max_imbalance', 3.5)
    monkeypatch.setattr(settings, 'topic_auto_max_share', 0.50)

    seimbang = [0] * 5 + [1] * 5
    menelan = [0] * 8 + [1] + [2] + [3] * 0  # 8/10 = 80% pada k=3
    peta = {
        3: (menelan, {0: ['pajak', 'bayar'], 1: ['jalan', 'rusak'], 2: ['zzz', 'qqq']}),
        2: (seimbang, {0: ['pajak', 'bayar'], 1: ['jalan', 'rusak']}),
    }
    model = _ModelTopik(peta)

    k = service._select_num_topics(model, ['pajak bayar'] * 5 + ['jalan rusak'] * 5,
                                   model.topics_)

    assert k == 2, '80% pada k=3 harus ditolak batas mutlak'


# ── Deteksi klasterisasi degenerate ─────────────────────────────────────────

def test_klaster_terlalu_sedikit_terdeteksi(service, monkeypatch):
    monkeypatch.setattr(settings, 'topic_degenerate_min_clusters', 3)

    alasan = service._is_degenerate([0, 0, 0, 1, 1, 1])

    assert alasan and 'klaster' in alasan


def test_satu_klaster_menelan_korpus_terdeteksi(service, monkeypatch):
    """
    Terukur pada uji ketahanan: youtube n=60 menghasilkan satu topik 88%,
    ulasan n=250 satu topik 69%. Itu keranjang sisa, bukan topik.
    """
    monkeypatch.setattr(settings, 'topic_degenerate_min_clusters', 3)
    monkeypatch.setattr(settings, 'topic_degenerate_max_share', 0.50)
    assigned = [0] * 8 + [1, 2]

    alasan = service._is_degenerate(assigned)

    assert alasan and 'menelan' in alasan


def test_klasterisasi_seimbang_dianggap_wajar(service, monkeypatch):
    monkeypatch.setattr(settings, 'topic_degenerate_min_clusters', 3)
    monkeypatch.setattr(settings, 'topic_degenerate_max_share', 0.50)

    assert service._is_degenerate([0, 0, 1, 1, 2, 2]) is None


def test_outlier_tidak_dihitung_sebagai_klaster(service, monkeypatch):
    """-1 adalah penanda outlier, bukan topik."""
    monkeypatch.setattr(settings, 'topic_degenerate_min_clusters', 3)

    alasan = service._is_degenerate([-1, -1, 0, 0, 1, 1])

    assert alasan and '2 klaster' in alasan


# ── Vectorizer c-TF-IDF vs vectorizer dokumen ───────────────────────────────

def test_vectorizer_ctfidf_tidak_tabrakan_pada_topik_sedikit(service):
    """
    BERTopic menerapkan vectorizer pada dokumen GABUNGAN PER-TOPIK.

    Dengan 2 topik, `max_df=0,85` memberi 1 dokumen sementara `min_df=2`
    meminta 2: sklearn melempar `max_df corresponds to < documents than
    min_df`, seluruh jalur BERTopic gagal, dan hasilnya diam-diam jatuh ke
    fallback frekuensi kata yang paling kasar.
    """
    vectorizer = service._build_vectorizer(n_texts=500, for_ctfidf=True)

    # Dua "dokumen" - persis jumlah topik pada kasus yang gagal.
    vectorizer.fit(['pajak bayar negara', 'jalan rusak jembatan'])

    assert len(vectorizer.get_feature_names_out()) > 0


def test_vectorizer_dokumen_tetap_memakai_ambang(service):
    """Jalur LDA memakai dokumen sungguhan, jadi ambangnya tetap bermakna."""
    vectorizer = service._build_vectorizer(n_texts=500, for_ctfidf=False)

    assert vectorizer.min_df == 2
    assert vectorizer.max_df == 0.85


def test_lda_menerima_mode_otomatis(service):
    """
    num_topics=0 berarti "pilih otomatis", tetapi LDA harus diberi angka.

    Diteruskan mentah, sklearn menolak dengan `n_components must be an int in
    the range (0, inf)` dan hasilnya jatuh ke fallback frekuensi kata.
    """
    texts = ['pajak naik bayar negara'] * 8 + ['jalan rusak jembatan akses'] * 8

    topics, document_topics = service._extract_topics_lda(texts, num_topics=0)

    assert len(topics) >= 2
    assert all(t['method'] == 'lda' for t in topics), 'tidak boleh jatuh ke fallback'
    assert len(document_topics) == len(texts)
