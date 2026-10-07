import re
from typing import List, Dict, Any, Optional, Union
from collections import Counter

# BAHAYA URUTAN IMPOR - jangan diubah. Pada Windows, memuat nltk sebelum torch
# memicu `OSError WinError 1114` saat memuat c10.dll. text_cleaner menarik nltk,
# sedangkan bertopic menarik torch lewat sentence_transformers.
#
# Modul ini dulu mengimpor text_cleaner LEBIH DULU dan tetap berjalan hanya
# karena main.py kebetulan memuat sentiment_service serta aspect_service (yang
# menarik torch) sebelumnya. Ketergantungan tak tertulis itu pecah begitu
# topic_service diimpor sendirian - misalnya oleh berkas tes. Impor torch di
# sini membuat modul berdiri sendiri.
import torch  # noqa: F401

try:
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    BERTOPIC_AVAILABLE = True
except ImportError:
    BERTOPIC_AVAILABLE = False

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from app.config import hub_kwargs, settings
from app.preprocessing.text_cleaner import NEGATION_WORDS, TextCleaner
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

# Kata yang boleh tetap ada di TEKS tetapi tidak layak menjadi LABEL topik.
#
# Dua lapis stopword ini sengaja berbeda. TextCleaner mempertahankan kata negasi
# (lihat NEGATION_WORDS) karena membuangnya membalik polaritas bagi konsumen
# hilir. Tetapi sebagai kata kunci topik, "tidak" dan "susah" tidak memberi tahu
# apa pun tentang isi topik - terukur pada baseline, keduanya muncul di kata
# teratas T0 dan T1 sekaligus. Menyaringnya di lapis vectorizer hanya mengubah
# LABEL topik; penugasan dokumen ke topik tidak tersentuh.
TOPIC_KEYWORD_STOPWORDS = set(NEGATION_WORDS) | {
    # Penilaian umum: muncul di mana-mana, tidak membedakan topik apa pun.
    'susah', 'gampang', 'mudah', 'banget', 'sekali', 'sangat', 'lebih',
    'paling', 'makin', 'terus', 'aja', 'saja', 'juga', 'udah', 'sudah',
    'masih', 'lagi', 'harus', 'bisa', 'ada', 'punya', 'jadi', 'buat',
    # Interjeksi dan penanda percakapan.
    'hahaha', 'haha', 'wkwk', 'wkwkwk', 'ngapain', 'emang', 'memang',
    'kali', 'kayak', 'gitu', 'gini', 'begitu', 'begini', 'bilang', 'orang',
    'yg', 'nya',
    # Partikel yang lolos ke label topik pada pengukuran korpus produksi.
    # Semuanya tidak menerangkan isi topik apa pun, tetapi ikut terpilih
    # karena langka - dan MMR justru menyukai kata langka.
    'ha', 'hah', 'loh', 'lho', 'lha', 'noh', 'nih', 'tuh', 'yah', 'deh',
    'dong', 'kok', 'toh', 'bu', 'pak', 'mas', 'mbak', 'bang',
    # Sisa stemming yang bukan kata.
    'ny', 'lah', 'kah', 'pun',
}


class TopicService:
    """Topic Modeling Service with IndoBERT + BERTopic (lazy-loaded)"""

    def __init__(self, use_bertopic: bool = True):
        logger.info("Initializing Topic Service (lazy mode)...")
        self.use_bertopic = use_bertopic and BERTOPIC_AVAILABLE
        self.topic_model = None
        self.embedding_model = None
        self._loaded = False
        # Dipakai hanya untuk memulihkan bentuk kata yang terbaca; lihat
        # _make_words_readable.
        self._stemmer = None

        if not self.use_bertopic and BERTOPIC_AVAILABLE is False:
            logger.warning("BERTopic not installed. Will use LDA method.")

        logger.info("✅ Topic service ready (BERTopic model will load on first use)")

    def _ensure_loaded(self):
        """Lazy-load BERTopic + IndoBERT embedding model on first use."""
        if self._loaded:
            return
        if not self.use_bertopic:
            self._loaded = True
            return
        try:
            logger.info("Loading IndoBERT embedding model for BERTopic...")
            self.embedding_model = SentenceTransformer(
                'indobenchmark/indobert-base-p1',
                **hub_kwargs('indobenchmark/indobert-base-p1'),
            )
            logger.info("✅ IndoBERT embedding model loaded")
        except Exception as e:
            logger.warning(f"Failed to load IndoBERT: {str(e)}. Falling back to LDA.")
            self.use_bertopic = False
        self._loaded = True

    async def analyze(
        self,
        texts: List[str],
        preprocessing_config: Optional[Union[Dict, Any]] = None,
        num_topics: int = 5
    ) -> Dict[str, Any]:
        """Identify topics from texts"""
        # Lazy-load BERTopic/IndoBERT on first call
        self._ensure_loaded()

        try:
            # Convert preprocessing_config to dict if it's a Pydantic model
            config_dict = None
            if preprocessing_config:
                if hasattr(preprocessing_config, 'dict'):
                    config_dict = preprocessing_config.dict()
                elif isinstance(preprocessing_config, dict):
                    config_dict = preprocessing_config
            
            # Preprocessing. Profil 'bag_of_words' memastikan stemming dan
            # stopword removal benar-benar aktif untuk jalur ini - pembersihan
            # agresif memang menguntungkan model bag-of-words, kebalikan dari
            # jalur transformer pada modul sentimen.
            # Dijalankan SELALU, bukan hanya ketika pemanggil mengirim config.
            # Sebelumnya seluruh blok ini dilewati bila preprocessing_config
            # kosong, sehingga pemodelan topik berjalan di atas teks MENTAH -
            # tanpa stopword, tanpa stemming, tanpa normalisasi slang. Kebijakan
            # 'bag_of_words' yang seharusnya wajib untuk jalur ini justru tidak
            # pernah dipakai, dan labelnya dipenuhi kata fungsi serta ejaan
            # panjang (`tapi`, `wah`, `chuaaakksss`).
            #
            # Laravel selalu mengirim config (resolvernya punya FALLBACK), jadi
            # jalur produksi tidak terkena - tetapi setiap pemanggil lain
            # terkena: skrip evaluasi, /api/analyze/topic tanpa config, dan
            # pengukuran apa pun yang dilakukan dari Python. Artinya angka yang
            # diukur lewat jalur itu TIDAK menggambarkan yang dilihat pengguna.
            cleaner = TextCleaner.for_task('bag_of_words', config_dict or {})
            preprocessed_texts = cleaner.clean_texts(texts)

            # Token langka dibuang setelah pembersihan per-dokumen, karena
            # penyaringan ini butuh melihat korpus utuh. Bawaannya mati
            # (topic_min_token_freq = 1) karena terukur merugikan di bawah ~900
            # dokumen; penyaringan salah ketik dikerjakan di tingkat LABEL oleh
            # _label_vocabulary, yang tidak memindahkan dokumen antar topik.
            preprocessed_texts = TextCleaner.filter_rare_tokens(
                preprocessed_texts, settings.topic_min_token_freq
            )


            # Teks yang kosong setelah pembersihan tidak bisa dimodelkan, tetapi
            # posisinya HARUS dipertahankan: document_topics wajib sejajar
            # dengan daftar texts milik pemanggil. Membuangnya begitu saja
            # menggeser seluruh indeks setelahnya, dan AssociationService serta
            # Laravel memetakan hasil berdasarkan posisi.
            kept_indices = [i for i, t in enumerate(preprocessed_texts) if t.strip()]
            modeling_texts = [preprocessed_texts[i] for i in kept_indices]

            if len(modeling_texts) < 3:
                logger.warning("Not enough texts for topic modeling")
                topics, sub_topics = self._get_simple_topics(modeling_texts)
            else:
                # Choose topic extraction method
                if self.use_bertopic:
                    logger.info("Using BERTopic with IndoBERT for topic extraction")
                    topics, sub_topics = self._extract_topics_bertopic(modeling_texts, num_topics)
                else:
                    logger.info("Using LDA for topic extraction")
                    topics, sub_topics = self._extract_topics_lda(modeling_texts, num_topics)

            # Kembalikan ke posisi asli; -1 menandai dokumen tanpa topik, nilai
            # yang memang sudah dipakai untuk outlier HDBSCAN.
            document_topics = [-1] * len(texts)
            for position, original_index in enumerate(kept_indices):
                if position < len(sub_topics) and original_index < len(document_topics):
                    document_topics[original_index] = sub_topics[position]

            # Label dibuat terbaca SETELAH pemodelan dan setelah document_topics
            # dibentuk, sehingga tidak ada tahap hilir yang bergantung padanya.
            topics = self._make_words_readable(topics, texts)

            word_frequencies = self._get_word_frequencies(
                modeling_texts, original_texts=texts
            )

            # Metrik mutu ikut dikembalikan supaya hasil dapat dinilai tanpa
            # menjalankan skrip terpisah - dan supaya Laravel bisa menampilkan
            # dasar kuantitatif, bukan sekadar daftar kata.
            quality = self._quality_metrics(topics, modeling_texts, document_topics)

            return {
                'topics': topics,
                'word_frequencies': word_frequencies,
                'num_texts': len(texts),
                'num_topics': len(topics),
                'summary': self._generate_summary(topics, len(texts)),
                'document_topics': document_topics,
                'quality': quality,
            }
            
        except Exception as e:
            logger.error(f"Topic modeling error: {str(e)}")
            raise
    
    def _quality_metrics(
        self,
        topics: List[Dict],
        modeling_texts: List[str],
        document_topics: List[int],
    ) -> Dict[str, Any]:
        """
        Hitung coherence, diversity, dan outlier rate untuk hasil ini.

        Memakai `words_stemmed` bila ada: coherence harus dinilai pada kosakata
        yang SAMA dengan korpus rujukan, dan korpus rujukan di sini adalah teks
        yang sudah dibersihkan-dan-di-stem. Menilai bentuk terbaca terhadap
        korpus ter-stem akan membuat setiap kata tidak ditemukan dan seluruh
        skor jatuh ke nol tanpa sebab yang terlihat.

        Kegagalan tidak boleh menggagalkan analisis; metrik hanya pelengkap.

        Args:
            topics: Topik hasil ekstraksi.
            modeling_texts: Teks bersih yang benar-benar dimodelkan.
            document_topics: Id topik per dokumen masukan.

        Returns:
            Dict metrik, atau dict kosong bila gagal dihitung.
        """
        if not topics:
            return {}

        try:
            from app.utils.topic_metrics import evaluate

            topic_words = [
                t.get('words_stemmed') or t.get('words', []) for t in topics
            ]
            return evaluate(
                topic_words=topic_words,
                tokenized_docs=[t.split() for t in modeling_texts if t.strip()],
                document_topics=document_topics,
                top_n=settings.topic_top_n_words,
            )
        except Exception as exc:                        # noqa: BLE001
            logger.warning(f"Metrik mutu topik gagal dihitung: {exc}")
            return {}

    def _make_words_readable(
        self,
        topics: List[Dict],
        original_texts: List[str],
    ) -> List[Dict]:
        """
        Ganti kata topik hasil stemming dengan bentuk asli yang paling sering.

        Pemodelan dilakukan atas teks yang sudah di-stem - itu benar, karena
        menyatukan "bayar", "membayar", dan "dibayar" menjadi satu sinyal.
        Tetapi hasil stemming BUKAN kata yang bisa dibaca orang: label yang
        muncul di antarmuka berbunyi "jabat" (dari pejabat), "anggar" (dari
        anggaran), "laku", "mu". Pengguna tidak bisa menafsirkan itu.

        Bentuk stem tetap disimpan pada `words_stemmed` supaya perhitungan
        coherence - yang harus memakai kosakata yang sama dengan korpus
        pemodelan - tidak ikut berubah, dan angka di skripsi tetap bisa
        direproduksi.

        Args:
            topics: Topik dengan kunci `words` berisi bentuk hasil stemming.
            original_texts: Teks asli sebelum pembersihan.

        Returns:
            Daftar topik yang sama, dengan `words` dibuat terbaca dan
            `words_stemmed` ditambahkan.
        """
        wanted = {w for topic in topics for w in topic.get('words', [])}
        surfaces = self._surface_map(wanted, original_texts)
        if not surfaces:
            return topics

        for topic in topics:
            stemmed = list(topic.get('words', []))
            topic['words_stemmed'] = stemmed
            topic['words'] = [
                surfaces[w].most_common(1)[0][0] if surfaces.get(w) else w
                for w in stemmed
            ]

        return topics

    def _surface_map(
        self,
        wanted: set,
        original_texts: List[str],
    ) -> Dict[str, Counter]:
        """
        Petakan tiap stem ke bentuk permukaannya beserta frekuensinya.

        Dipakai bersama oleh label topik dan word cloud supaya keduanya
        menampilkan bentuk kata yang sama - sebelumnya label topik berbunyi
        "pejabat" sementara word cloud di halaman yang sama berbunyi "jabat".

        Args:
            wanted: Stem yang perlu dicarikan bentuk aslinya.
            original_texts: Teks asli sebelum pembersihan.

        Returns:
            Dict stem -> Counter bentuk permukaan; kosong bila tak bisa dihitung.
        """
        if not wanted or not original_texts:
            return {}

        if self._stemmer is None:
            try:
                from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
                self._stemmer = StemmerFactory().create_stemmer()
            except Exception as exc:                    # noqa: BLE001
                logger.warning(f"Stemmer tak tersedia, label dibiarkan: {exc}")
                return {}

        # Satu kali sapuan korpus: catat bentuk permukaan per stem, tetapi hanya
        # untuk stem yang benar-benar muncul sebagai kata topik.
        surfaces: Dict[str, Counter] = {w: Counter() for w in wanted}
        seen: Dict[str, str] = {}

        # Menstem SELURUH kosakata korpus memakan ~85 detik pada 885 komentar.
        # Imbuhan bahasa Indonesia bersifat konkatenatif, jadi akar kata hampir
        # selalu menjadi substring bentuk berimbuhannya ("pejabat" memuat
        # "jabat", "pemimpin" memuat "pimpin"). Menyaring dengan uji substring
        # yang murah lebih dulu membuat stemmer hanya dipanggil untuk kandidat
        # yang mungkin cocok. Kata dengan perubahan morfofonemis (misalnya
        # "menyapu" -> "sapu") lolos dari saringan ini dan tetap ditampilkan
        # dalam bentuk stem - itu penurunan mutu tampilan, bukan kesalahan.
        for text in original_texts:
            for token in re.findall(r'[a-zA-Z]+', text.lower()):
                if len(token) < 3:
                    continue
                stem = seen.get(token)
                if stem is None:
                    # w[1:] menangkap peluluhan huruf awal: "pimpin" ->
                    # "pemimpin" kehilangan p-nya, sehingga uji substring penuh
                    # meleset dan labelnya tetap tampil sebagai "pimpin".
                    # Berlaku juga untuk t->n, s->ny, k->ng.
                    if not any(w in token or w[1:] in token for w in wanted):
                        seen[token] = ''
                        continue
                    stem = self._stemmer.stem(token)
                    seen[token] = stem
                if stem and stem in surfaces:
                    surfaces[stem][token] += 1

        return surfaces

    def _label_vocabulary(self, docs: List[str]) -> Optional[List[str]]:
        """Kosakata yang boleh menjadi LABEL topik.

        Menjawab masalah yang terlihat langsung di korpus produksi: sebagian
        topik berlabel kata yang hanya muncul sekali - `chuaaakksss`,
        `membaaaanguuuuun`, `indonesiiiiaaaa`, `mahpuuudd`. Itu bukan topik,
        melainkan klaster salah ketik dan kata yang dipanjang-panjangkan.

        Penyebabnya struktural, bukan kebetulan: c-TF-IDF menghargai kata yang
        JARANG, sehingga hapax justru punya bobot tertinggi di topiknya. MMR
        memperkuatnya lagi karena kata langka selalu tampak "melengkapi" kata
        lain. Menambah stopword tidak menyelesaikannya - salah ketik tidak bisa
        didaftar satu per satu.

        Pembatasan ini hanya menyentuh PEMILIHAN KATA LABEL. Pengelompokan
        dokumen memakai embedding atas teks penuh dan tidak berubah sama sekali,
        jadi tidak ada dokumen yang berpindah topik karena filter ini. Itu
        bedanya dengan `filter_rare_tokens()`, yang membuang token dari dokumen
        SEBELUM pemodelan dan sudah diukur merugikan (c_v 0,4602 -> 0,3940 pada
        n=120); lihat catatan `topic_min_token_freq` di config.

        Dimatikan pada korpus kecil dan bila kosakata sisanya terlalu tipis,
        karena di sana ambang berapa pun akan menghabiskan kata yang justru
        paling penting.
        """
        ambang = settings.topic_label_min_freq
        if ambang <= 1 or len(docs) < settings.topic_label_min_docs:
            return None

        pola = re.compile(r'\b[a-z][a-z]+\b')
        frekuensi: Dict[str, int] = {}
        for dok in docs:
            for token in pola.findall(dok.lower()):
                frekuensi[token] = frekuensi.get(token, 0) + 1

        kosakata = sorted(
            kata for kata, n in frekuensi.items()
            if n >= ambang and kata not in TOPIC_KEYWORD_STOPWORDS
        )

        # Kalau penyaringan menyisakan terlalu sedikit kata, label justru jadi
        # lebih buruk daripada sebelum disaring - lebih baik tidak menyaring.
        if len(kosakata) < settings.topic_label_min_vocab:
            logger.info(
                f"Filter kosakata label dilewati: hanya {len(kosakata)} kata "
                f"lolos ambang {ambang}x"
            )
            return None

        logger.info(
            f"Kosakata label: {len(kosakata)} kata (>= {ambang}x) "
            f"dari {len(frekuensi)} kata unik"
        )
        return kosakata

    def _filter_label_words(
        self, pasangan: List[tuple], kosakata: Optional[set], batas: int
    ) -> List[tuple]:
        """Saring kata label: buang yang kosong, lalu yang terlalu langka.

        Dijalankan SETELAH model selesai, atas daftar kata yang dikembalikan
        BERTopic - jadi tidak ada penggabungan topik maupun penugasan dokumen
        yang terpengaruh. Itu bedanya dengan menyaring lewat `vocabulary=` pada
        vectorizer, yang terukur meruntuhkan 9 topik menjadi 2 pada korpus
        berita n=250.

        Kata kosong dibuang lebih dulu: BERTopic membantali daftarnya sampai
        top_n_words dengan string kosong ketika sebuah topik punya lebih sedikit
        istilah daripada itu.

        Kalau penyaringan frekuensi menghabiskan seluruh kata sebuah topik,
        hasil TANPA penyaringan dipakai - label langka masih lebih berguna
        daripada topik tanpa label sama sekali.
        """
        bersih = [(w, b) for w, b in pasangan if str(w).strip()]

        if not kosakata:
            return bersih[:batas]

        disaring = [(w, b) for w, b in bersih if w in kosakata]
        return (disaring or bersih)[:batas]

    def _build_vectorizer(
        self, n_texts: int, for_ctfidf: bool = False,
        docs: Optional[List[str]] = None
    ) -> CountVectorizer:
        """
        Vectorizer bersama untuk c-TF-IDF (BERTopic) maupun LDA.

        Memakai HITUNGAN kata, bukan TF-IDF. LDA adalah model generatif atas
        hitungan; menjalankannya di atas bobot TF-IDF - seperti versi sebelumnya
        - melanggar asumsi Dirichlet-multinomial dan tidak dianjurkan
        dokumentasi scikit-learn. Untuk BERTopic pun hitungan yang benar, karena
        pembobotan dilakukan sendiri oleh ClassTfidfTransformer setelahnya.

        Args:
            n_texts: Jumlah dokumen yang dimodelkan, untuk menyesuaikan min_df.

        Returns:
            CountVectorizer yang siap dipakai.
        """
        if for_ctfidf:
            # BERTopic menerapkan vectorizer pada dokumen GABUNGAN PER-TOPIK,
            # bukan per-dokumen. Jumlah "dokumen" yang dilihatnya sama dengan
            # jumlah topik - biasanya 2 sampai 20.
            #
            # Dengan min_df=2 dan max_df=0,85, korpus yang menghasilkan 2 topik
            # memberi max_df 0,85 x 2 = 1 dokumen, sementara min_df meminta 2:
            # sklearn melempar `max_df corresponds to < documents than min_df`,
            # seluruh jalur BERTopic gagal, dan hasilnya diam-diam jatuh ke
            # fallback frekuensi kata yang paling kasar. Terukur pada uji
            # ketahanan: 3 dari 14 kombinasi berakhir di jalur itu tanpa satu
            # pun tanda di keluaran API.
            #
            # Penyaringan kata terlalu umum di jalur ini memang bukan tugas
            # vectorizer, melainkan ClassTfidfTransformer(reduce_frequent_words).
            min_df, max_df = 1, 1.0
        else:
            # Jalur LDA memakai dokumen sungguhan, jadi ambangnya bermakna.
            # Pada korpus kecil, min_df > 1 bisa menghabiskan seluruh kosakata.
            min_df = 2 if n_texts >= 50 else 1

            # max_df sebagai pecahan tidak stabil pada korpus kecil: dengan 10
            # dokumen, ambang 0,85 membuang kata yang muncul di 9 dokumen - dan
            # pada korpus bertema tunggal itu justru kata paling pentingnya
            # ("pajak" hilang seluruhnya dari kosakata).
            max_df = 0.85 if n_texts >= 50 else 1.0

        # Kosakata TIDAK dibatasi di sini, dan itu keputusan yang diukur.
        #
        # Versi sebelumnya memasang `vocabulary=` hasil penyaringan frekuensi
        # pada vectorizer c-TF-IDF, dengan alasan "hanya menyentuh label".
        # Alasan itu SALAH: BERTopic memakai c-TF-IDF untuk menggabungkan topik
        # yang mirip dan menarik outlier, sehingga kosakata yang dipersempit
        # membuat topik tampak lebih mirip satu sama lain dan lebih banyak yang
        # dilebur. Terukur pada korpus berita n=250: 9 topik (terbesar 22,4%)
        # runtuh menjadi 2 topik (terbesar 59,6%), dan suite robustness turun
        # dari 14/14 ke 13/14.
        #
        # Penyaringan frekuensi tetap dilakukan, tetapi di tempat yang benar-
        # benar hanya label: saat kata diambil untuk ditampilkan
        # (`_filter_label_words`), setelah model selesai. Di sana ia tidak bisa
        # memengaruhi penggabungan topik maupun penugasan dokumen.
        return CountVectorizer(
            stop_words=sorted(TOPIC_KEYWORD_STOPWORDS),
            min_df=min_df,
            max_df=max_df,
            # Unigram saja. Bigram pada komentar pendek menghasilkan label yang
            # rusak dan berulang - terukur: "negara negara", "utang diskon" dan
            # "diskon utang" sekaligus di topik yang sama, dan c_v turun dari
            # 0,426 ke 0,281 dibanding baseline.
            ngram_range=(1, 1),
            token_pattern=r'\b[a-z][a-z]+\b',
        )

    def _postprocess_clustering(
        self,
        topic_model: Any,
        texts: List[str],
        topics_assigned: List[int],
        num_topics: int,
        embeddings: Any,
    ) -> tuple:
        """
        Selesaikan hasil klasterisasi: pilih k, pangkas topik, tarik outlier.

        Dijadikan satu method karena harus dijalankan IDENTIK pada jalur HDBSCAN
        maupun jalur pengganti KMeans. Kalau tidak, keduanya dibandingkan dalam
        keadaan yang berbeda dan keputusannya tidak sahih.

        Args:
            topic_model: Model BERTopic yang sudah di-fit.
            texts: Dokumen yang dimodelkan.
            topics_assigned: Penugasan awal hasil klasterisasi.
            num_topics: Jumlah topik diminta; <= 0 berarti pilih otomatis.
            embeddings: Embedding yang sudah dihitung, untuk pemasangan ulang.

        Returns:
            (topics_assigned, topic_info, num_topics_efektif)
        """
        unique_topics = set(topics_assigned)

        # num_topics = 0 berarti "pilihkan yang terbaik menurut coherence".
        if num_topics <= 0:
            num_topics = self._select_num_topics(
                topic_model, texts, topics_assigned
            )
            # Pencarian meninggalkan model pada k terendah yang dicoba,
            # sedangkan reduce_topics tidak bisa menaikkan kembali. Model
            # dipasang ulang dari embedding yang sama - tanpa encoding ulang -
            # lalu dipangkas sekali ke k terpilih di bawah.
            #
            # nr_topics WAJIB dikosongkan lebih dulu: reduce_topics menyimpan
            # nilainya di dalam model, dan fit_transform menerapkannya kembali
            # secara otomatis. Tanpa baris ini, pemasangan ulang langsung
            # terpangkas ke k terakhir yang dicoba pencarian (4), lalu
            # reduce_topics ke 18 tidak bisa menaikkannya - terukur: hasil
            # akhir 3 topik dengan satu topik menelan 78% dokumen, padahal
            # k=18 sudah terpilih benar.
            topic_model.nr_topics = None
            topics_assigned, _ = topic_model.fit_transform(
                texts, embeddings=embeddings
            )
            topics_assigned = list(topics_assigned)
            unique_topics = set(topics_assigned)

        topic_info = topic_model.get_topic_info()

        # Pangkas jumlah topik bila melebihi yang diminta.
        if len(unique_topics) - 1 > num_topics:  # -1 untuk topik outlier
            logger.info(f"Reducing {len(unique_topics)-1} topics to {num_topics}")
            topic_model.reduce_topics(texts, nr_topics=num_topics)
            topics_assigned = list(topic_model.topics_)
            topic_info = topic_model.get_topic_info()

        # ── Tarik dokumen outlier ke topik terdekat ─────────────────────────
        # HARUS setelah reduce_topics: reduce_topics menulis ulang
        # topic_model.topics_, sehingga penarikan outlier yang dilakukan lebih
        # dulu akan tertimpa dan hilang tanpa jejak.
        #
        # HDBSCAN menandai dokumen yang tak masuk klaster mana pun sebagai -1.
        # Terukur pada 885 komentar dengan parameter lama: 34% dokumen berakhir
        # -1, dan dokumen itu juga hilang dari perhitungan PMI aspek-topik -
        # jadi outlier bukan angka kosmetik, melainkan data yang terbuang.
        if settings.topic_reduce_outliers and -1 in set(topics_assigned):
            topics_assigned = self._reduce_outliers(
                topic_model, texts, topics_assigned
            )
            topic_info = topic_model.get_topic_info()

        return topics_assigned, topic_info, num_topics

    @staticmethod
    def _is_degenerate(assigned: List[int]) -> Optional[str]:
        """
        Kembalikan alasan bila hasil klasterisasi tidak layak dipakai.

        Dua tanda yang diperiksa, keduanya berarti "ini bukan pengelompokan
        topik yang berguna":

        1. Klaster terlalu sedikit - tidak ada yang bisa dibedakan.
        2. Satu klaster menelan sebagian besar korpus - klaster itu sebenarnya
           keranjang sisa, bukan topik.

        Args:
            assigned: Id topik per dokumen; -1 berarti outlier.

        Returns:
            Alasan sebagai teks, atau None bila hasilnya wajar.
        """
        ids = [t for t in set(assigned) if t != -1]
        total = len(assigned)
        if not total:
            return "tidak ada dokumen"

        # Lantai ikut tumbuh bersama korpus. Tiga klaster wajar untuk 50
        # dokumen dan jelas tidak wajar untuk 1 500 - tetapi batas mutlak tidak
        # bisa membedakan keduanya, dan itulah yang meloloskan korpus Kejaksaan
        # Agung (1 493 dokumen, 3 klaster, terbesar 40,4%) dari kedua penjaga.
        lantai = max(
            settings.topic_degenerate_min_clusters,
            min(
                settings.topic_degenerate_max_floor,
                total // settings.topic_degenerate_docs_per_cluster,
            ),
        )
        if len(ids) < lantai:
            return f"hanya {len(ids)} klaster untuk {total} dokumen (lantai {lantai})"

        sizes = [sum(1 for t in assigned if t == tid) for tid in ids]
        share = max(sizes) / total
        if share > settings.topic_degenerate_max_share:
            return f"satu klaster menelan {share:.0%} korpus"

        return None

    def _cluster_with_kmeans(
        self,
        texts: List[str],
        embeddings: Any,
        num_topics: int,
        umap_model: Any,
        representation_model: Any,
    ) -> Optional[tuple]:
        """
        Ulangi klasterisasi memakai KMeans sebagai jaring pengaman.

        Dipakai ketika HDBSCAN menghasilkan pengelompokan degenerate. Berbeda
        dari HDBSCAN, KMeans menerima jumlah klaster sebagai masukan dan
        menjamin k kelompok berukuran relatif seimbang - tepat yang dibutuhkan
        ketika kepadatan tidak membentuk klaster alami. Tidak ada outlier pada
        KMeans, sehingga seluruh dokumen mendapat topik.

        Args:
            texts: Dokumen yang dimodelkan.
            embeddings: Embedding yang sudah dihitung, agar tidak encode ulang.
            num_topics: Jumlah topik yang diminta; <= 0 berarti heuristik.
            umap_model: Model UMAP yang sama dengan jalur utama.
            representation_model: Model representasi yang sama.

        Returns:
            (model, assigned) atau None bila gagal.
        """
        try:
            from bertopic import BERTopic
            from bertopic.vectorizers import ClassTfidfTransformer
            from sklearn.cluster import KMeans
        except ImportError:
            return None

        n_texts = len(texts)
        if num_topics > 0:
            k = num_topics
        else:
            # Akar dari separuh jumlah dokumen: kaidah praktis yang lazim untuk
            # jumlah klaster, dibatasi agar tetap masuk akal untuk ditampilkan.
            k = int((n_texts / 2) ** 0.5)
        k = max(2, min(20, k, n_texts // 2))

        logger.info(f"Klasterisasi ulang dengan KMeans, n_clusters={k}")

        try:
            model = BERTopic(
                embedding_model=self.embedding_model,
                umap_model=umap_model,
                hdbscan_model=KMeans(
                    n_clusters=k, random_state=settings.topic_seed, n_init=10
                ),
                vectorizer_model=self._build_vectorizer(
                    n_texts, for_ctfidf=True, docs=texts
                ),
                ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=True),
                representation_model=representation_model,
                top_n_words=settings.topic_top_n_words,
                calculate_probabilities=False,
                verbose=False,
            )
            assigned, _ = model.fit_transform(texts, embeddings=embeddings)
            return model, list(assigned)
        except Exception as exc:                        # noqa: BLE001
            logger.warning(f"Jaring pengaman KMeans gagal: {exc}")
            return None

    def _select_num_topics(
        self,
        topic_model: Any,
        texts: List[str],
        topics_assigned: List[int],
    ) -> int:
        """
        Pilih jumlah topik yang memaksimalkan C_v, dengan kendala keseimbangan.

        Memaksimalkan coherence saja TIDAK aman. Terukur pada 885 komentar:
        C_v tertinggi jatuh di k=8 (0,5182), tetapi di sana satu topik menelan
        58% dokumen - "topik" itu sebenarnya sisa gabungan segala hal. Pilihan
        berimbang ada di k=18 (C_v 0,5123, topik terbesar 15%). Karena itu
        kandidat yang topik terbesarnya melewati settings.topic_auto_max_share
        dibuang lebih dulu, baru C_v tertinggi diambil dari sisanya.

        Pencarian berjalan MENURUN karena `reduce_topics` hanya bisa mengurangi;
        model dipulihkan ke jumlah klaster aslinya dengan cara dipangkas ulang
        oleh pemanggil sesudahnya.

        Args:
            topic_model: Model BERTopic yang sudah di-fit.
            texts: Dokumen yang dimodelkan.
            topics_assigned: Penugasan topik hasil klasterisasi.

        Returns:
            Jumlah topik terpilih; jumlah klaster alami bila pencarian gagal.
        """
        natural = len(set(topics_assigned) - {-1})
        if natural <= settings.topic_auto_min_k:
            return natural

        try:
            from app.utils.topic_metrics import cv_coherence
        except Exception:                               # noqa: BLE001
            return natural

        reference = [t.split() for t in texts if t.strip()]

        def evaluate_current(assigned: List[int]) -> Optional[tuple]:
            ids = sorted(t for t in set(assigned) if t != -1)
            words = [
                [w for w, _ in topic_model.get_topic(t)]
                for t in ids if topic_model.get_topic(t)
            ]
            if not words:
                return None
            score, _ = cv_coherence(words, reference,
                                    top_n=settings.topic_top_n_words)
            sizes = [sum(1 for x in assigned if x == t) for t in ids]
            share = max(sizes) / len(assigned) if sizes and assigned else 1.0
            # Ketimpangan diukur relatif terhadap bagian yang seimbang (1/k),
            # bukan sebagai ambang mutlak; lihat catatan di config.py.
            imbalance = share * len(ids) if ids else float('inf')
            return score, share, imbalance

        best_k, best_score = natural, -1.0
        def diterima(evaluated) -> bool:
            """Kedua batas harus dipenuhi; lihat catatan di config.py."""
            _, share, imbalance = evaluated
            return (share <= settings.topic_auto_max_share
                    and imbalance <= settings.topic_auto_max_imbalance)

        current = evaluate_current(list(topics_assigned))
        if current and diterima(current):
            best_score = current[0]

        for k in range(natural - 1, settings.topic_auto_min_k - 1, -1):
            try:
                topic_model.reduce_topics(texts, nr_topics=k)
            except Exception:                           # noqa: BLE001
                break
            evaluated = evaluate_current(list(topic_model.topics_))
            if not evaluated:
                continue
            if diterima(evaluated) and evaluated[0] > best_score:
                best_k, best_score = k, evaluated[0]

        logger.info(
            f"Jumlah topik dipilih otomatis: k={best_k} (C_v {best_score:.4f}) "
            f"dari {natural} klaster alami"
        )
        return best_k

    def _reduce_outliers(
        self,
        topic_model: Any,
        texts: List[str],
        topics_assigned: List[int],
    ) -> List[int]:
        """
        Tarik dokumen outlier ke topik terdekat memakai kemiripan c-TF-IDF.

        Hanya dokumen yang kemiripannya melewati ambang yang dipindahkan;
        sisanya tetap -1. Memaksa SELURUH dokumen masuk topik akan mengotori
        topik dengan dokumen yang memang tidak berkaitan.

        Kegagalan di sini tidak boleh menggagalkan analisis: bila BERTopic
        menolak (misalnya representasi belum siap), penugasan semula
        dikembalikan apa adanya.

        Args:
            topic_model: Model BERTopic yang sudah di-fit.
            texts: Dokumen yang dimodelkan, sejajar dengan topics_assigned.
            topics_assigned: Id topik hasil klasterisasi.

        Returns:
            Daftar id topik dengan panjang dan urutan yang sama.
        """
        before = sum(1 for t in topics_assigned if t == -1)
        if not before:
            return topics_assigned

        try:
            reduced = topic_model.reduce_outliers(
                texts,
                list(topics_assigned),
                strategy='c-tf-idf',
                threshold=settings.topic_outlier_threshold,
            )
        except Exception as exc:                       # noqa: BLE001
            logger.warning(f"Penarikan outlier dilewati: {exc}")
            return topics_assigned

        if len(reduced) != len(topics_assigned):
            logger.warning(
                "Penarikan outlier mengubah jumlah dokumen "
                f"({len(reduced)} != {len(topics_assigned)}); hasil lama dipakai"
            )
            return topics_assigned

        after = sum(1 for t in reduced if t == -1)
        logger.info(
            f"Outlier {before} -> {after} dari {len(texts)} dokumen "
            f"({before / len(texts):.1%} -> {after / len(texts):.1%})"
        )
        return list(reduced)

    def _extract_topics_bertopic(self, texts: List[str], num_topics: int) -> tuple:
        """Extract topics using BERTopic with IndoBERT embeddings"""
        try:
            from bertopic import BERTopic
            from sklearn.cluster import KMeans
            from bertopic.vectorizers import ClassTfidfTransformer
            from umap import UMAP
            
            n_texts = len(texts)
            logger.info(f"Starting BERTopic extraction for {n_texts} texts")
            
            # Parameter di bawah DIUKUR, bukan ditebak. Rumus lama
            # (min_cluster_size = n//20, min_samples = n//50, n_neighbors =
            # n//20) memberi 44/17/44 pada 885 komentar - terlalu kasar:
            # hanya 3 topik terbentuk, satu menyerap 53% dokumen, dan 34%
            # dokumen tak masuk topik mana pun.
            #
            # Sapuan (scripts/sweep_topic_params.py) pada korpus yang sama:
            #   44/17/44 (asli) : 3 topik,  c_v 0,390, terbesar 53%
            #   20/ 3/15        : 12 topik, c_v 0,474, terbesar 14%
            #   15/ 2/15 (kini) : 16 topik, c_v 0,518, terbesar 14%
            #
            # Parameter DISETEL ULANG setelah normalisasi slang ditambahkan.
            # Normalisasi menggabungkan varian kata sehingga dokumen menjadi
            # lebih mirip satu sama lain dan ruang embedding memadat; nilai yang
            # optimal pada ruang lama (20/3/15) menghasilkan klaster terlalu
            # sedikit pada ruang baru - terukur: 13 topik turun ke 9, satu di
            # antaranya kembali menelan 52% dokumen. Menyetel parameter
            # klasterisasi tanpa mengulang setelah mengubah praperolehan teks
            # adalah kesalahan yang mudah terlewat.
            n_neighbors = max(10, min(30, n_texts // 59))
            n_components = 5

            umap_model = UMAP(
                n_neighbors=n_neighbors,
                n_components=n_components,
                min_dist=0.0,
                metric='cosine',
                random_state=settings.topic_seed,
                low_memory=False  # Better for large datasets
            )
            
            logger.info(f"UMAP config: n_neighbors={n_neighbors}, n_components={n_components}")
            
            # ✅ FIX 2: HDBSCAN parameters optimized
            hdbscan_model = None
            if n_texts >= 10:
                try:
                    from hdbscan import HDBSCAN
                    
                    # Lihat catatan pengukuran pada blok UMAP di atas.
                    # min_samples 2: pada sapuan, nilai rendah secara konsisten
                    # menaikkan coherence maupun menurunkan outlier, sedangkan
                    # 17 (nilai asli) mendorong outlier melewati 50%.
                    min_cluster_size = max(5, min(25, n_texts // 59))
                    min_samples = 2

                    hdbscan_model = HDBSCAN(
                        min_cluster_size=min_cluster_size,
                        min_samples=min_samples,
                        metric='euclidean',
                        cluster_selection_method='eom',
                        prediction_data=True,
                        cluster_selection_epsilon=0.0  # Allow all clusters
                    )
                    logger.info(f"HDBSCAN config: min_cluster_size={min_cluster_size}, min_samples={min_samples}")
                except ImportError:
                    logger.warning("HDBSCAN not available, using KMeans")
                    hdbscan_model = None
            
            if hdbscan_model is None:
                # Fallback to KMeans.
                #
                # `n_texts // 50` memberi 2 klaster untuk korpus di bawah 100
                # dokumen dan mengabaikan num_topics yang diminta pengguna
                # sepenuhnya sampai korpusnya sangat besar. Karena KMeans - tidak
                # seperti HDBSCAN - memang menerima jumlah klaster sebagai
                # masukan, permintaan pengguna dipakai langsung dan hanya dibatasi
                # agar tidak melebihi jumlah dokumen.
                n_clusters = max(2, min(num_topics, n_texts // 2))
                hdbscan_model = KMeans(
                    n_clusters=n_clusters,
                    random_state=settings.topic_seed,
                    n_init=10
                )
                logger.info(f"KMeans config: n_clusters={n_clusters}")
            
            # ✅ FIX 3: Remove nr_topics parameter - let clustering work naturally
            # vectorizer_model dulu None, sehingga BERTopic memakai
            # CountVectorizer bawaan tanpa stopword Indonesia sama sekali.
            # Akibatnya kata teratas topik dipenuhi "tidak", "susah", "hahaha",
            # "ngapain" - terukur pada baseline. `language` sengaja tidak
            # dikirim: parameter itu hanya dipakai bila BERTopic membuat model
            # embedding sendiri, sedangkan di sini embedding_model diberikan.
            # MaximalMarginalRelevance memilih kata yang saling melengkapi alih
            # alih sinonim yang berulang; lihat angka pengukurannya di
            # settings.topic_mmr_diversity. Dipasang hanya bila BERTopic versi
            # ini menyediakannya, supaya bukan menjadi dependensi keras.
            representation_model = None
            if settings.topic_mmr_diversity > 0:
                try:
                    from bertopic.representation import MaximalMarginalRelevance
                    representation_model = MaximalMarginalRelevance(
                        diversity=settings.topic_mmr_diversity
                    )
                except ImportError:
                    logger.warning("MaximalMarginalRelevance tidak tersedia")

            topic_model = BERTopic(
                embedding_model=self.embedding_model,
                umap_model=umap_model,
                hdbscan_model=hdbscan_model,
                vectorizer_model=self._build_vectorizer(
                    n_texts, for_ctfidf=True, docs=texts
                ),
                ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=True),
                representation_model=representation_model,
                top_n_words=settings.topic_top_n_words,
                calculate_probabilities=False,
                verbose=False,
            )
            
            # Embedding dihitung SEKALI di sini, bukan diserahkan ke BERTopic.
            # Dengan begitu model bisa dipasang ulang tanpa membayar encoding
            # lagi - yang dibutuhkan oleh pemilihan jumlah topik otomatis,
            # karena `reduce_topics` bersifat destruktif dan hanya bisa
            # mengurangi, sehingga setelah pencarian model tertinggal di k
            # terendah dan harus dikembalikan ke k terpilih.
            logger.info(f"Fitting BERTopic model on {n_texts} texts...")
            embeddings = None
            if self.embedding_model is not None:
                try:
                    embeddings = self.embedding_model.encode(
                        texts, show_progress_bar=False
                    )
                except Exception as exc:                # noqa: BLE001
                    logger.warning(f"Encoding terpisah gagal: {exc}")

            topics_assigned, probs = topic_model.fit_transform(
                texts, embeddings=embeddings
            )
            
            # ✅ FIX 4: Check for valid topics
            unique_topics = set(topics_assigned)
            logger.info(f"Unique topics found: {unique_topics}")
            
            # Check if all texts are outliers
            if unique_topics == {-1} or len(unique_topics) <= 1:
                logger.warning("All texts classified as outliers or single topic. Falling back to LDA.")
                return self._extract_topics_lda(texts, num_topics)

            # Metode klasterisasi dilaporkan apa adanya pada tiap topik, supaya
            # terlihat kapan jaring pengaman ikut bekerja.
            clustering_method = 'bertopic-indobert'
            num_topics_diminta = num_topics

            topics_assigned, topic_info, num_topics = self._postprocess_clustering(
                topic_model, texts, list(topics_assigned), num_topics, embeddings
            )

            # ── Jaring pengaman: hasil akhir yang degenerate ────────────────
            # HDBSCAN berbasis kepadatan; pada korpus kecil atau ruang embedding
            # yang tidak terpisah rapi ia menghasilkan sedikit klaster dengan
            # satu klaster menelan sebagian besar korpus. Terukur pada uji
            # ketahanan 3 domain x 5 ukuran (scripts/robustness_topics.py):
            # 5 dari 14 kombinasi degenerate. KMeans tidak mengandalkan
            # kepadatan dan menjamin k kelompok yang seimbang.
            #
            # Pemeriksaan dilakukan pada hasil AKHIR, bukan pada klaster mentah.
            # Ketimpangan sebagian besar baru muncul setelah pemangkasan topik
            # dan penarikan outlier: outlier diserap ke topik terdekat, dan
            # topik terbesarlah yang paling banyak menyerap. Memeriksa klaster
            # mentah membuat tiga kasus lolos padahal hasil akhirnya timpang
            # (berita n=120 64%, ulasan n=250 69%).
            alasan = self._is_degenerate(topics_assigned)
            if alasan:
                logger.warning(f"Hasil klasterisasi degenerate ({alasan}); mencoba KMeans")
                pengganti = self._cluster_with_kmeans(
                    texts, embeddings, num_topics_diminta,
                    umap_model, representation_model
                )
                if pengganti:
                    kandidat_model, kandidat_assigned = pengganti
                    kandidat_assigned, kandidat_info, _ = self._postprocess_clustering(
                        kandidat_model, texts, kandidat_assigned,
                        num_topics_diminta, embeddings
                    )
                    # Hanya dipakai bila benar-benar lebih baik; kalau KMeans
                    # pun degenerate, hasil HDBSCAN dipertahankan agar tidak
                    # menukar satu masalah dengan masalah lain.
                    if not self._is_degenerate(kandidat_assigned):
                        topic_model = kandidat_model
                        topics_assigned = kandidat_assigned
                        topic_info = kandidat_info
                        clustering_method = 'bertopic-kmeans'
                        logger.info(
                            f"KMeans dipakai: "
                            f"{len(set(topics_assigned) - {-1})} topik"
                        )
                    else:
                        logger.warning("KMeans juga degenerate; hasil HDBSCAN dipakai")

            # Kosakata label dihitung SEKALI dari dokumen yang dimodelkan.
            # Dipakai hanya untuk menyaring kata yang ditampilkan; model sudah
            # selesai pada titik ini sehingga tidak ada topik yang bergeser.
            daftar = self._label_vocabulary(texts)
            kosakata_label = set(daftar) if daftar else None

            # Convert to expected format
            topics = []
            for idx, row in topic_info.iterrows():
                topic_id = row['Topic']
                
                # Skip outlier topic (-1)
                if topic_id == -1:
                    continue
                
                # Get top words for this topic
                topic_words = topic_model.get_topic(topic_id)
                if not topic_words:
                    continue
                
                # Kata kosong dan kata terlalu langka disaring DI SINI, atas
                # keluaran model - bukan lewat vectorizer. Lihat
                # _filter_label_words untuk alasan terukurnya.
                pasangan = self._filter_label_words(
                    list(topic_words), kosakata_label, settings.topic_top_n_words
                )
                words = [w for w, _ in pasangan]
                weights = [float(bobot) for _, bobot in pasangan]
                
                # Calculate proportion
                topic_count = np.sum(np.array(topics_assigned) == topic_id)
                proportion = topic_count / n_texts
                
                topics.append({
                    'topic_id': int(topic_id),
                    'words': words,
                    'weights': [round(w, 4) for w in weights],
                    'proportion': round(proportion, 4),
                    'size': int(topic_count),
                    'method': clustering_method
                })
            
            # ✅ FIX 6: Fallback if no valid topics
            if not topics:
                logger.warning("No valid topics extracted with BERTopic. Falling back to LDA.")
                return self._extract_topics_lda(texts, num_topics)
            
            # Sort by proportion
            topics.sort(key=lambda x: x['proportion'], reverse=True)
            
            # Reassign topic_id based on sorted order
            old_to_new_id = {}
            for new_id, topic in enumerate(topics):
                old_to_new_id[topic['topic_id']] = new_id
                topic['topic_id'] = new_id
            
            document_topics = [old_to_new_id.get(t, -1) for t in topics_assigned]
            
            logger.info(f"✅ BERTopic extracted {len(topics)} topics successfully")
            
            # Log topic distribution
            for topic in topics[:3]:  # Log top 3 topics
                logger.info(f"  Topic {topic['topic_id']}: {', '.join(topic['words'][:5])} ({topic['proportion']*100:.1f}%)")
            
            return topics, document_topics
            
        except Exception as e:
            logger.error(f"BERTopic extraction error: {str(e)}", exc_info=True)
            logger.info("Falling back to LDA method")
            return self._extract_topics_lda(texts, num_topics)
    
    def _extract_topics_lda(self, texts: List[str], num_topics: int) -> tuple:
        """Extract topics using Latent Dirichlet Allocation (fallback method)"""
        try:
            logger.info(f"Starting LDA extraction for {len(texts)} texts")
            
            # Hitungan kata, BUKAN TF-IDF: lihat _build_vectorizer.
            vectorizer = self._build_vectorizer(len(texts))
            count_matrix = vectorizer.fit_transform(texts)
            logger.info(f"Count matrix shape: {count_matrix.shape}")

            if count_matrix.shape[1] == 0:
                logger.warning("Kosakata kosong setelah penyaringan; memakai frekuensi kata")
                return self._get_simple_topics(texts)

            # num_topics <= 0 berarti "pilih otomatis", tetapi LDA harus diberi
            # angka. Diteruskan mentah, min(...) menghasilkan 0 dan sklearn
            # menolak dengan `n_components must be an int in the range (0, inf)`
            # - jalur LDA ikut gagal, lalu hasilnya jatuh ke fallback frekuensi
            # kata yang paling kasar tanpa satu pun tanda di keluaran API.
            if num_topics <= 0:
                # Akar dari separuh jumlah dokumen: kaidah praktis yang lazim
                # untuk jumlah klaster.
                num_topics = max(2, min(20, int((len(texts) / 2) ** 0.5)))
                logger.info(f"LDA memilih {num_topics} topik secara otomatis")

            # ✅ Adaptive num_topics
            actual_num_topics = max(
                2, min(num_topics, len(texts) - 1, count_matrix.shape[1])
            )

            lda_model = LatentDirichletAllocation(
                n_components=actual_num_topics,
                random_state=settings.topic_seed,
                max_iter=20,  # Increase iterations
                learning_method='batch',
                # n_jobs=1, BUKAN -1.
                #
                # `n_jobs=-1` menyuruh joblib memunculkan satu proses worker per
                # inti CPU, dan tiap worker mewarisi memori proses induk yang
                # sudah memuat bobot torch. Pada kontainer dengan memori ketat -
                # tepatnya lingkungan yang menjadi sasaran deploy - worker itu
                # dibunuh OS, dan LDA jatuh diam-diam ke fallback frekuensi kata
                # yang kasar. Terpicu nyata saat suite tes dijalankan utuh:
                # "A worker process managed by the executor was unexpectedly
                # terminated", padahal tes yang sama lolos bila dijalankan sendiri.
                #
                # Paralelismenya pun nyaris tidak berguna di sini: LDA adalah
                # jalur CADANGAN yang baru dipakai ketika BERTopic gagal, dan
                # korpusnya kecil (belasan sampai ratusan dokumen). Menukar
                # kecepatan yang tidak terasa dengan kegagalan senyap adalah
                # pertukaran yang salah.
                #
                # Alasan yang sama membuat service dijalankan dengan --workers 1.
                n_jobs=1
            )

            logger.info(f"Fitting LDA with {actual_num_topics} topics...")
            lda_model.fit(count_matrix)
            feature_names = vectorizer.get_feature_names_out()
            topic_distribution = lda_model.transform(count_matrix)
            doc_topics_raw = np.argmax(topic_distribution, axis=1)
            
            topics = []
            for topic_idx, topic in enumerate(lda_model.components_):
                top_word_indices = topic.argsort()[-10:][::-1]
                top_words = [feature_names[i] for i in top_word_indices]
                top_weights = [float(topic[i]) for i in top_word_indices]
                
                # size dan proportion dihitung dari penugasan KERAS (argmax),
                # sama seperti jalur BERTopic, karena keduanya ditampilkan
                # sebagai "muncul pada N teks (X% dari total)".
                #
                # Versi sebelumnya memakai `topic_distribution[:, k] > 0.1`:
                # ambang sembarang yang membuat satu dokumen dihitung pada
                # beberapa topik sekaligus, sehingga jumlah seluruh `size` bisa
                # melebihi jumlah teks - dan angka persentase di antarmuka tidak
                # sesuai dengan document_topics yang dipakai PMI.
                topic_size = int(np.sum(doc_topics_raw == topic_idx))

                topics.append({
                    'topic_id': topic_idx,
                    'words': top_words[:10],
                    'weights': [round(w, 4) for w in top_weights[:10]],
                    'proportion': round(topic_size / len(texts), 4) if texts else 0.0,
                    'size': topic_size,
                    'method': 'lda'
                })
            
            topics.sort(key=lambda x: x['proportion'], reverse=True)
            
            old_to_new_id = {}
            for new_id, topic in enumerate(topics):
                old_to_new_id[topic['topic_id']] = new_id
                topic['topic_id'] = new_id
                
            document_topics = [old_to_new_id.get(int(t), -1) for t in doc_topics_raw]
            
            logger.info(f"✅ LDA extracted {len(topics)} topics")
            
            return topics, document_topics
            
        except Exception as e:
            logger.error(f"LDA topic extraction error: {str(e)}", exc_info=True)
            return self._get_simple_topics(texts)
    
    def _get_simple_topics(self, texts: List[str]) -> tuple:
        """Get simple topics based on word frequency (fallback)"""
        logger.info("Using simple frequency-based topic extraction")
        
        all_words = []
        for text in texts:
            all_words.extend(
                w for w in text.split() if w not in TOPIC_KEYWORD_STOPWORDS
            )

        # Pemanggil membongkar dua nilai (topics, document_topics). Versi
        # sebelumnya mengembalikan [] di sini, sehingga korpus yang seluruhnya
        # kosong setelah pembersihan memicu ValueError alih-alih fallback -
        # tepat pada jalur yang seharusnya menjadi jaring pengaman terakhir.
        if not all_words:
            return [], [-1] * len(texts)

        word_counts = Counter(all_words)
        most_common = word_counts.most_common(20)
        
        topics = []
        words_per_topic = 5
        buckets: List[set] = []

        for i in range(0, min(len(most_common), 15), words_per_topic):
            topic_words = [word for word, count in most_common[i:i + words_per_topic]]
            topic_weights = [count for word, count in most_common[i:i + words_per_topic]]

            if topic_words:
                buckets.append(set(topic_words))
                topics.append({
                    'topic_id': len(topics),
                    'words': topic_words,
                    'weights': topic_weights,
                    'proportion': 0.0,   # diisi setelah dokumen ditugaskan
                    'size': 0,
                    'method': 'frequency-based'
                })

        # Dokumen ditugaskan ke bucket yang paling banyak berbagi kata dengannya.
        #
        # Versi sebelumnya mengembalikan -1 untuk SELURUH dokumen, sehingga pada
        # jalur fallback ini asosiasi PMI aspek-topik mati total: tidak ada satu
        # pun pasangan dokumen-topik yang bisa dihitung. Dokumen yang tidak
        # berbagi kata apa pun tetap -1, sebagaimana mestinya.
        document_topics = []
        for text in texts:
            tokens = {
                w for w in text.split() if w not in TOPIC_KEYWORD_STOPWORDS
            }
            overlaps = [len(tokens & bucket) for bucket in buckets]
            best = max(range(len(overlaps)), key=overlaps.__getitem__) if overlaps else -1
            document_topics.append(best if overlaps and overlaps[best] > 0 else -1)

        # proportion dan size mengikuti penugasan yang sama dengan jalur lain,
        # karena keduanya ditampilkan sebagai "muncul pada N teks (X% dari
        # total)". Sebelumnya `size` berisi JUMLAH KATA pada topik dan
        # `proportion` adalah pangsa frekuensi kata - bukan pangsa dokumen.
        for topic in topics:
            topic['size'] = sum(1 for t in document_topics if t == topic['topic_id'])
            topic['proportion'] = (
                round(topic['size'] / len(texts), 4) if texts else 0.0
            )

        return topics, document_topics
    
    def _get_word_frequencies(
        self,
        texts: List[str],
        top_n: int = 50,
        original_texts: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Frekuensi kata untuk word cloud.

        Memakai penyaring yang sama dengan kata kunci topik, supaya word cloud
        dan label topik konsisten. Sebelumnya word cloud menampilkan "di", "yg",
        "dan", "tidak", "ada" di posisi teratas - dan menghitung "rakyat" serta
        "Rakyat" sebagai dua kata berbeda karena tidak ada penyeragaman huruf.

        Bentuk kata juga dipulihkan seperti label topik. Tanpa itu satu halaman
        hasil menampilkan "pejabat" pada daftar topik dan "jabat" pada word
        cloud - dua bentuk untuk kata yang sama.

        Args:
            texts: Teks yang sudah dibersihkan dan di-stem.
            top_n: Jumlah kata yang dikembalikan.
            original_texts: Teks asli, untuk memulihkan bentuk yang terbaca.
        """
        all_words = []
        for text in texts:
            all_words.extend(
                w for w in text.lower().split()
                if w not in TOPIC_KEYWORD_STOPWORDS and len(w) > 2
            )

        word_counts = Counter(all_words)

        if original_texts:
            wanted = {w for w, _ in word_counts.most_common(top_n)}
            surfaces = self._surface_map(wanted, original_texts)
            if surfaces:
                readable = Counter()
                for word, count in word_counts.most_common(top_n):
                    display = (
                        surfaces[word].most_common(1)[0][0]
                        if surfaces.get(word) else word
                    )
                    readable[display] += count
                word_counts = readable
        most_common = word_counts.most_common(top_n)
        
        return [
            {'word': word, 'frequency': count}
            for word, count in most_common
        ]
    
    def _generate_summary(self, topics: List[Dict], num_texts: int) -> str:
        """Generate topic modeling summary"""
        if not topics:
            return "Tidak dapat mengidentifikasi topik dari teks yang diberikan."
        
        main_topic = topics[0]
        top_words = ', '.join(main_topic['words'][:5])
        
        # Add method info if available
        method_info = ""
        if 'method' in main_topic:
            method_map = {
                'bertopic-indobert': '(menggunakan IndoBERT)',
                'lda': '(menggunakan LDA)',
                'frequency-based': '(berbasis frekuensi)'
            }
            method_info = f" {method_map.get(main_topic['method'], '')}"
        
        return (
            f"Dari {num_texts} teks, teridentifikasi {len(topics)} topik utama{method_info}. "
            f"Topik dominan terkait dengan: {top_words}. "
            f"Topik ini muncul pada {main_topic['size']} teks "
            f"({round(main_topic['proportion'] * 100, 1)}% dari total)."
        )