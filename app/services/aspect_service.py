import asyncio
import os
import re
import torch
from typing import List, Dict, Any, Optional, Tuple, Union
from collections import defaultdict
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification
from app.config import hub_kwargs, settings
from app.preprocessing.text_cleaner import ROOT_WORDS
from app.services.sentiment_service import SentimentService
from app.utils.logger import setup_logger
from app.utils.training import (
    compute_class_weights,
    lepas_memori,
    pool_fingerprint,
    set_seed,
    should_accept_checkpoint,
    stratified_split,
    token_f1,
    write_provenance,
)
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

logger = setup_logger(__name__)


def jalur_bobot_dasar_aspek() -> str:
    """
    Lokasi bobot dasar protokol kolam untuk modul aspek.

    Setelan eksplisit menang. Bila kosong, jatuh ke `<model_path>/aspect_baseline.pt`
    - tempat yang memang dipakai semua jalur deploy (volume Railway, repo HF yang
    ditarik `weights_store`, dan salinan manual di VPS).

    Bawaannya dulu string kosong, dan itu jebakan: `retrain_from_baseline` tetap
    menyala, bobot dasar tidak ditemukan, satu `WARNING` tercatat, lalu pelatihan
    MELANJUTKAN checkpoint. Kesalahan yang persis sama pernah membatalkan satu
    iterasi active learning. Panduan deploy VPS pertama pun lupa menyetelnya -
    jadi yang diperbaiki bawaannya, bukan hanya dokumennya.
    """
    return settings.aspect_baseline_checkpoint or os.path.join(
        settings.model_path, "aspect_baseline.pt"
    )
factory = StopWordRemoverFactory()


class AspectService:
    """
    Aspect Extraction Service menggunakan IndoBERT + NER
    Simplified architecture tanpa BiLSTM dan CRF
    Backward compatible dengan service lama
    """
    
    def __init__(self, model_path: Optional[str] = None,
                 sentiment_service: Optional[SentimentService] = None):
        """
        Args:
            model_path: Checkpoint aspek; default dari settings.
            sentiment_service: SentimentService yang DIPAKAI BERSAMA. Bila None,
                service ini membuat instance sendiri (dipakai skrip dan tes).

        Berbagi instance itu penting karena dua alasan, dan keduanya nyata:

        **Memori.** `main.py` sudah membuat satu SentimentService global.
        Membuat satu lagi di sini berarti bobot model sentimen (~500 MB) dimuat
        DUA KALI dalam satu proses, di samping model aspek dan encoder topik.
        Pada kuota RAM Railway itu cukup untuk membuat kontainer dimatikan di
        tengah analisis - persis gejala "analisis gagal" yang tampak acak.
        Terpicu juga di suite tes sebagai
        `Windows fatal exception: access violation` saat memuat safetensors.

        **Kebenaran.** Setelah `/api/retrain/sentiment` berhasil, instance global
        memuat ulang bobot barunya. Salinan milik AspectService TIDAK, sehingga
        polaritas per-aspek diam-diam tetap memakai model LAMA - hasil koreksi
        pengguna tidak pernah sampai ke analisis aspek.
        """
        logger.info("Initializing IndoBERT-NER Aspect Service (lazy mode)...")

        self.sentiment_service = sentiment_service or SentimentService()
        # Sengaja TIDAK ada TextCleaner di sini. Ekstraksi aspek berbasis offset
        # karakter, jadi pembersihan apa pun akan menggeser seluruh span - itu
        # sebabnya PROFILES['span'] bernilai None. Dulu ada `self.text_cleaner =
        # TextCleaner()` yang tidak pernah dipakai satu kali pun, sehingga kode
        # dan komentar kebijakannya saling bertentangan.
        self.base_model_name = settings.aspect_base_model

        # Stemmer khusus pencocokan span aspek (bukan untuk membersihkan teks).
        # Dibuat terpisah dari TextCleaner supaya tidak bergantung pada config-nya.
        self._stemmer = StemmerFactory().create_stemmer()
        self._stem_cache: Dict[str, str] = {}

        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        # Label mapping untuk BIO tagging
        self.id2label = {0: 'O', 1: 'B-ASPECT', 2: 'I-ASPECT'}
        self.label2id = {'O': 0, 'B-ASPECT': 1, 'I-ASPECT': 2}
        self.num_labels = 3

        # Model loaded lazily on first use
        self.model = None
        self.tokenizer = None
        self.model_trained = False
        # Encoder terpisah untuk pemetaan aspek semantik; lihat _ensure_embedder.
        self._embedder = None
        self._embedder_failed = False
        self._prototype_cache: Dict[Tuple[str, ...], Any] = {}
        # Checkpoint hasil retraining dipulihkan setelah restart. Sebelumnya
        # AspectService() dipanggil tanpa argumen di main.py, sehingga
        # aspect_retrained.pt tidak pernah dibaca dan hasil active learning hilang.
        self._model_path = model_path or self._discover_checkpoint()
        self._loaded = False

        if self._model_path:
            logger.info(f"🔁 Checkpoint aspek ditemukan: {self._model_path}")

        # Stopwords bahasa Indonesia
        self.stopwords = set(factory.get_stop_words())
        
        # Common aspect keywords untuk post-processing
        self.common_aspects = {
            'pelayanan', 'layanan', 'service', 'kualitas', 'quality',
            'harga', 'price', 'kemasan', 'packaging', 'pengiriman', 'delivery',
            'rasa', 'taste', 'tekstur', 'ukuran', 'size', 'warna', 'color',
            'desain', 'design', 'fitur', 'feature', 'performa', 'performance',
            'ketahanan', 'durability', 'garansi', 'warranty',
            'respon', 'response', 'kecepatan', 'speed', 'admin', 'seller',
            'penjual', 'toko', 'shop', 'store', 'aplikasi', 'app',
            'tampilan', 'interface', 'menu', 'proses', 'hasil', 'kebersihan',
            'suasana', 'lokasi', 'tempat', 'parkir', 'akses', 'fasilitas',
            'customer', 'produk', 'barang', 'item', 'stok', 'stock',
            'cicilan', 'ongkir', 'ongkos', 'biaya', 'voucher', 'diskon',
            'promo', 'cashback', 'bonus', 'gratis', 'komplain', 'refund'
        }
        
        logger.info("✅ Aspect service initialized")
    
    @staticmethod
    def _discover_checkpoint() -> Optional[str]:
        """Kembalikan path checkpoint retraining aspek bila filenya ada."""
        path = settings.aspect_checkpoint_path
        return path if os.path.isfile(path) else None

    def _initialize_model(self, model_path: Optional[str] = None):
        """
        Initialize IndoBERT model untuk token classification (NER)
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_name, **hub_kwargs(self.base_model_name)
            )
            logger.info("✅ IndoBERT tokenizer loaded")
            
            if model_path:
                self._load_trained_model(model_path)
            else:
                self.model = AutoModelForTokenClassification.from_pretrained(
                    self.base_model_name,
                    **hub_kwargs(self.base_model_name),
                    num_labels=self.num_labels,
                    id2label=self.id2label,
                    label2id=self.label2id,
                    ignore_mismatched_sizes=True,
                    # Memuat langsung ke tensor model, tanpa salinan penuh di
                    # RAM lebih dulu - puncak memori kira-kira separuh. Alasan
                    # sama seperti pada SentimentService: tiga model besar dalam
                    # satu proses membuat lonjakan sesaat itu memicu OOM, dan di
                    # Windows muncul sebagai access violation di dalam torch.
                    low_cpu_mem_usage=True,
                )
                self.model.to(self.device)
                self.model.eval()
                self.model_trained = False
                logger.info("✅ IndoBERT-NER model initialized in zero-shot mode")
                
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise
    
    def _load_trained_model(self, model_path: str):
        """
        Load trained model dari checkpoint
        """
        try:
            self.model = AutoModelForTokenClassification.from_pretrained(
                self.base_model_name,
                **hub_kwargs(self.base_model_name),
                num_labels=self.num_labels,
                id2label=self.id2label,
                label2id=self.label2id,
                ignore_mismatched_sizes=True,
                # Lihat catatan pada _initialize_model: memuat hemat memori.
                low_cpu_mem_usage=True,
            )
            
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.to(self.device)
            self.model.eval()
            self.model_trained = True
            logger.info(f"✅ Trained model loaded from {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load trained model: {e}")
            logger.info("⚠️ Fallback to untrained model")
            self.model = AutoModelForTokenClassification.from_pretrained(
                self.base_model_name,
                **hub_kwargs(self.base_model_name),
                num_labels=self.num_labels,
                id2label=self.id2label,
                label2id=self.label2id,
                ignore_mismatched_sizes=True,
                # Lihat catatan pada _initialize_model: memuat hemat memori.
                low_cpu_mem_usage=True,
            )
            self.model.to(self.device)
            self.model.eval()
            self.model_trained = False

    def _ensure_loaded(self):
        """Lazy-load aspect model on first use to keep startup RAM low."""
        if self._loaded:
            return
        try:
            self._initialize_model(self._model_path)
            self._loaded = True
        except Exception as e:
            logger.error(f"Failed lazy-load aspect model: {e}")
            self._loaded = True  # Don't retry on every call

    async def analyze(
        self,
        texts: List[str],
        preprocessing_config: Optional[Union[Dict, Any]] = None,
        predefined_aspects: Optional[List[str]] = None,
        mode: str = "automatic"
    ) -> Dict[str, Any]:
        """
        Extract aspects from texts
        Backward compatible dengan service lama

        Args:
            texts: List of text untuk dianalisis
            preprocessing_config: Config untuk preprocessing (optional)
            predefined_aspects: List aspek predefined untuk rule-based mode
            mode: "automatic" atau "rule-based"

        Returns:
            Dict berisi aspects, aspect_sentiments, statistics, summary, document_aspects
        """
        # Lazy-load model on first call
        self._ensure_loaded()

        # Ekstraksi aspek berbasis span: offset karakter harus menunjuk teks asli,
        # sehingga teks TIDAK dibersihkan. Parameter tetap diterima demi
        # kompatibilitas dengan pemanggil (Laravel mengirim config yang sama ke
        # semua endpoint), tapi sengaja diabaikan.
        if preprocessing_config:
            logger.debug(
                "preprocessing_config diabaikan pada ekstraksi aspek: "
                "span aspek harus sejajar dengan teks asli"
            )

        try:
            logger.info(f"Analyzing {len(texts)} texts with mode: {mode}")
            
            if mode == "rule-based" and predefined_aspects:
                # TIDAK digabung: kategorinya dideklarasikan pengguna, jadi
                # menggabungkannya berarti mengabaikan pilihan mereka. Kalau
                # pengguna menyebut `gaji` dan `gajinya` sebagai dua kategori,
                # itu keputusan mereka.
                aspects = self._extract_rule_based(texts, predefined_aspects)
            else:
                if self.model_trained:
                    logger.info("Using trained IndoBERT-NER model for extraction")
                    aspects = self._extract_with_model(texts)
                else:
                    logger.info("Using zero-shot extraction (untrained model + heuristics)")
                    aspects = self._extract_zero_shot(texts)

                # Digabung SEBELUM sentimen/statistik/document_aspects dihitung,
                # supaya seluruh turunannya memakai daftar yang sama. Menggabung
                # belakangan akan menyisakan statistik per aspek yang terbelah.
                aspects = self._merge_aspect_variants(aspects)
            
            aspect_sentiments = await self._analyze_aspect_sentiments(texts, aspects)
            statistics = self._calculate_statistics(aspect_sentiments)
            summary = self._generate_summary(aspect_sentiments)
            document_aspects = self._build_document_aspects(aspects, len(texts))

            return {
                'aspects': aspects,
                'aspect_sentiments': aspect_sentiments,
                'statistics': statistics,
                'summary': summary,
                'document_aspects': document_aspects,
                # Bentuk dasar tiap aspek, sejajar indeks dengan document_aspects.
                # Dipakai untuk membandingkan dengan label manusia; tanpa ini
                # 'pelayanannya' vs 'pelayanan' terhitung salah dua kali.
                'document_aspects_normalized':
                    self._build_document_aspects_normalized(document_aspects),
            }
            
        except Exception as e:
            logger.error(f"Aspect extraction error: {str(e)}")
            raise
    
    def _build_document_aspects(self, aspects: List[Dict], num_texts: int) -> List[List[str]]:
        """
        Petakan indeks dokumen -> daftar nama aspek yang muncul di dalamnya.

        Aspek yang muncul beberapa kali dalam satu dokumen dicatat SEKALI saja.
        Sebelumnya setiap kemunculan ditambahkan, sehingga satu komentar
        panjang bisa menghasilkan 'pelayanan' dua belas kali berturut-turut.
        Duplikat itu tidak menambah informasi apa pun - jumlah kemunculan sudah
        tersimpan di aspects[].count - tetapi ikut tersimpan ke basis data dan
        tampil berulang di antarmuka Laravel.

        Args:
            aspects: Hasil ekstraksi, satu entri per aspek.
            num_texts: Jumlah teks masukan; panjang keluaran harus sama persis
                agar penjajaran indeks dengan pemanggil terjaga.

        Returns:
            List sepanjang num_texts berisi daftar nama aspek unik per dokumen,
            terurut sesuai kemunculan pertamanya.
        """
        doc_aspects: List[List[str]] = [[] for _ in range(num_texts)]
        seen: List[set] = [set() for _ in range(num_texts)]

        for aspect_data in aspects:
            name = aspect_data.get('aspect')
            if not name or 'occurrences' not in aspect_data:
                continue
            for occ in aspect_data['occurrences']:
                idx = occ.get('text_index', -1)
                if 0 <= idx < num_texts and name not in seen[idx]:
                    seen[idx].add(name)
                    doc_aspects[idx].append(name)

        return doc_aspects

    # Klitik posesif. Dipisahkan dari stemming Sastrawi dengan sengaja: ini
    # infleksi murni, tidak pernah mengubah makna maupun kelas kata, sehingga
    # aman dipotong tanpa kamus. Sastrawi hanya memotong imbuhan bila HASILNYA
    # ada di 29 933 kata dasarnya, jadi kosakata di luar kamus tidak tersentuh -
    # `alutsistanya` tetap `alutsistanya` dan `danramilnya` tetap `danramilnya`.
    # Justru kosakata semacam itu yang banyak muncul di korpus lembaga.
    CLITICS = ('nya', 'ku', 'mu')

    # Sisa minimal setelah klitik dipotong. Penjaga KEDUA, bukan yang utama.
    # Batas 3 memang melewatkan akronim yang banyak di korpus lembaga:
    # `tninya` -> `tni`, `kpknya` -> `kpk`. Kata yang berakhiran -nya tanpa
    # makna posesif dan menyisakan hanya 2 huruf (`punya`, `hanya`, `tanya`)
    # tetap tertolak di sini.
    MIN_STEM_AFTER_CLITIC = 3

    def _strip_clitic(self, word: str) -> str:
        """Potong klitik posesif dari SATU kata, bila aman.

        Penjaga UTAMA adalah kamus: bila kata utuhnya sendiri sebuah kata dasar,
        akhiran -nya-nya bukan klitik dan tidak boleh dipotong. Tanpa ini,
        batas panjang saja salah memotong kata dasar yang sah - diperiksa
        terhadap 29 933 kata dasar Sastrawi, ada 13 yang berakhiran -nya dan
        enam di antaranya lolos batas panjang: `makanya` -> `maka`,
        `empunya` -> `empu`, `bahwasanya` -> `bahwasa`, `mangkanya`,
        `adakalanya`, `segianya`. Semuanya kini tertahan oleh pemeriksaan kamus.

        Kosakata di luar kamus tetap terlayani, dan memang itu tujuannya:
        `alutsistanya`, `danramilnya`, `tninya` bukan kata dasar, jadi
        klitiknya dipotong seperti seharusnya.
        """
        if word in ROOT_WORDS:
            return word

        for klitik in self.CLITICS:
            if word.endswith(klitik):
                sisa = word[: -len(klitik)]
                if len(sisa) >= self.MIN_STEM_AFTER_CLITIC:
                    return sisa
        return word

    def _strip_clitics(self, name: str) -> str:
        """Versi frasa: tiap kata dibersihkan dari klitiknya.

        `pelatihan prajuritnya` -> `pelatihan prajurit`.

        Klitik yang ditulis TERPISAH juga dibuang: penulis komentar sering
        mengetik `gaji nya`, `tunjangan nya`, `gedung nya`. Terukur pada studi
        kasus enam lembaga, bentuk berspasi itu tidak pernah bergabung dengan
        induknya sehingga `gaji` dan `gaji nya` tetap menjadi dua aspek. Klitik
        yang berdiri sendiri tidak pernah menjadi aspek, jadi aman dibuang -
        tetapi hanya bila masih ada kata lain yang tersisa.
        """
        kata = name.split()
        while len(kata) > 1 and kata[-1] in self.CLITICS:
            kata.pop()
        return ' '.join(self._strip_clitic(k) for k in kata)

    # Awalan derivasional Bahasa Indonesia. Dipakai HANYA untuk memutuskan
    # apakah dua bentuk permukaan yang berstem sama boleh digabung - bukan untuk
    # memotong kata. Urut dari yang terpanjang supaya `peng` cocok lebih dulu
    # daripada `pe`.
    MERGE_PREFIXES = (
        'menge', 'peng', 'meng', 'pem', 'pen', 'per', 'mem', 'men', 'ber',
        'ter', 'pe', 'me', 'di', 'ke', 'se',
    )

    def _prefix_class(self, surface: str, stem: str) -> str:
        """Awalan derivasional yang dibawa `surface` terhadap stem-nya.

        Ini penjaga terhadap salah gabung. Stem Sastrawi meratakan awalan
        pembentuk pelaku: `petugas` dan `tugas` sama-sama menjadi `tugas`,
        padahal keduanya aspek yang BERBEDA - satu orangnya, satu pekerjaannya.
        Dengan membandingkan kelas awalan lebih dulu, `petugas`+`petugasnya`
        tetap bergabung sementara `tugasnya` berdiri sendiri.

        Bentuk yang identik dengan stem-nya tidak dianggap berawalan, sehingga
        kata seperti `perang` (stem `perang`) tidak salah dibaca sebagai `per-`.
        """
        if surface == stem:
            return ''
        for awalan in self.MERGE_PREFIXES:
            if surface.startswith(awalan) and not stem.startswith(awalan):
                return awalan
        return ''

    def _merge_key(self, name: str) -> Tuple[str, str]:
        """Kunci penggabungan: (stem, kelas awalan).

        Klitik dipotong LEBIH DULU. Tanpa itu `alutsistanya` dan `alutsista`
        tidak pernah bertemu, karena Sastrawi meninggalkan keduanya apa adanya -
        dan kosakata di luar kamusnya justru yang paling banyak di korpus
        lembaga (alutsista, danramil, kodim).
        """
        bersih = self._strip_clitics(self._normalize_aspect_text(name))
        stem = self.normalize_aspect_name(bersih)
        return stem, self._prefix_class(bersih.replace(' ', ''), stem.replace(' ', ''))

    def _merge_aspect_variants(self, aspects: List[Dict]) -> List[Dict]:
        """Gabungkan aspek yang hanya berbeda bentuk infleksi.

        Bahasa Indonesia aglutinatif, dan ekstraksi mengembalikan kata seperti
        yang tertulis - sehingga satu aspek yang sama terpecah menjadi beberapa
        entri: `gaji` dan `gajinya`, `pelayanan` dan `pelayanannya`. Dasbor lalu
        menampilkan aspek kembar dengan jumlah terbelah, dan statistik per aspek
        ikut terbelah.

        Terukur pada 22 komentar bergaya TNI: 24 aspek menjadi 18.

        Label yang ditampilkan adalah **bentuk permukaan paling sering**, bukan
        stem-nya. Prinsip yang sama dengan `_make_words_readable()` di modul
        topik: memodelkan pada stem itu benar, tetapi pengguna tidak boleh
        disuguhi `layan` ketika yang mereka tulis `pelayanan`. Seri dipecah oleh
        bentuk terpendek lalu abjad, supaya hasilnya sama di setiap pemanggilan.

        `variants` memuat seluruh bentuk yang tergabung, dan `merged_from`
        menyebut entri asalnya - jadi penggabungan yang salah tetap terlihat,
        tidak tersembunyi.

        Batasannya diketahui dan TIDAK dihilangkan oleh penjaga awalan: bentuk
        yang berbeda hanya pada AKHIRAN derivasional tetap bergabung, mis.
        `pendidikan` (prosesnya) dengan `pendidik` (orangnya) - keduanya berstem
        `didik` dan sama-sama berawalan `pen`. Memisahkannya butuh pengetahuan
        kelas kata, bukan morfologi permukaan.
        """
        if not settings.aspect_merge_variants:
            return aspects

        kelompok: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
        for entri in aspects:
            nama = entri.get('aspect')
            if not nama:
                continue
            kelompok[self._merge_key(nama)].append(entri)

        hasil = []
        for anggota in kelompok.values():
            if len(anggota) == 1:
                # Tetap dibersihkan klitiknya. Aspek yang hanya muncul sebagai
                # `anggarannya` tidak punya varian untuk digabung, tetapi
                # labelnya tetap harus terbaca `anggaran` - itu yang dilihat
                # pengguna di dasbor.
                tunggal = dict(anggota[0])
                rapi = self._strip_clitics(
                    self._normalize_aspect_text(tunggal['aspect'])
                )
                if rapi and rapi != tunggal['aspect']:
                    tunggal.setdefault('variants', [])
                    tunggal['variants'] = sorted(
                        set(tunggal['variants']) | {tunggal['aspect']}
                    )
                    tunggal['aspect'] = rapi
                hasil.append(tunggal)
                continue

            # Label = bentuk yang paling sering muncul; seri dipecah secara
            # deterministik supaya dua analisis atas korpus yang sama tidak
            # menghasilkan label berbeda.
            utama = max(
                anggota,
                key=lambda e: (e.get('count', 0), -len(e['aspect']), e['aspect']),
            )

            gabungan = dict(utama)
            gabungan['aspect'] = self._strip_clitics(
                self._normalize_aspect_text(utama['aspect'])
            ) or utama['aspect']
            gabungan['count'] = sum(e.get('count', 0) for e in anggota)
            gabungan['occurrences'] = [
                occ for e in anggota for occ in e.get('occurrences', [])
            ]

            varian = set()
            for e in anggota:
                varian.update(e.get('variants') or [])
                varian.add(e['aspect'])
            gabungan['variants'] = sorted(varian)
            gabungan['merged_from'] = sorted(e['aspect'] for e in anggota)

            # Confidence dirata-rata BERBOBOT jumlah kemunculan; rata-rata polos
            # akan membuat satu varian langka menarik angkanya sekuat varian
            # yang muncul puluhan kali.
            berbobot = [
                (e.get('confidence'), e.get('count', 0))
                for e in anggota
                if e.get('confidence') is not None
            ]
            total_n = sum(n for _, n in berbobot)
            if total_n:
                gabungan['confidence'] = float(
                    sum(c * n for c, n in berbobot) / total_n
                )

            gabungan['score'] = (
                gabungan['count'] * gabungan['confidence']
                if gabungan.get('confidence') is not None
                else gabungan['count']
            )
            hasil.append(gabungan)

        hasil.sort(key=lambda e: e.get('score', e.get('count', 0)), reverse=True)

        digabung = sum(1 for e in hasil if e.get('merged_from'))
        if digabung:
            logger.info(
                f"Aspek digabung berdasarkan bentuk dasar: "
                f"{len(aspects)} -> {len(hasil)} ({digabung} kelompok)"
            )
        return hasil

    def normalize_aspect_name(self, name: str) -> str:
        """Bentuk dasar sebuah nama aspek, untuk PEMBANDINGAN - bukan untuk tampilan.

        Mode `automatic` mengembalikan bentuk permukaan seperti yang tertulis di
        teks (`pelayanannya`, `pajaknya`), sedangkan manusia menulis label emas
        dalam bentuk dasar (`pelayanan`, `pajak`). Bahasa Indonesia aglutinatif,
        jadi mencocokkan keduanya sebagai string persis akan menghitung pasangan
        yang sebenarnya BENAR sebagai false positive sekaligus false negative.

        Terukur pada 6 kalimat / 8 aspek emas, memakai keluaran `document_aspects`
        yang sama: pencocokan string persis memberi micro-F1 **0,118**
        (TP=1 FP=8 FN=7), sedangkan pencocokan setelah stemming memberi
        **0,941** (TP=8 FP=1 FN=0). Selisih itu seluruhnya artefak pengukuran -
        modelnya menemukan aspek yang benar pada kedua hitungan.

        Ini mesin yang sama dengan yang dipakai `_find_aspect_spans` untuk
        mencocokkan `pejabat` dengan `pejabatnya`, jadi evaluasi memakai definisi
        kecocokan yang sama dengan ekstraksinya sendiri.

        Frasa banyak-kata di-stem per kata lalu digabung lagi.
        """
        bersih = self._normalize_aspect_text(str(name or ''))
        if not bersih:
            return ''
        return ' '.join(self._stem(kata) for kata in bersih.split())

    def _build_document_aspects_normalized(
        self, doc_aspects: List[List[str]]
    ) -> List[List[str]]:
        """Versi bentuk-dasar dari `document_aspects`, sejajar indeks dan isinya.

        Dikirim BERDAMPINGAN, bukan menggantikan: bentuk permukaan tetap yang
        ditampilkan ke pengguna (`pelayanannya` adalah kata yang benar-benar
        ditulis), sementara bentuk dasar dipakai pihak yang perlu MEMBANDINGKAN
        dengan label manusia - terutama evaluasi hold-out di sisi Laravel.
        """
        hasil: List[List[str]] = []
        for daftar in doc_aspects:
            terlihat = set()
            baris = []
            for nama in daftar:
                dasar = self.normalize_aspect_name(nama)
                if dasar and dasar not in terlihat:
                    terlihat.add(dasar)
                    baris.append(dasar)
            hasil.append(baris)
        return hasil

    def _normalize_aspect_text(self, text: str) -> str:
        """
        Normalize aspect text: strip punctuation, lowercase, collapse whitespace.
        Ensures 'pajak.' and 'pajak' map to the same aspect.
        """
        text = text.strip()
        text = re.sub(r'^[^\w]+|[^\w]+$', '', text)
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _chunked_forward_pass(
        self,
        texts: List[str],
        max_chunk_size: int = 512,
        max_length: Optional[int] = None
    ) -> tuple:
        """
        Tokenize and run model inference in memory-safe chunks.
        Results are concatenated — NOT batched separately.
        """
        # Batas inferensi terpisah dari batas pelatihan: aspek yang berada di
        # luar 128 token pertama sebelumnya hilang tanpa jejak.
        if max_length is None:
            max_length = settings.aspect_inference_max_length

        all_predictions = []
        all_offsets = []
        all_probs = []

        for i in range(0, len(texts), max_chunk_size):
            chunk_texts = texts[i:i + max_chunk_size]
            encoding = self.tokenizer(
                chunk_texts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=max_length,
                return_offsets_mapping=True
            )
            offset_mapping = encoding.pop('offset_mapping')
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                probs = torch.softmax(logits, dim=-1).cpu().numpy()

            all_predictions.append(preds)
            all_offsets.append(offset_mapping)
            all_probs.append(probs)

            del input_ids, attention_mask, logits
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        predictions = np.concatenate(all_predictions, axis=0)
        offsets = torch.cat(all_offsets, dim=0)
        probs = np.concatenate(all_probs, axis=0)

        return predictions, offsets, probs

    # ── Pemetaan aspek semantik ─────────────────────────────────────────────
    #
    # Kalimat prototipe untuk mewakili sebuah kategori aspek. Label mentah
    # ("pelayanan") tidak bisa langsung dibandingkan dengan klausa: encoder
    # dilatih pada kalimat, sehingga membandingkan kata-lepas melawan kalimat
    # memicu efek hub - satu label menyedot semua kandidat.
    #
    # TERUKUR, dan inilah alasan bentuknya begini. Pada set uji 19 istilah /
    # 13 kalimat dengan tiga kategori (pelayanan, harga, infrastruktur):
    #   kata-vs-kata IndoBERT mentah      : 12/15 benar, celah pemisah -0,117
    #   kata-vs-kata multilingual-e5-small: 10/15 benar, celah -0,064
    #   klausa-vs-prototipe + centering   :   9/9 benar, celah -0,020
    # Dua pendekatan pertama TIDAK dipakai: kata asing ("cuaca" 0,83,
    # "makanan" 0,82) menyamai kandidat benar ("angkutan umum" 0,72), sehingga
    # tidak ada ambang yang memisahkan "masuk aspek" dari "tidak masuk aspek".
    _ASPECT_TEMPLATES = (
        "Ulasan ini membahas {}.",
        "Masalahnya ada pada {}.",
        "Saya mengeluhkan {}.",
        "Menurut saya {} nya bagus.",
        "Soal {}, saya kurang puas.",
    )

    # Latar generik untuk centering. Embedding BERT bersifat anisotropik -
    # seluruh vektor menumpuk pada satu kerucut sempit sehingga kosinus mentah
    # nyaris tak informatif. Mengurangi rata-rata latar memulihkan kontras
    # (rata-rata kandidat benar 0,857 -> 0,410 sementara kandidat asing
    # 0,760 -> 0,173, yakni rasio pisah 1,13x -> 2,37x).
    #
    # Latar sengaja berupa KONSTANTA, bukan rata-rata korpus permintaan:
    # korpus yang kebetulan seluruhnya membahas satu aspek akan menghapus
    # justru sinyal yang dicari, dan hasilnya jadi bergantung pada komposisi
    # batch - melanggar keterulangan yang dibutuhkan angka skripsi.
    _CENTERING_BACKGROUND = (
        "Hari ini saya pergi ke pasar.",
        "Buku itu tebal sekali.",
        "Dia berjalan pulang sore hari.",
        "Anak-anak bermain di lapangan.",
        "Mobil itu berwarna merah.",
        "Kami makan bersama keluarga.",
        "Udara pagi terasa sejuk.",
        "Pekerjaan itu selesai tepat waktu.",
        "Dia membaca koran setiap pagi.",
        "Sekolah libur minggu depan.",
    )

    def _ensure_embedder(self) -> Optional[Any]:
        """
        Muat encoder IndoBERT DASAR untuk penyematan, sekali saja.

        Encoder K6 hasil fine-tuning sengaja TIDAK dipakai di sini, walau sudah
        ada di memori. Fine-tuning penandaan BIO menata ulang ruang embedding ke
        arah "apakah token ini aspek" dan meruntuhkan struktur topikalnya.
        Terukur pada set uji 12 klausa / 3 kategori:

            encoder dasar   : 9/10 benar, celah pemisah +0,138 (0,260 vs 0,122)
            encoder K6      : 7/10 benar, celah pemisah +0,001 (0,201 vs 0,200)

        Dengan celah +0,001 tidak ada ambang yang memisahkan klausa dalam-aspek
        dari klausa asing, sehingga jalur semantik jadi tak berguna.

        Bobotnya berasal dari checkpoint yang sama dengan yang dipakai aspek,
        jadi tidak ada unduhan baru - hanya tambahan RAM, dan itu pun ditunda
        sampai mode rule-based benar-benar dipakai (mode automatic tidak
        membayar apa pun).

        Returns:
            Model encoder, atau None bila pemuatan gagal.
        """
        if getattr(self, '_embedder', None) is not None:
            return self._embedder
        if getattr(self, '_embedder_failed', False):
            return None

        try:
            from transformers import AutoModel
            logger.info("Memuat encoder dasar untuk pemetaan aspek semantik...")
            self._embedder = AutoModel.from_pretrained(
                self.base_model_name, **hub_kwargs(self.base_model_name)
            ).to(self.device).eval()
            return self._embedder
        except Exception as exc:
            logger.warning(f"Encoder penyemat gagal dimuat: {exc}")
            self._embedder_failed = True
            return None

    @torch.no_grad()
    def _embed(self, texts: List[str]) -> Optional[torch.Tensor]:
        """
        Encode teks menjadi vektor mean-pooled.

        Args:
            texts: Teks yang akan di-encode.

        Returns:
            Tensor [len(texts), hidden], atau None bila encoder tak tersedia.
        """
        if not texts or self.tokenizer is None:
            return None

        encoder = self._ensure_embedder()
        if encoder is None:
            return None

        vectors: List[torch.Tensor] = []

        # Dipotong seperti jalur inferensi lain demi batas RAM yang sama.
        for i in range(0, len(texts), 128):
            chunk = texts[i:i + 128]
            enc = self.tokenizer(
                chunk,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=64,
            ).to(self.device)

            hidden = encoder(**enc).last_hidden_state
            mask = enc['attention_mask'].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            vectors.append(pooled.cpu())

            del enc, hidden, mask, pooled
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

        return torch.cat(vectors, dim=0)

    @torch.no_grad()
    def _embed_spans(
        self, items: List[Tuple[str, int, int]]
    ) -> Optional[torch.Tensor]:
        """
        Sematkan ISTILAH pada posisinya, dengan konteks kalimat penuh.

        Yang dipetakan ke kategori adalah istilah aspek, bukan topik kalimat.
        Menyematkan seluruh klausa membuat penugasan digerakkan topik: pada
        korpus yang semuanya membahas pajak, 'DPR' ikut dipetakan ke kategori
        'pajak' (0,52) alih-alih 'pemerintah', karena klausanya memang bicara
        pajak. Mean-pool atas token milik istilah - dihitung di dalam kalimat
        penuh sehingga tetap peka konteks - memperbaikinya.

        Terukur pada dua domain (17 kasus berlabel + kalimat asing):
            klausa penuh   : 14/17 benar, celah -0,086
            span berkonteks: 15/17 benar, celah -0,071   <- dipakai
            istilah lepas  : 14/17 benar, celah -0,133

        Args:
            items: Tuple (teks, start, end) untuk tiap istilah.

        Returns:
            Tensor [len(items), hidden], atau None bila encoder tak tersedia.
        """
        if not items or self.tokenizer is None:
            return None

        encoder = self._ensure_embedder()
        if encoder is None:
            return None

        vectors: List[torch.Tensor] = []

        for i in range(0, len(items), 64):
            chunk = items[i:i + 64]
            enc = self.tokenizer(
                [t for t, _, _ in chunk],
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=128,
                return_offsets_mapping=True,
            )
            offsets = enc.pop('offset_mapping')
            enc = enc.to(self.device)
            hidden = encoder(**enc).last_hidden_state

            for row, (_, start, end) in enumerate(chunk):
                selected = [
                    k for k, (a, b) in enumerate(offsets[row].tolist())
                    if b > a and a < end and b > start
                ]
                if selected:
                    vectors.append(hidden[row, selected].mean(0).cpu())
                else:
                    # Istilah terpotong oleh truncation: pakai rata-rata kalimat
                    # agar panjang keluaran tetap sejajar dengan masukan.
                    mask = enc['attention_mask'][row].unsqueeze(-1).float()
                    vectors.append(
                        ((hidden[row] * mask).sum(0) / mask.sum().clamp(min=1e-9)).cpu()
                    )

            del enc, hidden, offsets
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

        return torch.stack(vectors, dim=0)

    def _aspect_prototypes(
        self, aspects: List[str]
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Bangun vektor prototipe untuk tiap kategori aspek yang dideklarasikan.

        Args:
            aspects: Nama kategori dari pengguna.

        Returns:
            (prototipe [n_aspek, hidden] sudah dinormalisasi, vektor latar),
            atau None bila encoder tidak tersedia.
        """
        cache_key = tuple(aspects)
        cached = self._prototype_cache.get(cache_key)
        if cached is not None:
            return cached

        flat = [t.format(a) for a in aspects for t in self._ASPECT_TEMPLATES]
        embedded = self._embed(flat + list(self._CENTERING_BACKGROUND))
        if embedded is None:
            return None

        n_tmpl = len(self._ASPECT_TEMPLATES)
        proto_raw = embedded[:len(flat)].reshape(len(aspects), n_tmpl, -1).mean(1)
        background = embedded[len(flat):].mean(0, keepdim=True)

        protos = torch.nn.functional.normalize(proto_raw - background, dim=-1)

        self._prototype_cache[cache_key] = (protos, background)
        return protos, background

    def _semantic_matches(
        self,
        texts: List[str],
        wanted: List[str],
        lexical_spans: Dict[int, List[Tuple[int, int]]],
    ) -> List[Dict[str, Any]]:
        """
        Petakan istilah aspek temuan model ke kategori yang dideklarasikan.

        Dua gerbang independen dipakai, dan keduanya diperlukan karena masing
        masing gagal pada contoh yang BERBEDA:

        1. Ekstraktor aspek terlatih harus menemukan istilah di kalimat itu.
           Terukur: gerbang ini sendiri sudah menolak "Film itu durasinya
           terlalu panjang" (tak ada istilah aspek) yang justru LOLOS ambang
           kemiripan dengan skor 0,291 - di atas kandidat benar terendah 0,271.
        2. Klausa di sekitar istilah harus cukup mirip dengan prototipe
           kategori, dan cukup unggul atas kategori peringkat dua. Gerbang ini
           menolak "cuaca" (0,07) dan "kucing" (0,12) yang lolos gerbang 1.

        Digabung, keduanya mencapai 13/13 pada set uji tersebut.

        Penyaringan terhadap lapis 1 dilakukan per-SPAN, bukan per-dokumen.
        Melewati seluruh dokumen yang sudah punya kecocokan harfiah tampak
        hemat, tetapi menghilangkan aspek lain di kalimat yang sama: pada
        "Pelayanannya bagus tapi antriannya lama sekali", 'pelayanannya' cocok
        harfiah lalu 'antriannya' tidak pernah dinilai - aspek 'pelayanan'
        tercatat positif saja, padahal keluhannya justru ada di klausa kedua.

        Args:
            texts: Seluruh teks masukan.
            wanted: Kategori aspek dari pengguna.
            lexical_spans: Span hasil lapis 1 per indeks teks; kandidat yang
                bertindihan dengannya dibuang agar tidak dihitung dua kali.

        Returns:
            List of {aspect, text_index, start, end, surface, score}.
        """
        candidates = [
            (i, t) for i, t in enumerate(texts) if t and t.strip()
        ]
        if not candidates:
            return []

        prototypes = self._aspect_prototypes(wanted)
        if prototypes is None:
            logger.warning("Encoder tidak tersedia; pemetaan semantik dilewati")
            return []
        protos, background = prototypes

        # ── Gerbang 1: ekstraktor menemukan istilah aspek ───────────────────
        try:
            extracted = self._extract_with_model([t for _, t in candidates])
        except Exception as exc:
            logger.warning(f"Pemetaan semantik dilewati, ekstraksi gagal: {exc}")
            return []

        local_to_global = {local: g for local, (g, _) in enumerate(candidates)}

        occurrences: List[Dict[str, Any]] = []
        for entry in extracted:
            for occ in entry.get('occurrences', []):
                start, end = occ.get('start'), occ.get('end')
                if start is None or end is None:
                    continue
                global_index = local_to_global.get(occ.get('text_index'))
                if global_index is None:
                    continue
                # Buang kandidat yang posisinya sudah ditandai lapis 1.
                if any(
                    start < ls_end and end > ls_start
                    for ls_start, ls_end in lexical_spans.get(global_index, ())
                ):
                    continue
                occurrences.append({
                    'text_index': global_index,
                    'start': start,
                    'end': end,
                    'surface': texts[global_index][start:end],
                })

        if not occurrences:
            return []

        # ── Gerbang 2: kemiripan istilah terhadap prototipe kategori ────────
        embedded = self._embed_spans([
            (texts[o['text_index']], o['start'], o['end']) for o in occurrences
        ])
        if embedded is None:
            return []

        vectors = torch.nn.functional.normalize(embedded - background, dim=-1)
        scores = vectors @ protos.T

        min_score = settings.aspect_semantic_min_score
        min_margin = settings.aspect_semantic_min_margin

        matches: List[Dict[str, Any]] = []
        for row, occ in zip(scores, occurrences):
            ranked = torch.argsort(row, descending=True)
            best = int(ranked[0])
            best_score = float(row[best])
            runner_up = float(row[int(ranked[1])]) if len(ranked) > 1 else -1.0

            if best_score < min_score or (best_score - runner_up) < min_margin:
                continue

            matches.append({
                'aspect': wanted[best],
                'text_index': occ['text_index'],
                'start': occ['start'],
                'end': occ['end'],
                'surface': occ['surface'],
                'score': round(best_score, 4),
            })

        logger.info(
            f"Pemetaan semantik: {len(occurrences)} istilah kandidat -> "
            f"{len(matches)} dipetakan ke {len(wanted)} kategori"
        )
        return matches

    def _extract_rule_based(self, texts: List[str], predefined_aspects: List[str]) -> List[Dict]:
        """
        Ekstraksi memakai daftar aspek yang ditentukan pemanggil.

        Mengembalikan struktur yang SAMA dengan _extract_with_model dan
        _extract_zero_shot, yaitu satu entri per aspek dengan daftar
        occurrences. Sebelumnya method ini mengembalikan struktur per-dokumen
        ({text, aspects[]}), sehingga seluruh hilir - _analyze_aspect_sentiments,
        _build_document_aspects, _calculate_statistics - melewatinya diam-diam
        dan mode rule-based menghasilkan aspect_sentiments kosong, statistik
        nol, serta document_aspects kosong.

        Pencocokan berjalan DUA LAPIS, karena mencocokkan string saja bukan
        klasifikasi aspek:

        Lapis 1 - harfiah. _find_aspect_spans, toleran imbuhan ('pelayanan'
        menemukan 'layanannya') dan menandai SEMUA kemunculan. Presisi tinggi,
        offset persis, jadi lapis ini selalu menang bila cocok.

        Lapis 2 - semantik (_semantic_matches). Untuk teks yang tidak tersentuh
        lapis 1: istilah aspek dicari dengan model terlatih, lalu dipetakan ke
        kategori terdekat. Tanpa lapis ini "Antriannya lama, petugasnya lambat"
        tidak pernah masuk ke kategori "pelayanan" yang diminta pengguna.

        Terukur pada set uji 6 kalimat / 3 kategori: recall pemetaan naik dari
        2/6 (33%) menjadi 6/6, tanpa satu pun kalimat asing ikut terpetakan.

        Setiap kemunculan menyimpan match_type ('lexical'/'semantic') dan skor
        kemiripannya, sehingga keputusan pemetaan dapat diaudit - penting
        karena lapis 2 bersifat probabilistik sedangkan lapis 1 deterministik.

        Args:
            texts: Daftar teks yang dianalisis.
            predefined_aspects: Aspek yang diminta pemanggil.

        Returns:
            List of {aspect, count, occurrences, score, variants}.
        """
        wanted = [a for a in (predefined_aspects or []) if str(a).strip()]
        if not wanted:
            return []

        collected: Dict[str, Dict[str, Any]] = {}

        def record(key: str, text: str, idx: int, start: int, end: int,
                   match_type: str, score: Optional[float] = None) -> None:
            entry = collected.setdefault(key, {
                'aspect': key,
                'count': 0,
                'occurrences': [],
                'variants': set(),
                'lexical': 0,
                'semantic': 0,
            })
            entry['count'] += 1
            entry[match_type] += 1
            entry['variants'].add(text[start:end])
            occurrence = {
                'text': text,
                'context': self._extract_context(key, text),
                'text_index': idx,
                'start': start,
                'end': end,
                'match_type': match_type,
            }
            if score is not None:
                occurrence['similarity'] = score
            entry['occurrences'].append(occurrence)

        # ── Lapis 1: pencocokan harfiah ─────────────────────────────────────
        lexical_spans: Dict[int, List[Tuple[int, int]]] = defaultdict(list)

        for idx, text in enumerate(texts):
            for aspect in wanted:
                spans, _ = self._find_aspect_spans(text, [aspect])
                if not spans:
                    continue
                key = self._normalize_aspect_text(aspect)
                for start, end in spans:
                    lexical_spans[idx].append((start, end))
                    record(key, text, idx, start, end, 'lexical')

        # ── Lapis 2: pemetaan semantik untuk span yang belum tertandai ──────
        if settings.aspect_semantic_mapping:
            for match in self._semantic_matches(texts, wanted, lexical_spans):
                record(
                    self._normalize_aspect_text(match['aspect']),
                    texts[match['text_index']],
                    match['text_index'],
                    match['start'],
                    match['end'],
                    'semantic',
                    match['score'],
                )

        results = [{
            'aspect': data['aspect'],
            'count': data['count'],
            'occurrences': data['occurrences'],
            'score': data['count'],
            'variants': sorted(data['variants']),
            'mode': 'rule-based',
            'lexical_matches': data['lexical'],
            'semantic_matches': data['semantic'],
        } for data in collected.values()]

        results.sort(key=lambda x: x['count'], reverse=True)
        logger.info(
            f"Rule-based menemukan {len(results)} aspek dari {len(wanted)} "
            f"yang diminta "
            f"(harfiah {sum(r['lexical_matches'] for r in results)}, "
            f"semantik {sum(r['semantic_matches'] for r in results)})"
        )
        return results
    
    def _extract_with_model(self, texts: List[str]) -> List[Dict]:
        """
        Extract aspects menggunakan trained IndoBERT-NER model.
        Seluruh texts diproses dengan chunked forward pass untuk mencegah OOM.
        """
        logger.info("Extracting aspects with trained model (chunked)...")

        predictions, offset_mappings, probs = self._chunked_forward_pass(texts)

        # ── Decode hasil prediksi per teks ──────
        all_aspects = defaultdict(lambda: {
            'count': 0,
            'occurrences': [],
            'variants': set(),
            'confidence': []
        })

        for idx, (text, preds, offsets, prob) in enumerate(
            zip(texts, predictions, offset_mappings, probs)
        ):
            aspects = self._decode_predictions(text, preds, offsets, prob)

            for aspect in aspects:
                aspect_name = self._normalize_aspect_text(aspect['text'])

                if len(aspect_name) < 3 or aspect_name in self.stopwords:
                    continue

                confidence = aspect.get('confidence', 0.0)
                all_aspects[aspect_name]['count'] += 1
                all_aspects[aspect_name]['occurrences'].append({
                    'text': text,
                    'context': aspect.get('context', ''),
                    'text_index': idx,
                    # Offset dipakai untuk mengambil klausa saat menghitung
                    # sentimen per-aspek.
                    'start': aspect.get('start'),
                    'end': aspect.get('end'),
                })
                all_aspects[aspect_name]['variants'].add(aspect['text'])
                all_aspects[aspect_name]['confidence'].append(confidence)

        # ── Convert to list format ──────────────────────────────────────────────
        aspect_results = []
        for aspect_name, data in all_aspects.items():
            avg_confidence = np.mean(data['confidence']) if data['confidence'] else 0.0
            aspect_results.append({
                'aspect':      aspect_name,
                'count':       data['count'],
                'occurrences': data['occurrences'],
                'score':       data['count'] * avg_confidence,
                'variants':    list(data['variants']),
                'confidence':  float(avg_confidence)
            })

        aspect_results.sort(key=lambda x: x['score'], reverse=True)
        logger.info(f"Extracted {len(aspect_results)} aspects with trained model")
        return aspect_results
    
    def _extract_zero_shot(self, texts: List[str]) -> List[Dict]:
        """
        Zero-shot extraction menggunakan untrained model + heuristics.
        Seluruh texts diproses dengan chunked forward pass untuk mencegah OOM.
        Hybrid approach: Model predictions (low confidence) + keyword matching (high confidence)
        """
        logger.info("Extracting aspects with zero-shot mode (chunked)...")

        predictions, offset_mappings, probs = self._chunked_forward_pass(texts)

        # ── Decode hasil prediksi + merge keyword per teks ─────────────────────
        all_aspects = defaultdict(lambda: {
            'count': 0,
            'occurrences': [],
            'variants': set()
        })

        for idx, (text, preds, offsets, prob) in enumerate(
            zip(texts, predictions, offset_mappings, probs)
        ):
            # Approach 1: decode model predictions dengan confidence filtering
            model_aspects = self._decode_predictions_with_confidence(
                text, preds, offsets, prob, threshold=0.5
            )

            # Approach 2: keyword-based extraction (lebih reliable)
            keyword_aspects = self._extract_keyword_aspects(text)

            # Merge: keywords prioritas lebih tinggi
            combined_aspects = self._merge_aspects(model_aspects, keyword_aspects)

            for aspect in combined_aspects:
                aspect_name = self._normalize_aspect_text(aspect['text'])

                if aspect_name in self.stopwords or len(aspect_name) < 3:
                    continue

                context = self._extract_context(aspect_name, text)
                start = aspect.get('start')
                end = aspect.get('end')
                if start is None:
                    found = text.lower().find(aspect_name)
                    if found != -1:
                        start, end = found, found + len(aspect_name)
                all_aspects[aspect_name]['count'] += 1
                all_aspects[aspect_name]['occurrences'].append({
                    'text': text,
                    'context': context,
                    'text_index': idx,
                    'start': start,
                    'end': end,
                })
                all_aspects[aspect_name]['variants'].add(aspect['text'])

        # ── Filter: minimal muncul di 2 dokumen ATAU common aspect ─────────────
        min_freq = max(2, len(texts) * 0.05)
        filtered_aspects = {
            name: data
            for name, data in all_aspects.items()
            if data['count'] >= min_freq or name in self.common_aspects
        }

        # ── Convert to list format ──────────────────────────────────────────────
        aspect_results = []
        for aspect_name, data in filtered_aspects.items():
            aspect_results.append({
                'aspect':      aspect_name,
                'count':       data['count'],
                'occurrences': data['occurrences'],
                'score':       data['count'],
                'variants':    list(data['variants'])
            })

        aspect_results.sort(key=lambda x: x['count'], reverse=True)
        logger.info(f"Extracted {len(aspect_results)} aspects with zero-shot mode")
        return aspect_results
    
    def _extract_keyword_aspects(self, text: str) -> List[Dict]:
        """
        Extract aspects berdasarkan common aspect keywords
        Lebih reliable untuk untrained model
        """
        text_lower = text.lower()
        words = re.findall(r'\b[a-z]+\b', text_lower)
        
        aspects = []
        
        # Extract unigrams
        for word in words:
            if word in self.common_aspects and word not in self.stopwords:
                aspects.append({
                    'text': word,
                    'confidence': 1.0,
                    'source': 'keyword'
                })
        
        # Extract bigrams
        for i in range(len(words) - 1):
            bigram = f"{words[i]} {words[i+1]}"
            if bigram in self.common_aspects:
                aspects.append({
                    'text': bigram,
                    'confidence': 1.0,
                    'source': 'keyword'
                })
        
        # Deduplicate
        seen = set()
        unique_aspects = []
        for asp in aspects:
            if asp['text'] not in seen:
                seen.add(asp['text'])
                unique_aspects.append(asp)
        
        return unique_aspects
    
    def _merge_aspects(
        self, 
        model_aspects: List[Dict], 
        keyword_aspects: List[Dict]
    ) -> List[Dict]:
        """
        Merge aspects dari model dan keywords
        Priority: keywords (confidence 1.0) > high-confidence model predictions
        """
        merged = {asp['text']: asp for asp in keyword_aspects}
        
        for asp in model_aspects:
            if asp['text'] not in merged:
                if asp.get('confidence', 0) > 0.6:
                    merged[asp['text']] = asp
        
        return list(merged.values())
    
    def _decode_predictions(
        self, 
        text: str, 
        predictions: np.ndarray, 
        offset_mapping: torch.Tensor,
        probs: Optional[np.ndarray] = None
    ) -> List[Dict]:
        """
        Decode BIO predictions ke aspect spans
        """
        aspects = []
        current_offsets = []
        current_confidences = []
        
        for idx, (pred, offset) in enumerate(zip(predictions, offset_mapping)):
            if offset[0] == 0 and offset[1] == 0:
                continue
            
            label = self.id2label.get(int(pred), 'O')
            confidence = probs[idx][int(pred)] if probs is not None else 1.0
            
            if label == 'B-ASPECT':
                if current_offsets:
                    aspect_info = self._create_aspect_info(text, current_offsets, current_confidences)
                    if aspect_info:
                        aspects.append(aspect_info)
                current_offsets = [offset.tolist()]
                current_confidences = [confidence]
            
            elif label == 'I-ASPECT' and current_offsets:
                current_offsets.append(offset.tolist())
                current_confidences.append(confidence)
            
            elif label == 'O' and current_offsets:
                aspect_info = self._create_aspect_info(text, current_offsets, current_confidences)
                if aspect_info:
                    aspects.append(aspect_info)
                current_offsets = []
                current_confidences = []
        
        if current_offsets:
            aspect_info = self._create_aspect_info(text, current_offsets, current_confidences)
            if aspect_info:
                aspects.append(aspect_info)
        
        return aspects
    
    def _decode_predictions_with_confidence(
        self,
        text: str,
        predictions: np.ndarray,
        offset_mapping: torch.Tensor,
        probs: np.ndarray,
        threshold: float = 0.5
    ) -> List[Dict]:
        """
        Decode predictions dengan confidence filtering
        Hanya keep predictions yang confidence >= threshold
        """
        aspects = []
        current_offsets = []
        current_confidences = []
        
        for idx, (pred, offset, prob) in enumerate(zip(predictions, offset_mapping, probs)):
            if offset[0] == 0 and offset[1] == 0:
                continue
            
            label = self.id2label.get(int(pred), 'O')
            confidence = prob[int(pred)]
            
            if confidence < threshold:
                label = 'O'
            
            if label == 'B-ASPECT':
                if current_offsets:
                    avg_conf = np.mean(current_confidences)
                    if avg_conf >= threshold:
                        aspect_info = self._create_aspect_info(text, current_offsets, current_confidences)
                        if aspect_info:
                            aspects.append(aspect_info)
                current_offsets = [offset.tolist()]
                current_confidences = [confidence]
            
            elif label == 'I-ASPECT' and current_offsets:
                current_offsets.append(offset.tolist())
                current_confidences.append(confidence)
            
            elif label == 'O' and current_offsets:
                avg_conf = np.mean(current_confidences)
                if avg_conf >= threshold:
                    aspect_info = self._create_aspect_info(text, current_offsets, current_confidences)
                    if aspect_info:
                        aspects.append(aspect_info)
                current_offsets = []
                current_confidences = []
        
        if current_offsets:
            avg_conf = np.mean(current_confidences)
            if avg_conf >= threshold:
                aspect_info = self._create_aspect_info(text, current_offsets, current_confidences)
                if aspect_info:
                    aspects.append(aspect_info)
        
        return aspects
    
    def _create_aspect_info(
        self,
        text: str,
        offsets: List[List[int]],
        confidences: List[float]
    ) -> Optional[Dict]:
        """
        Create aspect info dict dari offsets dan confidences
        """
        if not offsets:
            return None
        
        aspect_text = self._reconstruct_text(text, offsets)
        if not aspect_text or len(aspect_text) < 2:
            return None
        
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        start_pos = offsets[0][0]
        end_pos = offsets[-1][1]
        context_start = max(0, start_pos - 30)
        context_end = min(len(text), end_pos + 30)
        context = text[context_start:context_end].strip()
        
        return {
            'text': aspect_text,
            'confidence': float(avg_confidence),
            'context': context,
            'start': start_pos,
            'end': end_pos
        }
    
    def _reconstruct_text(self, original_text: str, offsets: List[List[int]]) -> str:
        """
        Reconstruct text dari token offsets
        Handle subword tokens (##) dan spacing
        """
        if not offsets:
            return ""
        
        start = offsets[0][0]
        end = offsets[-1][1]
        
        if start >= end or start < 0 or end > len(original_text):
            return ""
        
        text = original_text[start:end].strip()
        text = text.replace('##', '')
        text = ' '.join(text.split())
        
        return text
    
    def _extract_context(self, aspect: str, text: str, window: int = 3) -> str:
        """
        Extract context sekitar aspect (±window words)
        Compatible dengan service lama
        """
        text_lower = text.lower()
        aspect_lower = aspect.lower()
        words = text_lower.split()
        
        try:
            aspect_words = aspect_lower.split()
            
            for i in range(len(words) - len(aspect_words) + 1):
                if words[i:i+len(aspect_words)] == aspect_words:
                    start = max(0, i - window)
                    end = min(len(words), i + len(aspect_words) + window)
                    return ' '.join(words[start:end])
            
            if aspect_lower in words:
                idx = words.index(aspect_lower)
                start = max(0, idx - window)
                end = min(len(words), idx + window + 1)
                return ' '.join(words[start:end])
                
        except (ValueError, IndexError):
            pass
        
        return text_lower[:50]
    
    # Penanda batas klausa. Selain tanda baca kalimat, konjungsi pertentangan
    # ikut memotong karena justru di situlah polaritas berbalik:
    # "Kamarnya bersih TAPI wifi lemot".
    _CLAUSE_CONNECTIVES = (
        'tetapi', 'namun', 'sayangnya', 'sedangkan', 'padahal', 'walaupun',
        'meskipun', 'walau', 'meski', 'hanya saja', 'cuma', 'tapi', 'kecuali',
    )

    def _clause_around(self, text: str, start: int, end: int) -> str:
        """
        Ambil klausa yang memuat rentang [start, end) pada teks.

        Sentimen per-aspek harus dihitung dari klausa tempat aspek itu berada,
        bukan dari kalimat penuh. Tanpa ini, "Pelayanannya bagus tapi harganya
        mahal" memberi polaritas yang sama kepada 'pelayanan' dan 'harga' -
        yang menghapus seluruh guna analisis berbasis aspek.

        Args:
            text: Teks asli.
            start: Awal span aspek.
            end: Akhir span aspek.

        Returns:
            Potongan klausa; jatuh kembali ke teks penuh bila terlalu pendek
            untuk menyimpan opini.
        """
        if start is None or end is None or start < 0 or end > len(text):
            return text

        boundaries = {0, len(text)}

        for i, ch in enumerate(text):
            if ch in '.!?;\n':
                boundaries.add(i + 1)

        lowered = text.lower()
        for word in self._CLAUSE_CONNECTIVES:
            pos = 0
            while True:
                idx = lowered.find(word, pos)
                if idx == -1:
                    break
                before_ok = idx == 0 or not lowered[idx - 1].isalnum()
                after = idx + len(word)
                after_ok = after == len(lowered) or not lowered[after].isalnum()
                if before_ok and after_ok:
                    boundaries.add(idx)
                pos = idx + 1

        ordered = sorted(boundaries)
        left = max((b for b in ordered if b <= start), default=0)
        right = min((b for b in ordered if b >= end), default=len(text))

        clause = text[left:right].strip(' ,.;!?\n')

        # Buang konjungsi pertentangan di awal klausa. Ia menandai hubungan
        # antar-klausa, bukan opini terhadap aspeknya, dan membuat model salah
        # baca: "padahal fasilitas umum memadai" terbaca negative, sedangkan
        # "fasilitas umum memadai" positive (0,99).
        # Kata negasi sengaja TIDAK termasuk di sini - membuangnya akan
        # membalik polaritas, persis kesalahan yang dihindari di jalur sentimen.
        lowered_clause = clause.lower()
        for word in self._CLAUSE_CONNECTIVES:
            if lowered_clause.startswith(word + ' '):
                clause = clause[len(word):].lstrip(' ,')
                break

        # Klausa yang isinya hanya aspek itu sendiri tidak memuat opini apa pun;
        # untuk kasus itu teks penuh lebih informatif daripada memaksakan
        # 'neutral'. Ambangnya dibandingkan terhadap panjang aspeknya, bukan
        # angka tetap: "harganya mahal" hanya dua kata tetapi sudah memuat
        # opininya, sedangkan "wifi" sendirian tidak.
        aspect_word_count = len(text[start:end].split())
        if len(clause.split()) <= aspect_word_count:
            return text
        return clause

    async def _resolve_low_confidence(
        self,
        predictions: List[Dict[str, Any]],
        clauses: List[str],
        full_texts: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Ganti prediksi klausa yang tidak meyakinkan dengan prediksi teks penuh.

        Pemotongan klausa membuat polaritas benar-benar per-aspek, tetapi klausa
        yang sangat pendek bisa kehilangan konteks yang dibutuhkan model.
        Terukur: IndoBERT menilai "Pajak naik" positive dengan keyakinan 0,53
        (praktis lemparan koin), sedangkan "Pajak naik terus" negative 0,73.

        Hanya kemunculan yang keyakinannya di bawah ambang DAN klausanya memang
        berbeda dari teks penuh yang dihitung ulang, sehingga biayanya kecil.

        Args:
            predictions: Hasil sentimen atas klausa, sejajar dengan clauses.
            clauses: Klausa yang dinilai.
            full_texts: Teks penuh asal tiap klausa.

        Returns:
            Daftar prediksi dengan panjang dan urutan yang sama.
        """
        threshold = settings.aspect_clause_min_confidence

        retry_positions = [
            i for i, pred in enumerate(predictions)
            if pred.get('confidence', 1.0) < threshold
            and i < len(full_texts)
            and clauses[i] != full_texts[i]
        ]

        if not retry_positions:
            return predictions

        logger.info(
            f"{len(retry_positions)} klausa berkeyakinan < {threshold}; "
            f"memakai teks penuh untuk kemunculan tersebut"
        )

        retry_result = await self.sentiment_service.analyze(
            [full_texts[i] for i in retry_positions]
        )
        retry_predictions = retry_result.get('predictions', [])

        for position, original_index in enumerate(retry_positions):
            if position < len(retry_predictions):
                predictions[original_index] = retry_predictions[position]

        return predictions

    async def _analyze_aspect_sentiments(
        self,
        texts: List[str],
        aspects: List[Dict]
    ) -> List[Dict]:
        """
        Hitung distribusi sentimen untuk setiap aspek.

        Polaritas diambil dari KLAUSA tempat aspek berada, bukan kalimat penuh
        (lihat _clause_around) - tanpa itu semua aspek dalam satu kalimat
        mendapat polaritas yang sama dan analisis berbasis aspek kehilangan
        artinya.

        Seluruh klausa dari semua aspek dikirim dalam SATU panggilan sentimen,
        lalu hasilnya dipotong kembali per aspek. Sebelumnya setiap aspek
        memicu forward pass sendiri; pada 80 komentar dengan 45 aspek itu
        berarti 45 pemanggilan model yang mendominasi waktu analisis.

        Args:
            texts: Daftar teks masukan (dipertahankan demi kompatibilitas).
            aspects: Hasil ekstraksi, satu entri per aspek.

        Returns:
            List of {aspect, count, sentiments, dominant_sentiment, score, variants}.
        """
        # ── Kumpulkan klausa seluruh aspek sekaligus ────────────────────────
        pending: List[Dict[str, Any]] = []
        all_clauses: List[str] = []
        all_full_texts: List[str] = []

        for aspect_data in aspects:
            if 'aspect' not in aspect_data:
                continue

            clauses: List[str] = []
            full_texts: List[str] = []
            if 'occurrences' in aspect_data:
                for occ in aspect_data['occurrences']:
                    clauses.append(
                        self._clause_around(occ['text'], occ.get('start'), occ.get('end'))
                    )
                    full_texts.append(occ['text'])
            elif 'aspects' in aspect_data:
                clauses = [aspect_data.get('text', '')]
                full_texts = list(clauses)

            paired = [(c, f) for c, f in zip(clauses, full_texts) if c and c.strip()]
            if not paired:
                continue
            clauses = [c for c, _ in paired]
            full_texts = [f for _, f in paired]

            pending.append({
                'data': aspect_data,
                'start': len(all_clauses),
                'end': len(all_clauses) + len(clauses),
            })
            all_clauses.extend(clauses)
            all_full_texts.extend(full_texts)

        if not all_clauses:
            return []

        # ── Satu forward pass untuk semuanya ────────────────────────────────
        sentiment_result = await self.sentiment_service.analyze(all_clauses)
        predictions = sentiment_result.get('predictions', [])

        predictions = await self._resolve_low_confidence(
            predictions, all_clauses, all_full_texts
        )

        # ── Potong kembali per aspek ────────────────────────────────────────
        aspect_sentiments: List[Dict[str, Any]] = []

        for item in pending:
            aspect_data = item['data']
            aspect_name = aspect_data['aspect']
            slice_ = predictions[item['start']:item['end']]

            if not slice_:
                continue

            distribution = self.sentiment_service._calculate_distribution(slice_)

            aspect_sentiments.append({
                'aspect': aspect_name,
                'count': len(slice_),
                'sentiments': distribution,
                'dominant_sentiment': max(distribution, key=distribution.get),
                'score': aspect_data.get('score', 0),
                'variants': aspect_data.get('variants', [aspect_name]),
            })

        return aspect_sentiments
    
    def _calculate_statistics(self, aspect_sentiments: List[Dict]) -> Dict:
        """
        Calculate aspect statistics
        Unchanged from original
        """
        if not aspect_sentiments:
            return {
                'total_aspects': 0,
                'most_discussed': None,
                'most_positive': None,
                'most_negative': None
            }
        
        sorted_by_count    = sorted(aspect_sentiments, key=lambda x: x['count'], reverse=True)
        sorted_by_positive = sorted(aspect_sentiments, key=lambda x: x['sentiments'].get('positive', 0), reverse=True)
        sorted_by_negative = sorted(aspect_sentiments, key=lambda x: x['sentiments'].get('negative', 0), reverse=True)
        
        return {
            'total_aspects':  len(aspect_sentiments),
            'most_discussed': sorted_by_count[0]    if sorted_by_count    else None,
            'most_positive':  sorted_by_positive[0] if sorted_by_positive else None,
            'most_negative':  sorted_by_negative[0] if sorted_by_negative else None
        }
    
    def _generate_summary(self, aspect_sentiments: List[Dict]) -> str:
        """
        Generate human-readable summary
        Unchanged from original
        """
        if not aspect_sentiments:
            return "Tidak ditemukan aspek spesifik dalam teks."
        
        most_discussed = max(aspect_sentiments, key=lambda x: x['count'])
        
        variants_info = ""
        if 'variants' in most_discussed and len(most_discussed['variants']) > 1:
            variants_info = f" (termasuk variasi: {', '.join(most_discussed['variants'][:3])})"
        
        mode_info = "dengan model IndoBERT-NER" if self.model_trained else "secara otomatis"
        
        return (
            f"Ditemukan {len(aspect_sentiments)} aspek utama {mode_info}. "
            f"Aspek yang paling banyak dibahas adalah '{most_discussed['aspect']}'"
            f"{variants_info} dengan {most_discussed['count']} mentions, "
            f"cenderung {most_discussed['dominant_sentiment']}."
        )
    
    def get_uncertainty_scores(self, texts: List[str]) -> List[float]:
        """
        Calculate uncertainty scores untuk active learning.
        Seluruh texts diproses dalam chunk.

        Returns:
            List of uncertainty scores (0-1, higher = more uncertain)
        """
        self._ensure_loaded()
        
        uncertainties = []
        max_chunk_size = 512
        max_length = 128

        for i in range(0, len(texts), max_chunk_size):
            chunk_texts = texts[i:i + max_chunk_size]
            encoding = self.tokenizer(
                chunk_texts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=max_length
            )

            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                
                top2_probs, _ = torch.topk(probs, k=2, dim=-1)
                margin = top2_probs[:, :, 0] - top2_probs[:, :, 1]
                avg_margin = margin.mean(dim=-1)
                
                chunk_uncertainties = (1 - avg_margin).tolist()
                uncertainties.extend(chunk_uncertainties)
                
            del input_ids, attention_mask, logits, probs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return uncertainties
    
    def save_model(self, save_path: str, **kwargs):
        """
        Save trained model to disk
        """
        self._ensure_loaded()
        if self.model is None:
            raise ValueError("Model is not loaded")
        try:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'id2label': self.id2label,
                'label2id': self.label2id,
                **kwargs
            }, save_path)
            logger.info(f"✅ Model saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise
    
    def train_mode(self):
        """Set model to training mode"""
        if self.model is not None:
            self.model.train()
    
    def eval_mode(self):
        """Set model to evaluation mode"""
        if self.model is not None:
            self.model.eval()

    def _stem(self, word: str) -> str:
        """Stem satu kata dengan cache — Sastrawi cukup lambat untuk dipanggil berulang."""
        key = word.lower()
        if key not in self._stem_cache:
            self._stem_cache[key] = self._stemmer.stem(key)
        return self._stem_cache[key]

    def _word_spans(self, text: str) -> List[Tuple[int, int, str]]:
        """Pecah teks menjadi (start, end, kata) memakai batas kata."""
        return [(m.start(), m.end(), m.group()) for m in re.finditer(r'\w+', text)]

    def _find_spans_by_stem(
        self,
        text: str,
        aspect: str
    ) -> List[Tuple[int, int]]:
        """
        Cari aspek lewat pencocokan bentuk dasar, untuk menangkap imbuhan.

        Bahasa Indonesia aglutinatif: koreksi 'pejabat' tidak cocok literal dengan
        'pejabatnya', dan 'pelayanan' tidak cocok dengan 'layanannya'. Tanpa ini
        sampel tersebut berlabel seluruhnya 'O' dan mengajarkan model bahwa
        kalimatnya tidak beraspek.

        Hanya dipakai pada ekstraksi aspek. Jalur sentimen sengaja tidak memakai
        stemming sama sekali (lihat SentimentService._sanitize_config).

        Args:
            text: Teks mentah.
            aspect: Nama aspek hasil koreksi user.

        Returns:
            Daftar span (start, end) pada teks asli.
        """
        aspect_words = [w for w in re.findall(r'\w+', aspect.lower()) if w]
        if not aspect_words:
            return []

        aspect_stems = [self._stem(w) for w in aspect_words]
        words = self._word_spans(text)
        n = len(aspect_stems)

        spans: List[Tuple[int, int]] = []
        for i in range(len(words) - n + 1):
            window = words[i:i + n]
            if [self._stem(w[2]) for w in window] != aspect_stems:
                continue
            # Tolak turunan verbal: 'menghargai' berakar sama dengan 'harga',
            # tapi ia predikat, bukan target opini. Menandainya sebagai aspek
            # justru mengajari model memberi label pada verba.
            if any(
                self._is_verbal_derivation(w[2], a)
                for w, a in zip(window, aspect_words)
            ):
                continue
            spans.append((window[0][0], window[-1][1]))

        return spans

    @staticmethod
    def _is_verbal_derivation(surface: str, aspect_word: str) -> bool:
        """
        True bila kata di teks memakai awalan pembentuk verba yang tidak dimiliki
        aspeknya (me-, di-, ter-, ber-).

        Awalan nominal seperti pe- dan ke- sengaja tidak masuk daftar, sehingga
        'pejabatnya' dan 'kebersihan' tetap dianggap cocok.
        """
        verbal_prefixes = ('me', 'di', 'ter', 'ber')
        s, a = surface.lower(), aspect_word.lower()

        for prefix in verbal_prefixes:
            if s.startswith(prefix) and not a.startswith(prefix):
                return True
        return False

    def _find_aspect_spans(
        self,
        text: str,
        aspects: List[str],
        use_stemming: bool = True
    ) -> Tuple[List[Tuple[int, int]], List[str]]:
        """
        Cari posisi kemunculan tiap aspek di dalam teks (whole-word match).

        Aspek yang tidak ditemukan dikembalikan terpisah supaya pemanggil bisa
        melaporkannya. Ini penting: kalau koreksi user berbunyi "pelayanan"
        sementara teksnya menulis "layanannya", pencocokan gagal dan sampel itu
        berlabel semua-O — model justru diajari bahwa kalimat tersebut tidak
        punya aspek sama sekali.

        Args:
            text: Teks mentah.
            aspects: Daftar nama aspek hasil koreksi user.

        Returns:
            Tuple (spans, unmatched) dengan spans berupa pasangan (start, end).
        """
        text_lower = text.lower()
        spans: List[Tuple[int, int]] = []
        unmatched: List[str] = []

        # Aspek terpanjang lebih dulu agar tumpang tindih ditangani benar
        for aspect in sorted(aspects, key=len, reverse=True):
            aspect_lower = self._normalize_aspect_text(aspect)
            if not aspect_lower:
                continue

            found = False
            start = 0
            while True:
                idx = text_lower.find(aspect_lower, start)
                if idx == -1:
                    break

                # Check word boundaries
                is_start_word = (idx == 0 or not text_lower[idx - 1].isalnum())
                end_idx = idx + len(aspect_lower)
                is_end_word = (end_idx == len(text_lower) or not text_lower[end_idx].isalnum())

                if is_start_word and is_end_word:
                    spans.append((idx, end_idx))
                    found = True

                start = idx + 1

            # Pass 2: cocokkan lewat bentuk dasar bila literal gagal.
            if not found and use_stemming:
                stem_spans = self._find_spans_by_stem(text, aspect_lower)
                if stem_spans:
                    spans.extend(stem_spans)
                    found = True

            if not found:
                unmatched.append(aspect)

        return spans, unmatched

    def _generate_bio_tags(
        self,
        text: str,
        aspects: List[str],
        char_spans: Optional[List[Tuple[int, int]]] = None
    ) -> List[int]:
        """
        Auto-generate BIO tags dari pasangan (text, aspects).

        Args:
            text: Teks mentah.
            aspects: Daftar nama aspek; dipakai bila char_spans tidak diberikan.
            char_spans: Offset (start, end) yang sudah pasti. Dipakai untuk data
                berlabel manusia seperti IndoNLU TermA, di mana penurunan ulang
                lewat pencocokan string akan menandai ~5% kemunculan yang justru
                sengaja dibiarkan 'O' oleh anotator.

        Returns:
            List label_id sejajar dengan token IndoBERT.
        """
        encoding = self.tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=128
        )
        offsets = encoding['offset_mapping']
        input_ids = encoding['input_ids']

        labels = [self.label2id['O']] * len(input_ids)

        if char_spans is not None:
            spans = list(char_spans)
        else:
            spans, _ = self._find_aspect_spans(text, aspects)

        # Map spans to tokens
        for span_start, span_end in spans:
            in_span = False
            for i, (off_start, off_end) in enumerate(offsets):
                if off_start == 0 and off_end == 0:
                    labels[i] = -100
                    continue
                    
                if off_start >= span_start and off_end <= span_end:
                    if not in_span:
                        labels[i] = self.label2id['B-ASPECT']
                        in_span = True
                    else:
                        labels[i] = self.label2id['I-ASPECT']
                elif off_start >= span_end:
                    break
                    
        for i, (off_start, off_end) in enumerate(offsets):
            if off_start == 0 and off_end == 0:
                labels[i] = -100
                
        return labels

    # ── Retraining (active learning) ────────────────────────────────────────

    def _prepare_bio_dataset(
        self,
        training_data: List[Dict[str, Any]],
        max_length: int = 128
    ) -> Tuple[List[Any], List[Any], List[List[int]], Dict[str, Any]]:
        """
        Ubah data koreksi menjadi tensor input + label BIO, sekaligus mengumpulkan
        diagnostik kualitas pelabelan.

        Args:
            training_data: List of {text, aspects: [...]}.
            max_length: Panjang token; harus sama dengan yang dipakai inference.

        Returns:
            Tuple (input_ids, attention_masks, labels, report) di mana report
            memuat sampel yang dilewati dan aspek yang gagal dicocokkan.
        """
        all_input_ids: List[Any] = []
        all_attention_masks: List[Any] = []
        all_labels: List[List[int]] = []

        skipped_empty = 0
        unmatched_samples: List[Dict[str, Any]] = []
        all_o_samples = 0
        total_aspects = 0
        total_unmatched = 0

        gold_span_items = 0

        for item in training_data:
            text = (item.get('text') or '').strip()
            aspects = [a for a in (item.get('aspects') or []) if str(a).strip()]

            if not text:
                skipped_empty += 1
                continue

            # Data berlabel manusia membawa offset karakter; pakai apa adanya
            # agar label anotator tidak dirusak pencocokan string.
            raw_spans = item.get('spans')
            char_spans = None
            if raw_spans:
                char_spans = [
                    (s['start'], s['end']) for s in raw_spans
                    if isinstance(s, dict) and 'start' in s and 'end' in s
                ]
                gold_span_items += 1
                total_aspects += len(char_spans)
                labels = self._generate_bio_tags(text, aspects, char_spans=char_spans)
            else:
                total_aspects += len(aspects)
                _, unmatched = self._find_aspect_spans(text, aspects)
                total_unmatched += len(unmatched)
                labels = self._generate_bio_tags(text, aspects)

            # Sampel dengan aspek terkoreksi tetapi tidak satu pun cocok akan
            # mengajarkan hal yang salah: seluruh tokennya berlabel 'O'.
            unmatched = [] if raw_spans else unmatched
            if aspects and unmatched:
                if len(unmatched_samples) < 50:
                    unmatched_samples.append({
                        'text': text[:120],
                        'unmatched_aspects': unmatched,
                    })
                if len(unmatched) == len(aspects):
                    all_o_samples += 1

            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=max_length,
                return_tensors='pt'
            )

            padded_labels = labels + [-100] * (max_length - len(labels))
            padded_labels = padded_labels[:max_length]

            all_input_ids.append(encoding['input_ids'][0])
            all_attention_masks.append(encoding['attention_mask'][0])
            all_labels.append(padded_labels)

        report = {
            'samples_received': len(training_data),
            'samples_used': len(all_input_ids),
            'samples_with_gold_spans': gold_span_items,
            'samples_skipped_empty': skipped_empty,
            'aspects_total': total_aspects,
            'aspects_unmatched': total_unmatched,
            'aspects_unmatched_pct': (
                round((total_unmatched / total_aspects) * 100, 2) if total_aspects else 0.0
            ),
            'samples_all_O': all_o_samples,
            'unmatched_examples': unmatched_samples[:20],
        }

        return all_input_ids, all_attention_masks, all_labels, report

    def _evaluate_bio(self, input_ids, attention_masks, labels, batch_size: int = 16) -> Dict[str, Any]:
        """Ukur F1 token aspek pada satu set; dipakai sebelum & sesudah training."""
        from torch.utils.data import DataLoader, TensorDataset

        self.model.eval()
        loader = DataLoader(
            TensorDataset(input_ids, attention_masks, labels), batch_size=batch_size
        )

        y_true: List[List[int]] = []
        y_pred: List[List[int]] = []

        with torch.no_grad():
            for batch in loader:
                b_ids = batch[0].to(self.device)
                b_mask = batch[1].to(self.device)
                outputs = self.model(input_ids=b_ids, attention_mask=b_mask)
                preds = torch.argmax(outputs.logits, dim=-1).cpu().tolist()

                y_pred.extend(preds)
                y_true.extend(batch[2].cpu().tolist())

                del b_ids, b_mask, outputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        return token_f1(y_true, y_pred, ignore_index=-100, positive_labels=(1, 2))

    async def retrain(
        self,
        training_data: List[Dict[str, Any]],
        epochs: int = 3,
        learning_rate: float = 2e-5,
        save_path: Optional[str] = None,
        force: bool = False,
        eval_data: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = 16,
        warmup_ratio: float = 0.1,
        progress_callback: Optional[Any] = None,
        use_class_weights: bool = False
    ) -> Dict[str, Any]:
        """
        Pembungkus async tipis untuk `_retrain_sync`.

        Alasannya sama persis dengan `SentimentService.retrain`: loop pelatihan
        PyTorch sepenuhnya sinkron dan berjalan menit-menit, sehingga bila
        dieksekusi di dalam event loop maka SETIAP permintaan lain menggantung
        sampai selesai - termasuk `/health` dan `/api/analyze/*`. Sejak sisi web
        memicu retraining otomatis dari koreksi pengguna, waktunya tidak lagi
        dipilih manusia.

        `progress_callback` kini dipanggil dari thread pekerja. Semua pemakainya
        sinkron (`scripts/train_aspect.py`), jadi aman; jangan mengoper coroutine
        ke sini.

        `lepas_memori` di `finally` - alasannya di `SentimentService.retrain`.
        """
        try:
            return await asyncio.to_thread(
                self._retrain_sync,
                training_data=training_data,
                epochs=epochs,
                learning_rate=learning_rate,
                save_path=save_path,
                force=force,
                eval_data=eval_data,
                batch_size=batch_size,
                warmup_ratio=warmup_ratio,
                progress_callback=progress_callback,
                use_class_weights=use_class_weights,
            )
        finally:
            lepas_memori()

    def _retrain_sync(
        self,
        training_data: List[Dict[str, Any]],
        epochs: int = 3,
        learning_rate: float = 2e-5,
        save_path: Optional[str] = None,
        force: bool = False,
        eval_data: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = 16,
        warmup_ratio: float = 0.1,
        progress_callback: Optional[Any] = None,
        use_class_weights: bool = False
    ) -> Dict[str, Any]:
        """
        Retrain aspect extraction model dengan data berlabel.

        Seed dikunci, metrik diukur sebelum dan sesudah pada set validasi yang
        sama, bobot epoch terbaik yang dipakai, dan checkpoint hanya ditulis bila
        F1 token aspek tidak turun.

        Args:
            training_data: List of {text, aspects: [...]}; kunci opsional 'spans'
                berisi offset anotator yang dipakai apa adanya.
            epochs: Jumlah epoch fine-tuning.
            learning_rate: Learning rate AdamW.
            save_path: Lokasi checkpoint; default settings.aspect_checkpoint_path.
            force: Simpan walaupun metrik menurun.
            eval_data: Set validasi terpisah. Bila diberikan, seluruh
                training_data dipakai untuk melatih dan pembagian acak
                dilewati - dipakai untuk dataset yang punya split resmi
                (mis. IndoNLU TermA) agar angkanya sebanding dengan literatur.
            batch_size: Ukuran batch pelatihan.
            warmup_ratio: Porsi langkah untuk warmup linear pada scheduler.
            progress_callback: Dipanggil dengan dict kemajuan setiap beberapa
                batch. Murni pelaporan, tidak mengubah hasil pelatihan.
            use_class_weights: Bobot kelas berbanding terbalik dengan frekuensi
                pada loss. 87% token berlabel 'O', sehingga tanpa pembobotan
                model condong memprediksi 'O' dan recall aspek tertekan.
                Default False agar perilaku lama tidak berubah.

        Returns:
            Dict berisi metrik sebelum/sesudah, riwayat per epoch, dan laporan
            kualitas pelabelan BIO.
        """
        from torch.utils.data import DataLoader, TensorDataset
        from torch.optim import AdamW
        from transformers import get_linear_schedule_with_warmup

        save_path = save_path or settings.aspect_checkpoint_path
        set_seed(settings.retrain_seed)

        self._ensure_loaded()
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model aspek tidak berhasil dimuat")

        # Protokol kolam - lihat `settings.retrain_from_baseline`. Dasar modul
        # aspek BUKAN IndoBERT mentah melainkan checkpoint sebelum active
        # learning dimulai (K6); memakai yang mentah membuat titik awalnya
        # sistem yang tidak pernah dipakai siapa pun.
        dari_dasar = False
        dasar = jalur_bobot_dasar_aspek()
        sumber_awal = self._model_path

        if settings.retrain_from_baseline:
            if dasar and os.path.exists(dasar):
                logger.info(f"Protokol kolam: melatih dari bobot dasar {dasar}")
                self._initialize_model(dasar)
                dari_dasar = True
            else:
                logger.warning(
                    "retrain_from_baseline aktif tetapi aspect_baseline_checkpoint "
                    f"tidak ditemukan ({dasar or 'belum disetel'}); pelatihan "
                    "MELANJUTKAN checkpoint sekarang. Kurva iterasi jadi "
                    "bergantung pada urutan pelatihan, bukan hanya isi kolam."
                )

        logger.info(f"Starting aspect model retrain with {len(training_data)} samples")

        all_input_ids, all_attention_masks, all_labels, bio_report = \
            self._prepare_bio_dataset(training_data)

        if not all_input_ids:
            raise ValueError("No valid training data after processing")

        if bio_report['aspects_unmatched']:
            logger.warning(
                f"{bio_report['aspects_unmatched']} dari {bio_report['aspects_total']} aspek "
                f"({bio_report['aspects_unmatched_pct']}%) tidak ditemukan persis di teksnya; "
                f"{bio_report['samples_all_O']} sampel menjadi seluruhnya 'O'."
            )

        input_ids = torch.stack(all_input_ids)
        attention_masks = torch.stack(all_attention_masks)
        labels = torch.tensor(all_labels)

        if eval_data:
            # Split resmi dipakai apa adanya; seluruh training_data untuk melatih.
            ev_ids, ev_masks, ev_labels, _ = self._prepare_bio_dataset(eval_data)
            if not ev_ids:
                raise ValueError("eval_data tidak menghasilkan sampel valid")

            train_ds = TensorDataset(input_ids, attention_masks, labels)
            val_input_ids = torch.stack(ev_ids)
            val_masks = torch.stack(ev_masks)
            val_labels = torch.tensor(ev_labels)
            n_train, n_val = len(all_input_ids), len(ev_ids)
            split_kind = 'official'
        else:
            # Stratifikasi berdasarkan ada/tidaknya token aspek, supaya set validasi
            # tidak berisi kalimat tanpa aspek semua (F1 aspek akan jadi nol semu).
            has_aspect = [
                1 if any(l in (1, 2) for l in row) else 0 for row in all_labels
            ]
            train_idx, val_idx = stratified_split(
                has_aspect, settings.retrain_val_ratio, settings.retrain_seed
            )
            if not val_idx:
                raise ValueError("Set validasi kosong - tambahkan data koreksi terlebih dahulu")

            train_ds = TensorDataset(
                input_ids[train_idx], attention_masks[train_idx], labels[train_idx]
            )
            val_input_ids = input_ids[val_idx]
            val_masks = attention_masks[val_idx]
            val_labels = labels[val_idx]
            n_train, n_val = len(train_idx), len(val_idx)
            split_kind = 'stratified_80_20'

        metrics_before = self._evaluate_bio(val_input_ids, val_masks, val_labels)
        logger.info(f"Metrik sebelum retraining (F1 token aspek): {metrics_before['f1']}")

        loss_fn = None
        class_weights: List[float] = []
        if use_class_weights:
            flat = [l for row in all_labels for l in row if l != -100]
            class_weights = compute_class_weights(flat, self.num_labels)
            loss_fn = torch.nn.CrossEntropyLoss(
                weight=torch.tensor(class_weights, dtype=torch.float).to(self.device),
                ignore_index=-100,
            )
            logger.info(f"Bobot kelas (O, B-ASPECT, I-ASPECT): {class_weights}")

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)

        # Warmup linear lalu peluruhan: standar fine-tuning BERT, menstabilkan
        # langkah awal ketika head klasifikasi masih acak.
        total_steps = max(1, len(train_loader) * epochs)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * warmup_ratio),
            num_training_steps=total_steps,
        )

        train_losses: List[float] = []
        epoch_metrics_history: List[Dict[str, Any]] = []

        # Bobot epoch terbaik disimpan ke disk, bukan disalin di RAM: state_dict
        # IndoBERT ~440MB dan service ini berjalan dengan anggaran memori ketat.
        best_f1 = -1.0
        best_epoch = 0
        best_state_path = f'{save_path}.best.tmp'

        for epoch in range(epochs):
            self.model.train()
            total_train_loss = 0.0

            for step, batch in enumerate(train_loader, start=1):
                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_labels = batch[2].to(self.device)

                self.model.zero_grad()

                outputs = self.model(
                    input_ids=b_input_ids,
                    attention_mask=b_input_mask,
                    labels=None if loss_fn else b_labels
                )

                if loss_fn:
                    loss = loss_fn(
                        outputs.logits.view(-1, self.num_labels), b_labels.view(-1)
                    )
                else:
                    loss = outputs.loss
                total_train_loss += loss.item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                del b_input_ids, b_input_mask, b_labels, outputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if progress_callback and (step % 10 == 0 or step == len(train_loader)):
                    progress_callback({
                        'phase': 'train', 'epoch': epoch + 1, 'epochs': epochs,
                        'step': step, 'steps': len(train_loader),
                        'running_loss': round(total_train_loss / step, 4),
                    })

            avg_train_loss = total_train_loss / len(train_loader) if len(train_loader) else 0.0
            train_losses.append(round(avg_train_loss, 4))

            if progress_callback:
                progress_callback({'phase': 'eval', 'epoch': epoch + 1, 'epochs': epochs})

            epoch_metrics = self._evaluate_bio(val_input_ids, val_masks, val_labels)
            epoch_metrics_history.append({'epoch': epoch + 1,
                                          'train_loss': round(avg_train_loss, 4),
                                          **epoch_metrics})
            logger.info(
                f"Epoch {epoch + 1}/{epochs} - Train loss: {avg_train_loss:.4f} "
                f"- Val F1 aspek: {epoch_metrics['f1']:.4f}"
            )
            if progress_callback:
                progress_callback({'phase': 'epoch_done', 'epoch': epoch + 1,
                                   'epochs': epochs, 'train_loss': round(avg_train_loss, 4),
                                   **epoch_metrics})

            # Epoch terakhir belum tentu terbaik: setelah beberapa epoch model
            # mulai overfit dan F1 validasi turun. Simpan yang terbaik.
            if epoch_metrics['f1'] > best_f1:
                best_f1 = epoch_metrics['f1']
                best_epoch = epoch + 1
                os.makedirs(os.path.dirname(best_state_path) or '.', exist_ok=True)
                torch.save(self.model.state_dict(), best_state_path)

        # Kembalikan bobot epoch terbaik sebelum evaluasi akhir.
        if best_epoch and best_epoch != epochs and os.path.exists(best_state_path):
            logger.info(f"Memulihkan bobot epoch {best_epoch} (F1 val {best_f1:.4f})")
            self.model.load_state_dict(torch.load(best_state_path, map_location=self.device))

        metrics_after = self._evaluate_bio(val_input_ids, val_masks, val_labels)

        improved, delta = should_accept_checkpoint(
            metrics_before['f1'],
            metrics_after['f1'],
            settings.retrain_min_delta,
            force,
        )
        saved = False

        if improved:
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            self.save_model(
                save_path,
                training_metrics={
                    'train_losses': train_losses,
                    'metrics_before': metrics_before,
                    'metrics_after': metrics_after,
                }
            )
            self._load_trained_model(save_path)
            self._model_path = save_path
            saved = True
            write_provenance(save_path, {
                'modul': 'aspect',
                'bobot_awal': dasar if dari_dasar else (sumber_awal or 'model dasar IndoBERT'),
                'trained_from_baseline': dari_dasar,
                'jumlah_sampel': len(training_data),
                'sidik_jari_kolam': pool_fingerprint([d['text'] for d in training_data]),
                'val_size': n_val,
                'split': split_kind,
                'epochs': epochs,
                'f1_token_sebelum': metrics_before['f1'],
                'f1_token_sesudah': metrics_after['f1'],
                'bio_report': bio_report,
                'seed': settings.retrain_seed,
            })
            logger.info(f"Checkpoint aspek disimpan ke {save_path} (delta F1 {delta:+})")

        if os.path.exists(best_state_path):
            os.remove(best_state_path)

        if not improved:
            # Buang bobot hasil training dengan memuat ulang sumber sebelumnya.
            logger.warning(
                f"Retraining menurunkan F1 token aspek ({delta:+}); checkpoint TIDAK disimpan."
            )
            self._loaded = False
            self._ensure_loaded()

        return {
            'epochs_completed': epochs,
            'train_loss': train_losses,
            'val_loss': [],
            'model_path': save_path,
            'saved': saved,
            'rejected_for_regression': not saved,
            'seed': settings.retrain_seed,
            'train_size': n_train,
            'val_size': n_val,
            'split': split_kind,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'class_weights': class_weights,
            'best_epoch': best_epoch,
            'epoch_history': epoch_metrics_history,
            'metrics_before': metrics_before,
            'metrics_after': metrics_after,
            'f1_delta': delta,
            'bio_report': bio_report,
        }
