import re
from typing import List, Optional, Dict

# URUTAN IMPOR BERSIFAT FUNGSIONAL - JANGAN DIUBAH.
#
# Pada Windows, mengimpor nltk sebelum torch memicu
# `OSError WinError 1114` saat memuat c10.dll (konflik runtime OpenMP/MKL).
#
# Sebelumnya modul ini mengandalkan pemanggilnya untuk mengimpor torch lebih
# dulu, dan itu bertahan hanya karena main.py kebetulan memuat service
# ber-torch duluan. Begitu sebuah tes mengimpor text_cleaner sendirian, seluruh
# koleksi tes gagal. Perbaikan yang sama sudah dilakukan pada topic_service.py.
# Modul ini kini menjamin urutannya sendiri.
try:
    import torch  # noqa: F401
except ImportError:  # pragma: no cover - torch selalu ada di requirements
    pass

import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

from app.utils.logger import setup_logger

logger = setup_logger(__name__)


def ensure_nltk_stopwords() -> bool:
    """
    Pastikan korpus stopwords NLTK tersedia, unduh sekali bila belum ada.

    Sebelumnya baris nltk.download() hanya berupa komentar, sehingga instalasi
    baru (termasuk container Railway) diam-diam kehilangan stopwords NLTK:
    blok except mengembalikan set kosong tanpa peringatan apa pun.

    Returns:
        True bila korpus siap dipakai.
    """
    try:
        nltk.data.find('corpora/stopwords')
        return True
    except LookupError:
        try:
            nltk.download('stopwords', quiet=True)
            nltk.data.find('corpora/stopwords')
            return True
        except Exception:
            return False


# Diunduh sekali saat modul dimuat, bukan per instance TextCleaner.
NLTK_STOPWORDS_AVAILABLE = ensure_nltk_stopwords()


def _load_root_words():
    """
    Muat 29.933 kata dasar bawaan Sastrawi, sekali untuk seluruh proses.

    Dipakai memutuskan apakah huruf berulang dipangkas menjadi dua atau satu
    huruf ("maaaaf" -> "maaf", bukan "maf"), sehingga keputusannya berdasar
    kamus alih-alih tebakan.
    """
    try:
        return frozenset(StemmerFactory().get_words())
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Kamus kata dasar Sastrawi tidak terbaca ({exc})")
        return frozenset()


ROOT_WORDS = _load_root_words()


# Kata negasi & pembalik polaritas. Sastrawi/NLTK memasukkan semuanya ke daftar
# stopword, sehingga "tidak bagus" menjadi "bagus" dan sentimennya terbalik.
# Diverifikasi: 5 dari 6 kalimat bernegasi berubah negatif -> positif (conf ~0.99)
# ketika kata-kata ini dibuang. Karena itu ia dikecualikan secara default.
NEGATION_WORDS = {
    'tidak', 'tak', 'tdk', 'bukan', 'bkn', 'jangan', 'jgn', 'belum', 'blm',
    'kurang', 'krg', 'tanpa', 'gak', 'nggak', 'ga', 'gk', 'enggak', 'engga',
    'kagak', 'ndak', 'never', 'no', 'not',
}

# ── Profil preprocessing per tugas ──────────────────────────────────────────
#
# Kebijakan pembersihan BERBEDA untuk tiap modul analisis, dan perbedaannya
# disengaja serta terukur. Sebelumnya aturan itu tersebar: sentimen memaksanya
# di dalam service-nya sendiri, aspek melewatinya diam-diam, topik memakai apa
# adanya. Menyatukannya di sini membuat satu modul tidak bisa mengubah perilaku
# modul lain tanpa terlihat.
#
# Terapkan dengan `TextCleaner.for_task('transformer', config_pengguna)`.
PROFILES = {
    # Sentimen (IndoBERT). Model dilatih pada teks alami dengan tokenisasi
    # subword, sehingga morfologi dan kata fungsi JUSTRU sinyal. Terukur:
    # 5 dari 6 kalimat bernegasi berbalik negative -> positive pada ~0,99 saat
    # stemming aktif ("pelayanannya tidak bagus sama sekali" -> "layan bagus").
    'transformer': {
        'stemming': False,
        'remove_stopwords': False,
        'lemmatization': False,
        # Angka dan kata sependek "di" ikut bermakna bagi model.
        'min_token_length': 1,
        # "sangat sangat bagus" adalah penekanan, bukan duplikasi yang perlu
        # dirapikan.
        'collapse_repeated_words': False,
    },

    # Topic modeling. Model bag-of-words justru diuntungkan pembersihan agresif;
    # inilah satu-satunya profil yang memakai stemming dan stopword removal.
    'bag_of_words': {
        'stemming': True,
        'remove_stopwords': True,
        'min_token_length': 2,
        'collapse_repeated_words': True,
    },

    # Ekstraksi aspek. Ekstraksi berbasis SPAN, sehingga offset karakter harus
    # menunjuk teks asli - pembersihan apa pun akan menggeser seluruh offset.
    # Profil ini sengaja tidak mengubah apa pun; AspectService memang tidak
    # memanggil TextCleaner sama sekali, dan entri ini merekam alasannya.
    'span': None,
}


class TextCleaner:
    """
    Pembersihan teks media sosial berbahasa Indonesia.

    Sumber kebenaran tunggal untuk seluruh modul analisis; service tidak boleh
    mengimplementasikan pembersihannya sendiri. Kebijakan per-tugas BERBEDA dan
    perbedaannya disengaja - lihat PROFILES di bawah kelas ini.
    """

    # Tanda baca yang dipertahankan saat remove_punctuation=False. Terbatas pada
    # tanda batas kalimat dan penekanan, karena itu yang dipakai model
    # transformer maupun pemecah klausa pada modul aspek.
    KEPT_PUNCTUATION = ".,!?"

    @classmethod
    def for_task(cls, task: str, config=None) -> Optional['TextCleaner']:
        """
        Bangun cleaner dengan kebijakan yang benar untuk satu tugas analisis.

        Konfigurasi pengguna tetap dihormati untuk hal yang tidak berdampak pada
        kebenaran (case folding, tanda baca, normalisasi slang), tetapi kunci
        pada PROFILES SELALU menang. Itulah yang menjamin permintaan pengguna
        tidak bisa merusak modul yang tidak menoleransinya.

        Args:
            task: Salah satu kunci PROFILES.
            config: Konfigurasi dari pemanggil (dict atau model pydantic).

        Returns:
            TextCleaner siap pakai, atau None bila profilnya memang "jangan
            bersihkan apa pun" (tugas berbasis span).

        Raises:
            ValueError: Bila nama profil tidak dikenal.
        """
        if task not in PROFILES:
            raise ValueError(
                f"Profil preprocessing '{task}' tidak dikenal; "
                f"pilih salah satu dari {sorted(PROFILES)}"
            )

        profile = PROFILES[task]
        if profile is None:
            return None

        merged = dict(cls._normalize_config(config))
        ditolak = [k for k, v in profile.items() if k in merged and merged[k] != v]
        if ditolak:
            logger.info(
                f"Profil '{task}' menimpa {ditolak} dari konfigurasi pemanggil: "
                f"opsi itu merusak hasil untuk tugas ini"
            )
        merged.update(profile)
        return cls(merged)

    @staticmethod
    def _normalize_config(config) -> Dict:
        """
        Terima dict, model Pydantic (v1 .dict() / v2 .model_dump()), atau None.

        Args:
            config: Konfigurasi preprocessing dalam bentuk apa pun di atas.

        Returns:
            Dict konfigurasi; kosong bila config tidak diberikan.
        """
        if not config:
            return {}
        if isinstance(config, dict):
            return config
        for attr in ('model_dump', 'dict'):
            if hasattr(config, attr):
                return getattr(config, attr)()
        return {}


    def __init__(self, config=None):
        # Pemanggil mengirim dict (dari service) maupun objek PreprocessingConfig
        # (dari endpoint /api/preprocess). Sebelumnya bentuk kedua membuat
        # endpoint itu selalu gagal: BaseModel tidak punya .get().
        self.config = self._normalize_config(config)


        # Default configuration
        self.case_folding = self.config.get('case_folding', True)
        self.remove_punctuation = self.config.get('remove_punctuation', True)
        self.remove_numbers = self.config.get('remove_numbers', False)
        self.remove_stopwords = self.config.get('remove_stopwords', True)
        self.stemming = self.config.get('stemming', True)
        # `lemmatization` diterima demi kompatibilitas skema API tetapi TIDAK
        # diimplementasikan: tidak ada lemmatizer bahasa Indonesia yang mapan,
        # dan Sastrawi adalah stemmer (memotong imbuhan), bukan lemmatizer
        # (memetakan ke bentuk kamus). Dulu nilai ini dibaca lalu diabaikan
        # tanpa jejak, sehingga pemanggil mengira lemmatisasi berjalan.
        self.lemmatization = self.config.get('lemmatization', False)
        if self.lemmatization:
            logger.warning(
                "lemmatization=True diabaikan: tidak diimplementasikan. "
                "Pakai stemming=True (Sastrawi) bila ingin pelucutan imbuhan."
            )

        # Token lebih pendek dari ini dibuang. 2 cocok untuk model bag-of-words
        # (topik), tetapi jalur transformer memakai 1 supaya angka dan kata
        # sependek "di" tidak lenyap - dulu "a" menjadi string kosong.
        self.min_token_length = self.config.get('min_token_length', 2)

        # Penggabungan kata identik berurutan. Berguna untuk bag-of-words,
        # tetapi pada jalur sentimen ia menghapus penekanan: "sangat sangat
        # bagus" menjadi "sangat bagus".
        self.collapse_repeated_words = self.config.get('collapse_repeated_words', True)
        # Diberi awalan `enable_` karena `fix_typos` dan `normalize_slang` juga
        # nama METODE pada kelas ini; atribut bernama sama akan menimpanya.
        self.enable_fix_typos = self.config.get('fix_typos', True)
        self.enable_normalize_slang = self.config.get('normalize_slang', True)
        # Kata negasi dipertahankan walau stopword removal aktif; lihat NEGATION_WORDS.
        self.protect_negation = self.config.get('protect_negation', True)

        
        # Initialize Sastrawi Stemmer
        if self.stemming:
            factory = StemmerFactory()
            self.stemmer = factory.create_stemmer()
        
        # Initialize Sastrawi StopWords
        if self.remove_stopwords:
            stop_factory = StopWordRemoverFactory()
            # Dapatkan stopwords dari Sastrawi
            self.sastrawi_stopwords = stop_factory.get_stop_words()
            
            # Tambahkan NLTK Indonesian stopwords
            try:
                self.nltk_stopwords = set(stopwords.words('indonesian'))
            except Exception as e:
                # Sastrawi tetap menyediakan stopwords, tapi cakupannya berbeda —
                # kegagalan di sini harus terlihat, bukan hilang diam-diam.
                logger.warning(f"NLTK stopwords tidak tersedia ({e}); memakai Sastrawi saja")
                self.nltk_stopwords = set()
            
            # Gabungkan semua stopwords
            self.all_stopwords = set(self.sastrawi_stopwords) | self.nltk_stopwords
            
            # Tambahkan custom stopwords jika ada
            custom_stopwords = self.config.get('custom_stopwords', [])
            self.all_stopwords.update(custom_stopwords)
            
            # Tambahkan stopwords sosial media
            self.all_stopwords.update(self._get_social_media_stopwords())

            # Buang kata negasi dari daftar stopword agar polaritas kalimat utuh.
            if self.protect_negation:
                self.all_stopwords -= NEGATION_WORDS
        
        # Load kamus normalisasi dari library kbbi-python atau buat sendiri
        self.slang_dict = self._load_slang_dictionary()
        
        # Load kamus typo. Kunci berisi spasi dipisahkan menjadi pola frasa
        # yang dicocokkan pada teks utuh; sisanya dicocokkan per token.
        self.typo_dict = self._load_typo_dictionary()
        self._typo_phrases = [
            (re.compile(r'\b' + re.escape(k) + r'\b'), v)
            for k, v in self.typo_dict.items() if ' ' in k
        ]
        
        # Pattern untuk repeated characters
        # HANYA huruf, tidak pernah digit - lihat normalize_repeated_chars().
        self.repeated_char_pattern = re.compile(r'([a-zA-Z])\1{2,}')
    
    def _get_social_media_stopwords(self) -> set:
        """Stopwords tambahan khusus sosial media"""
        return {
            # Partikel sosmed
            'sih', 'kok', 'dong', 'deh', 'lho', 'nih', 'dunk', 'donk',
            # Singkatan umum
            'yg', 'utk', 'dgn', 'tsb', 'dll', 'dsb', 'dst', 'krn', 'pd', 'tdk',
            'dr', 'kpd', 'thd', 'dg', 'gak', 'ga', 'ngga', 'nggak', 'gk',
            # Kata tidak bermakna
            'ya', 'jadi', 'nah', 'wah', 'aduh', 'waduh', 'duh', 'ah', 'oh', 'eh',
            # Kata seru & keyboard-mash. Dipindahkan ke sini dari kamus slang,
            # yang dulu memetakannya ke string kosong. Pemetaan itu menghapusnya
            # walaupun pemanggil mematikan remove_stopwords - padahal jalur
            # sentimen sengaja mempertahankan kata fungsi. Sebagai stopword,
            # penghapusannya tunduk pada kebijakan yang benar.
            'anjay', 'anjir', 'astaga', 'asem', 'asdf', 'asdfgh',
            # Klitik yang sering ditulis terpisah. Terukur pada korpus produksi:
            # 'nya' muncul 110x sebagai token berdiri sendiri - terbanyak kedua
            # setelah kata isi - padahal ia tidak membawa makna apa pun.
            'nya', 'lah', 'kah', 'pun', 'tuh', 'yah', 'sih2',
        }
    
    def _load_slang_dictionary(self) -> Dict[str, str]:
        """
        Load kamus slang/singkatan Indonesia
        Alternatif: bisa gunakan library colloquial-indonesian-lexicon
        atau https://github.com/nasalsabila/kamus-alay
        """
        # Kamus komprehensif untuk normalisasi - DIPERBAIKI
        slang_dict = {
            # Negasi - variasi kata "tidak"
            'gak': 'tidak', 'ga': 'tidak', 'gk': 'tidak', 'ngga': 'tidak', 
            'nggak': 'tidak', 'enggak': 'tidak', 'kaga': 'tidak',
            'kagak': 'tidak', 'nda': 'tidak', 'ndak': 'tidak', 'gag': 'tidak',
            
            # PERBAIKAN: HAPUS mapping huruf tunggal yang berbahaya
            # DIHAPUS: 'g': 'tidak', karena menghapus huruf 'g' di semua kata
            
            # Singkatan umum
            'yg': 'yang', 'dgn': 'dengan', 'utk': 'untuk', 'krn': 'karena',
            'pd': 'pada', 'tdk': 'tidak', 'dr': 'dari', 'dg': 'dengan',
            'kpd': 'kepada', 'tsb': 'tersebut', 'thd': 'terhadap',
            'dll': 'dan lain lain', 'dsb': 'dan sebagainya', 'dst': 'dan seterusnya',
            'spt': 'seperti', 'krg': 'kurang', 'sdh': 'sudah', 'blm': 'belum',
            'jd': 'jadi', 'jg': 'juga', 'tp': 'tetapi', 'kl': 'kalau',
            'bs': 'bisa', 'hrs': 'harus', 'spy': 'supaya', 'trs': 'terus',
            'msg': 'masing', 'lwt': 'lewat', 'sgt': 'sangat', 
            'bgt': 'banget', 'bgt2': 'banget', 'bngt': 'banget', 'bingit': 'banget',
            'bgd': 'banget', 'bingits': 'banget', 'pisan': 'banget',
            'bener': 'benar', 'bnr': 'benar', 'bnrn': 'benar',
            'sbg': 'sebagai', 'sblm': 'sebelum', 'stlh': 'setelah',
            'kmrn': 'kemarin', 'skrg': 'sekarang', 'ntr': 'nanti', 'tar': 'nanti',
            'skg': 'sekarang', 'skr': 'sekarang',
            
            # Pronoun - kata ganti
            # 'aku' adalah kata baku, bukan slang - tidak dipetakan.
            'gue': 'saya', 'gw': 'saya', 'ane': 'saya',
            'lo': 'kamu', 'lu': 'kamu', 'elu': 'kamu', 'loe': 'kamu',
            'agan': 'kamu', 'sista': 'saudara', 'bro': 'saudara', 'sis': 'saudara',
            'bokap': 'ayah', 'nyokap': 'ibu', 'ortu': 'orang tua',
            
            # Kata kerja slang.
            #
            # Kata yang SUDAH baku tidak dipetakan. Versi sebelumnya memetakan
            # 'cari'->'mencari', 'lihat'->'melihat', 'coba'->'mencoba', yaitu
            # arah terbalik dari normalisasi: bentuk dasar diubah menjadi
            # berimbuhan, lalu stemming melucutinya kembali. Pada jalur sentimen
            # yang tidak memakai stemming, "coba lihat dulu" berubah menjadi
            # "mencoba melihat dulu". Terukur: 207 token di 187 dokumen.
            # Entri identitas ('makan'->'makan', 'beli'->'beli') juga dibuang.
            'nyari': 'cari', 'nyariin': 'mencarikan',
            'ngasih': 'memberi', 'ngasi': 'memberi',
            'ngomong': 'bicara', 'omong': 'bicara',
            'nyoba': 'coba', 'cobain': 'coba',
            'pake': 'pakai', 'pk': 'pakai', 'pke': 'pakai',
            'tau': 'tahu', 'tw': 'tahu', 'tauu': 'tahu',
            'liat': 'lihat', 'ngeliat': 'lihat', 'ngelihat': 'lihat',
            'mkn': 'makan', 'maem': 'makan',
            'bobo': 'tidur', 'bobe': 'tidur', 'turu': 'tidur',
            'ngerjain': 'kerjakan', 'kerjain': 'kerjakan',
            'bli': 'beli', 'jualin': 'jual',
            
            # Kata sifat & keterangan.
            #
            # ATURAN: kunci harus berupa bentuk TIDAK BAKU. Kata baku tidak
            # boleh dipetakan ke sinonimnya - itu substitusi makna, bukan
            # normalisasi. Versi sebelumnya memetakan 'mantap'/'keren'->'bagus'
            # dan 'jelek'->'buruk', padahal keempatnya kata baku dengan kadar
            # yang berbeda. Untuk jalur sentimen dampaknya langsung: token yang
            # dilihat model bukan lagi token yang ditulis penulisnya, dan
            # 'mantap' adalah penanda positif yang jauh lebih kuat dari 'bagus'.
            # Terukur: 178 token di 163 dokumen.
            'jlek': 'jelek', 'ancur': 'hancur',
            'mantul': 'mantap', 'mantep': 'mantap',
            'oke': 'baik', 'ok': 'baik', 'okeh': 'baik',
            'asoy': 'asyik', 'asik': 'asyik',
            'pinter': 'pintar', 'pntr': 'pintar',
            'males': 'malas', 'mls': 'malas',
            'cape': 'capek',
            
            # Kata tanya
            'gimana': 'bagaimana', 'gmn': 'bagaimana', 'gimn': 'bagaimana', 
            'gmna': 'bagaimana', 'gemana': 'bagaimana',
            # 'kenapa' TIDAK dipetakan: baku menurut KBBI, sama seperti
            # 'mengapa'. Memetakannya adalah substitusi makna, bukan
            # normalisasi - kelas yang sama dengan 'mantap'->'bagus'.
            'knp': 'mengapa', 'napa': 'mengapa', 'ngapa': 'mengapa',
            'kpn': 'kapan',
            # 'mana' TIDAK dipetakan: ia berdiri sendiri ("mana yang benar",
            # "ke mana", "mana mungkin"), bukan singkatan dari 'dimana'.
            'dmn': 'di mana',
            
            # Kata umum lainnya
            'emang': 'memang', 'emg': 'memang', 'mang': 'memang', 'mmg': 'memang',
            'aja': 'saja', 'aj': 'saja', 'ajah': 'saja',
            'udah': 'sudah', 'udh': 'sudah', 'dah': 'sudah',
            'belom': 'belum', 'blm': 'belum', 'blom': 'belum',
            # 'lagi' TIDAK dipetakan: ia bermakna "sedang" MAUPUN "kembali/lebih"
            # ("datang lagi", "lagi satu"). Memaksakan 'sedang' merusak makna,
            # dan ini pemetaan yang paling sering kena - terukur 604 token di
            # 514 dokumen. Hanya singkatannya yang dinormalkan, ke bentuk baku
            # 'lagi' sendiri, sehingga ambiguitasnya dibiarkan utuh untuk model.
            'lg': 'lagi', 'lgi': 'lagi',
            # Partikel TIDAK lagi dipetakan ke string kosong di sini.
            # Menghapusnya lewat kamus slang membuatnya hilang walaupun pemanggil
            # mematikan remove_stopwords - dan jalur sentimen justru sengaja
            # mempertahankan kata fungsi. Semuanya sudah ada di
            # _get_social_media_stopwords(), sehingga penghapusannya kini
            # mengikuti kebijakan stopword sebagaimana mestinya.
            'ama': 'sama', 'sm': 'sama',
            'abis': 'habis', 'abs': 'habis',
            'bkn': 'bukan',
            
            # Slang populer sosial media
            'kepo': 'ingin tahu', 'kpo': 'ingin tahu',
            'lebay': 'berlebihan', 'lbay': 'berlebihan', 'elbay': 'berlebihan',
            'php': 'pemberi harapan palsu',
            'bokek': 'tidak punya uang', 'boke': 'tidak punya uang',
            'baper': 'bawa perasaan', 'bapar': 'bawa perasaan',
            'kuy': 'ayo', 'yuk': 'ayo', 'yok': 'ayo', 'gas': 'ayo',
            'mager': 'malas gerak', 'gabut': 'tidak ada kegiatan',
            'sotoy': 'sok tahu', 'songong': 'sombong',
            'bucin': 'budak cinta', 'jones': 'jomblo ngenes',
            # Entri identitas dibuang seluruhnya (11 buah): 'bingung'->'bingung'
            # dan sejenisnya tidak melakukan apa pun selain memperbesar kamus.
            'bgg': 'bingung',
            'santuy': 'santai', 'slow': 'santai',
            # 'receh' TIDAK dipetakan ke 'murah': maknanya "uang kecil" atau
            # "lelucon ringan", bukan harga.
            'murahe': 'murah',
            
            # Terima kasih & salam
            'tq': 'terima kasih', 'thx': 'terima kasih', 'thanks': 'terima kasih', 
            'mksh': 'terima kasih', 'mksih': 'terima kasih', 'thks': 'terima kasih',
            'trims': 'terima kasih', 'trimakasih': 'terima kasih',
            'tengkyu': 'terima kasih', 'tenkyu': 'terima kasih',
            'maafin': 'maafkan', 'sori': 'maaf',
            'min': 'admin', 'kak': 'kakak', 'bg': 'abang', 'mba': 'mbak',
            
            # Angka dalam kata (leet speak).
            #
            # ANGKA TELANJANG SENGAJA TIDAK ADA DI SINI. Versi sebelumnya
            # memetakan '9'->'yang', '4'->'untuk', '8'->'delapan', sehingga
            # setiap bilangan yang berdiri sendiri ikut tergantikan:
            # "sesuai pasal 4 ayat 9" menjadi "sesuai pasal untuk ayat yang".
            # Terukur pada empat korpus proyek: 161 token di 145 dokumen rusak.
            # Hanya pola leet yang mengandung huruf yang dipertahankan, karena
            # pola itu tidak mungkin merupakan bilangan.
            'j4d1': 'jadi', 'd1': 'di', 'k3': 'ke', '4ku': 'aku', 'y4': 'ya',
            
            # PERBAIKAN: HAPUS 'g4': 'tidak' karena bisa menghapus 'g' di kata lain
            
            # Emoji tekstual & ekspresi
            'wkwk': 'tertawa', 'wkwkwk': 'tertawa', 'wkwkwkwk': 'tertawa',
            'haha': 'tertawa', 'hehe': 'tertawa', 'hihi': 'tertawa', 
            'huhu': 'sedih', 'hiks': 'sedih', 'uhuk': 'batuk',
            # Kata seru/keyboard-mash juga tidak lagi dipetakan ke string kosong
            # di sini; lihat catatan pada partikel di atas. Semuanya dipindahkan
            # ke _get_social_media_stopwords() agar tunduk pada remove_stopwords.
            
            # Kata asing yang sering dipakai
            'sorry': 'maaf', 'please': 'tolong', 'thank': 'terima kasih',
            'good': 'bagus', 'nice': 'bagus', 'cool': 'keren',
            'best': 'terbaik', 'worst': 'terburuk',
            'like': 'suka', 'love': 'cinta', 'hate': 'benci',
            'yes': 'ya', 'no': 'tidak', 'maybe': 'mungkin',
            
            # Typo & variasi umum
            'sama2': 'sama sama', 'sm2': 'sama sama',
            'jgn': 'jangan', 'jng': 'jangan',
            'brp': 'berapa', 'brpa': 'berapa',

            # ── Singkatan yang ditemukan secara empiris pada korpus produksi ──
            #
            # Daftar ini TIDAK ditebak. Seluruh token korpus dinormalkan dengan
            # kamus yang ada, lalu yang tersisa dicocokkan dengan 29.933 kata
            # dasar bawaan Sastrawi; yang tidak dikenali dan sering muncul
            # dikumpulkan di sini. Akronim dan nama diri yang ikut terjaring
            # (DPR, IKN, APBN, BUMN, NKRI, Kemenkeu, Jokowi) sengaja TIDAK
            # dimasukkan - keduanya memang bukan kata dasar, tetapi merupakan
            # istilah sah yang justru bernilai sebagai kata kunci topik.
            'klo': 'kalau', 'klau': 'kalau', 'klu': 'kalau', 'kalo': 'kalau',
            'karna': 'karena', 'krna': 'karena',
            'tpi': 'tetapi', 'tpe': 'tetapi',
            'org': 'orang', 'orng': 'orang',
            'byk': 'banyak', 'bnyak': 'banyak', 'bnyk': 'banyak',
            'byr': 'bayar',
            'dri': 'dari', 'dlm': 'dalam',
            'trus': 'terus',
            'sampe': 'sampai', 'smpe': 'sampai', 'smpai': 'sampai',
            'ngak': 'tidak', 'nggk': 'tidak',
            'sma': 'sama', 'mreka': 'mereka',
            'koq': 'kok',
            'mikir': 'pikir', 'mikirin': 'pikir',
            'nagih': 'tagih', 'nagihin': 'tagih',
            'ngutang': 'utang', 'ngutangin': 'utang',
            'bgmn': 'bagaimana', 'gmn': 'bagaimana', 'gmna': 'bagaimana',
            'jngn': 'jangan', 'smoga': 'semoga', 'gtu': 'begitu',
            'gitu': 'begitu', 'gini': 'begini',
            'sampek': 'sampai', 'mintak': 'minta', 'mintain': 'minta',
            'ngajarin': 'ajar', 'ngerti': 'mengerti', 'ngurusin': 'urus',
            # 'bikin' TIDAK dipetakan ke 'buat': 'buat' ada di
            # TOPIC_KEYWORD_STOPWORDS, sehingga pemetaan itu menghapus kata isi
            # secara diam-diam lewat pintu belakang. Bentuk baku 'bikin' saja.
            'bkin': 'bikin',
            'milyar': 'miliar', 'trilyun': 'triliun',
            'duwit': 'duit',
        }

        return slang_dict
    
    def _load_typo_dictionary(self) -> Dict[str, str]:
        """Kamus perbaikan typo umum"""
        return {
            # Kesalahan ejaan umum
            'terima kasi': 'terima kasih', 'terimakasi': 'terima kasih',
            'terimakasih': 'terima kasih', 'trimakasih': 'terima kasih',
            'silahkan': 'silakan', 'silahakan': 'silakan',
            
            # Double huruf yang salah
            'applikasi': 'aplikasi', 'aplication': 'aplikasi',
            'comunity': 'komunitas', 'communtiy': 'komunitas',
            'reccomend': 'rekomendasi', 'recomend': 'rekomendasi',
            'rekomen': 'rekomendasi',
            
            # Typo keyboard (huruf berdekatan)
            'drngan': 'dengan', 'drpat': 'dapat', 'adlah': 'adalah',
            'trlalu': 'terlalu', 'brlaku': 'berlaku',
            'brkata': 'berkata', 'brjalan': 'berjalan',
            'mngkin': 'mungkin', 'mnrt': 'menurut',
            'untk': 'untuk', 'untul': 'untuk', 'unruk': 'untuk',
            'dgan': 'dengan', 'denga': 'dengan',
            'yng': 'yang', 'yagn': 'yang',
            
            # Kata yang sering salah eja
            'apotik': 'apotek',
            'analisa': 'analisis', 'analisys': 'analisis',
            'ijin': 'izin',
            'jaman': 'zaman',
            'praktek': 'praktik',
            'resiko': 'risiko',
            'sistim': 'sistem', 'system': 'sistem',
            'tehnik': 'teknik',
            'tehnologi': 'teknologi', 'technology': 'teknologi',
            'methode': 'metode',
            'aktifitas': 'aktivitas',
            'effectif': 'efektif',
            'effisien': 'efisien',
            'standard': 'standar',
            'frequency': 'frekuensi',
            'kwalitas': 'kualitas',
            'kwantitas': 'kuantitas',

            # Brand/app yang sering salah eja
            'wa': 'whatsapp', 'wassap': 'whatsapp',
            'tokped': 'tokopedia', 'toped': 'tokopedia',
            'shope': 'shopee',
            'go-jek': 'gojek',
        }
    
    def normalize_slang(self, text: str) -> str:
        """Normalisasi slang dan singkatan"""
        words = text.split()
        normalized_words = []
        
        for word in words:
            if word in self.slang_dict:
                normalized = self.slang_dict[word]
                if normalized:  # Skip jika mapping ke empty string
                    normalized_words.append(normalized)
            else:
                normalized_words.append(word)
        
        return ' '.join(normalized_words)
    
    def fix_typos(self, text: str) -> str:
        """
        Perbaiki typo umum, frasa lebih dulu baru per kata.

        Kamus memuat kunci berisi spasi ("terima kasi" -> "terima kasih").
        Versi sebelumnya hanya mencocokkan per token hasil `split()`, sehingga
        kunci seperti itu TIDAK PERNAH cocok - entrinya ada tetapi mati.
        Frasa karena itu dicocokkan lebih dulu pada teks utuh, dengan batas kata
        agar tidak memotong di tengah kata lain.
        """
        for phrase, replacement in self._typo_phrases:
            text = phrase.sub(replacement, text)

        return ' '.join(self.typo_dict.get(word, word) for word in text.split())
    
    def normalize_repeated_chars(self, text: str) -> str:
        """
        Normalisasi huruf yang diulang untuk penekanan: "bagusssss" -> "bagus".

        DUA hal yang dulu salah di sini.

        **Angka tidak boleh disentuh.** Pola lama berlaku untuk karakter apa pun
        termasuk digit, sehingga bilangan ikut dipangkas: "utang 1000 triliun"
        menjadi "utang 10 triliun" dan "anggaran 100000" menjadi "anggaran 10".
        Fatal untuk korpus yang justru membahas nominal.

        **Memangkas ke SATU huruf merusak kata berhuruf ganda.** "maaaaf"
        menjadi "maf". Bentuk dua-huruf karena itu dicoba lebih dulu dan
        diterima bila kata utuhnya dikenal kamus kata dasar Sastrawi; kalau
        tidak, baru dipangkas menjadi satu huruf - yang benar untuk
        "bagusss" -> "bagus".
        """
        def ganti(match):
            huruf = match.group(1)
            awal, akhir = match.start(), match.end()
            kiri = text.rfind(' ', 0, awal) + 1
            kanan = text.find(' ', akhir)
            kanan = len(text) if kanan == -1 else kanan

            for kandidat in (huruf * 2, huruf):
                kata = (text[kiri:awal] + kandidat + text[akhir:kanan]).lower()
                if kata in ROOT_WORDS:
                    return kandidat

            # Tidak dikenal kamus: perlakukan sebagai pemanjangan gaya media
            # sosial, yang jauh lebih lazim daripada huruf ganda sungguhan.
            return huruf

        return self.repeated_char_pattern.sub(ganti, text)
    
    def remove_emoticons(self, text: str) -> str:
        """Hapus emoticon dan emoji"""
        emoji_pattern = re.compile(
            "["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map
            u"\U0001F1E0-\U0001F1FF"  # flags
            "]+",
            flags=re.UNICODE
        )
        return emoji_pattern.sub('', text)
    
    def clean_texts(self, texts: List[str]) -> List[str]:
        """Bersihkan daftar teks; jumlah dan urutannya dijaga."""
        return [self.clean_text(text) for text in texts]

    @staticmethod
    def filter_rare_tokens(texts: List[str], min_freq: int = 2) -> List[str]:
        """
        Buang token yang muncul kurang dari `min_freq` kali DI SELURUH korpus.

        Ini satu-satunya operasi yang butuh melihat korpus utuh, bukan satu
        dokumen, sehingga ia dipisah dari `clean_text`.

        Alasannya khusus untuk model bag-of-words. Terukur pada 885 komentar
        YouTube: **63,4% kosakata hanya muncul sekali** (1 523 dari 2 403 tipe),
        sebagian besar berupa salah ketik. c-TF-IDF memberi bobot tinggi pada
        kata langka, sehingga kata-kata itu justru naik menjadi label topik -
        terlihat sebagai topik berisi "mahakuasah, majak, kendaran, rakya",
        yaitu klaster salah ketik, bukan klaster makna.

        JANGAN dipakai pada jalur transformer: model subword menangani kata
        langka dengan memecahnya, dan membuang kata justru merusak kalimat.

        Args:
            texts: Teks yang SUDAH dibersihkan per dokumen.
            min_freq: Frekuensi minimum agar sebuah token dipertahankan.

        Returns:
            Daftar teks dengan panjang dan urutan yang sama; dokumen bisa
            menjadi kosong bila seluruh tokennya langka.
        """
        if min_freq <= 1:
            return list(texts)

        from collections import Counter

        counts: Counter = Counter()
        for text in texts:
            counts.update(text.split())

        kept = {token for token, n in counts.items() if n >= min_freq}
        dibuang = len(counts) - len(kept)
        if dibuang:
            logger.info(
                f"Filter token langka: {dibuang} dari {len(counts)} tipe kata "
                f"dibuang (frekuensi < {min_freq})"
            )

        return [' '.join(t for t in text.split() if t in kept) for text in texts]
    
    def clean_text(self, text: str) -> str:
        """Enhanced cleaning for single text"""
        if not text or not isinstance(text, str):
            return ""
        
        # Step 1: Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Step 2: Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Step 3: Remove mentions, hashtags, dan RT
        text = re.sub(r'@\w+|RT\s+', '', text)
        # Tagar: hanya tanda pagarnya yang dibuang, katanya DIPERTAHANKAN.
        # Pola lama `#\w+` membuang tagar beserta katanya, sehingga
        # "baca #pajak dan #korupsi" menjadi "baca dan" - justru kata paling
        # topikal yang hilang, tepat pada modul yang paling membutuhkannya.
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Step 4: Remove emoticons dan emoji
        text = self.remove_emoticons(text)
        
        # Step 5: Case folding
        if self.case_folding:
            text = text.lower()

        # Step 6: Buang karakter di luar alfabet Latin, angka, dan tanda baca
        # yang bermakna. Sisa emoji, simbol, dan karakter kontrol selalu dibuang;
        # tanda baca hanya dibuang bila diminta.
        #
        # Sebelumnya baris ini membuang SELURUH karakter non-alfanumerik tanpa
        # syarat, sehingga `remove_punctuation` hanya dibaca di __init__ dan
        # tidak pernah berefek apa pun. Itu penting bagi jalur transformer:
        # IndoBERT dilatih pada teks bertanda baca, dan tanda seru membawa
        # sinyal intensitas yang nyata pada komentar.
        if self.remove_punctuation:
            text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        else:
            # Pertahankan tanda baca kalimat; sisanya tetap dibersihkan.
            text = re.sub(r'[^a-zA-Z0-9\s' + re.escape(self.KEPT_PUNCTUATION) + r']', ' ', text)
        
        # Step 7: Normalize repeated characters
        text = self.normalize_repeated_chars(text)
        
    
        # Step 8: Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Step 9: Fix typos
        if self.enable_fix_typos:
            text = self.fix_typos(text)

        
        # Step 10: Normalize slang dan singkatan
        if self.enable_normalize_slang:
            text = self.normalize_slang(text)

        
        # Step 11: Remove numbers (opsional)
        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        # Step 12: Tokenization
        tokens = text.split()
        
        # Step 13: Remove stopwords menggunakan Sastrawi + NLTK
        if self.remove_stopwords:
            tokens = [token for token in tokens if token not in self.all_stopwords]
        
        # Step 14: Stemming menggunakan Sastrawi
        if self.stemming:
            tokens = [self.stemmer.stem(token) for token in tokens]
        
        # Step 15: Buang token yang terlalu pendek (ambang bergantung profil)
        if self.min_token_length > 1:
            tokens = [token for token in tokens if len(token) >= self.min_token_length]

        # Step 16: Gabungkan kata identik yang berurutan
        if self.collapse_repeated_words and len(tokens) > 1:
            cleaned_tokens = [tokens[0]]
            for i in range(1, len(tokens)):
                if tokens[i] != tokens[i-1]:
                    cleaned_tokens.append(tokens[i])
            tokens = cleaned_tokens
        
        # Rejoin tokens
        cleaned_text = ' '.join(tokens)
        
        return cleaned_text.strip()
    
    def get_statistics(self, original: str, cleaned: str) -> dict:
        """Get preprocessing statistics"""
        return {
            'original_length': len(original),
            'cleaned_length': len(cleaned),
            'original_words': len(original.split()),
            'cleaned_words': len(cleaned.split()),
            'removed_chars': len(original) - len(cleaned),
            'removed_words': len(original.split()) - len(cleaned.split()),
            'compression_ratio': round(len(cleaned) / len(original) * 100, 2) if len(original) > 0 else 0
        }
    
    def get_cleaning_report(self, original: str, cleaned: str) -> dict:
        """Get detailed cleaning report"""
        stats = self.get_statistics(original, cleaned)
        
        return {
            **stats,
            'original_text': original,
            'cleaned_text': cleaned,
            'steps_applied': self._get_applied_steps()
        }
    
    def _get_applied_steps(self) -> List[str]:
        """Get list of applied preprocessing steps"""
        steps = [
            'URL removal',
            'Email removal',
            'Mention/hashtag removal',
            'Emoticon/emoji removal',
            'Repeated character normalization',
        ]
        
        if self.case_folding:
            steps.append('Case folding')
        if self.enable_fix_typos:
            steps.append('Typo correction')
        if self.enable_normalize_slang:
            steps.append('Slang normalization')
        if self.remove_numbers:
            steps.append('Number removal')
        if self.remove_stopwords:
            steps.append('Stopword removal (Sastrawi + NLTK)')
        if self.stemming:
            steps.append('Stemming (Sastrawi)')
        
        steps.append('Short token removal')
        steps.append('Duplicate word removal')
        
        return steps