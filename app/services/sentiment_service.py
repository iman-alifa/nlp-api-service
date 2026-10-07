import asyncio
import gc
import json
import os
import shutil
import numpy as np
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from app.config import hub_kwargs, settings
from app.preprocessing.text_cleaner import PROFILES, TextCleaner
from app.utils.logger import setup_logger
from app.utils.training import (
    classification_metrics,
    fit_temperature,
    should_accept_checkpoint,
    compute_class_weights,
    label_distribution,
    lepas_memori,
    pool_fingerprint,
    set_seed,
    stratified_split,
    write_provenance,
)

logger = setup_logger(__name__)

# Peta label BAWAAN, dipakai bila config model tidak menyebutkan labelnya
# (mis. `mdhugol`, yang id2label-nya masih LABEL_0/1/2). Urutan ini sudah
# diverifikasi lewat kartu model mdhugol dan kebetulan sama untuk crypter70.
#
# JANGAN diandalkan untuk model baru: urutannya berbeda-beda antar checkpoint.
# `taufiqdp/indonesian-sentiment` memakai 0=negatif, yaitu KEBALIKANNYA - dan
# memakai peta yang salah membalik seluruh hasil tanpa memunculkan galat apa pun.
# Karena itu `_resolve_label_map()` membaca config model lebih dulu.
LABEL_MAP = {0: 'positive', 1: 'neutral', 2: 'negative'}
LABEL_TO_ID = {v: k for k, v in LABEL_MAP.items()}

# Ragam penulisan label yang dipakai berbagai checkpoint publik.
_LABEL_ALIASES = {
    'positive': 'positive', 'positif': 'positive', 'pos': 'positive',
    'neutral': 'neutral', 'netral': 'neutral', 'neu': 'neutral',
    'negative': 'negative', 'negatif': 'negative', 'neg': 'negative',
}

# Suhu kalibrasi disimpan bersama checkpoint, bukan di .env: ia milik BOBOT
# tertentu. Checkpoint hasil retraining punya kalibrasi sendiri, dan memakai
# suhu model dasar untuknya akan salah.
CALIBRATION_FILE = 'calibration.json'


class SentimentService:
    """Sentiment Analysis Service with lazy model loading for low-RAM environments"""

    def __init__(self, model_path: Optional[str] = None):
        logger.info("Initializing Sentiment Service (lazy mode)...")
        self.base_model_name = settings.sentiment_base_model

        # Checkpoint hasil retraining dipakai kembali setelah restart. Tanpa ini
        # setiap `python run.py` mengembalikan model ke bobot dasar, sehingga
        # seluruh hasil active learning hilang begitu terminal ditutup.
        resolved = model_path or self._discover_checkpoint()
        self.model_name = resolved or self.base_model_name
        self.using_retrained = bool(resolved)

        if self.using_retrained:
            logger.info(f"🔁 Checkpoint hasil retraining ditemukan: {self.model_name}")
        else:
            logger.info(f"Menggunakan model dasar: {self.model_name}")

        self.tokenizer = None
        self.model = None
        self.device = None
        self._loaded = False
        self.temperature = settings.sentiment_temperature
        # Peta label milik bobot yang sedang dipakai; diisi saat model dimuat.
        self.label_map = dict(LABEL_MAP)
        logger.info("✅ Sentiment service ready (model will load on first use)")

    @staticmethod
    def _discover_checkpoint() -> Optional[str]:
        """Kembalikan path checkpoint retraining bila direktorinya valid."""
        path = settings.sentiment_checkpoint_path
        if os.path.isdir(path) and os.path.exists(os.path.join(path, "config.json")):
            return path
        return None

    def _ensure_loaded(self):
        """Lazy-load model on first inference to keep startup RAM low."""
        if self._loaded:
            return
        try:
            logger.info(f"Loading sentiment model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, **hub_kwargs(self.model_name)
            )
            # `low_cpu_mem_usage=True` memuat bobot langsung ke tensor model
            # alih-alih menyalin seluruhnya ke RAM lebih dulu, sehingga PUNCAK
            # memori saat memuat kira-kira separuh. Lonjakan sesaat itulah yang
            # memicu OOM di kontainer berkuota ketat; pada Windows ia muncul
            # sebagai `Windows fatal exception: access violation` di dalam
            # safetensors ketika model ketiga dimuat pada satu proses.
            #
            # BUTUH paket `accelerate` (ada di requirements.txt). Tanpa paket
            # itu transformers melempar ImportError, blok except di bawah
            # menangkapnya, dan service jatuh ke jalur daftar kata TANPA satu
            # pun galat sampai ke pemanggil - terukur, akurasi SmSA anjlok
            # 0,9160 -> 0,6740 sementara seluruh suite tes tetap hijau.
            # tests/test_model_loading_smoke.py sekarang menjaga kelas kegagalan itu.
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, low_cpu_mem_usage=True, **hub_kwargs(self.model_name)
            )
            # `use_gpu` sebelumnya tidak pernah dibaca: GPU selalu dipakai bila
            # tersedia, sehingga USE_GPU=false tidak berefek apa pun.
            self.device = (
                "cuda" if settings.use_gpu and torch.cuda.is_available() else "cpu"
            )
            self.model.to(self.device)
            self.model.eval()
            self.label_map = self._resolve_label_map()
            self.temperature = self._load_calibration(self.model_name)
            self._loaded = True
            logger.info(
                f"✅ Sentiment model loaded on {self.device} (T={self.temperature})"
            )
        except Exception as e:
            logger.error(f"❌ Failed to load sentiment model: {str(e)}")

            # Checkpoint rusak tidak boleh menjatuhkan service ke rule-based;
            # coba model dasar dulu sebelum menyerah.
            if self.model_name != self.base_model_name:
                logger.warning(f"Checkpoint gagal dimuat, fallback ke {self.base_model_name}")
                self.model_name = self.base_model_name
                self.using_retrained = False
                self._loaded = False
                return self._ensure_loaded()

            self.model = None
            self.tokenizer = None
            self._loaded = True  # Don't retry on every call
            logger.warning("Using rule-based sentiment analysis as fallback")


    @property
    def label_to_id(self) -> Dict[str, int]:
        """
        Kebalikan `label_map`, untuk mengubah label teks menjadi id kelas.

        WAJIB dipakai pada jalur retraining. Melatih dengan peta bawaan padahal
        model memakai urutan lain akan memberi label yang tertukar - modelnya
        tetap terlatih, lossnya tetap turun, dan hasilnya kacau tanpa satu pun
        galat muncul.
        """
        return {v: k for k, v in self.label_map.items()}

    def _save_weights(self, path: str) -> None:
        """Tulis bobot, tahan terhadap berkas yang masih ter-mmap (Windows).

        safetensors memuat bobot lewat memory-map. Di Windows, menulis ke
        berkas yang masih dipetakan proses ini sendiri ditolak sistem dengan
        `os error 1224` (ERROR_USER_MAPPED_FILE) - dan pesannya muncul sebagai
        "Error while serializing", yang tidak menyebut penyebab sebenarnya.

        Terjadi nyata pada iterasi 2 studi kasus: service memuat checkpoint dari
        `sentiment_retrained/` saat start, lalu retraining menyimpan kembali ke
        direktori yang sama. Pelatihan 330 sampel berjalan 10 menit lalu hilang
        pada langkah penyimpanan - dan `config.json` sempat tertulis sementara
        `model.safetensors` tidak, meninggalkan checkpoint yang tidak konsisten.

        `retrain_from_baseline` membuat ini jarang terjadi (bobot lama dilepas
        saat ditukar ke bobot dasar), tetapi jalur tanpa opsi itu tetap rentan,
        dan pekerjaan sepuluh menit tidak boleh hilang karena pemetaan berkas.
        """
        try:
            self.model.save_pretrained(path)
            return
        except OSError as exc:
            logger.warning(
                f"Penyimpanan bobot ke {path} gagal ({exc}); melepas pemetaan "
                "berkas lalu mencoba sekali lagi."
            )

        # Paksa pelepasan referensi yang mungkin masih memetakan berkas tujuan.
        gc.collect()

        try:
            self.model.save_pretrained(path)
        except OSError:
            # Jalur terakhir: tulis ke direktori sementara di induk yang sama
            # (agar tetap satu volume), lalu tukar. Berkas lama dihapus dulu -
            # pada titik ini tidak ada lagi yang memetakannya.
            sementara = f"{path}.tmp_save"
            shutil.rmtree(sementara, ignore_errors=True)
            self.model.save_pretrained(sementara)

            for nama in os.listdir(sementara):
                tujuan = os.path.join(path, nama)
                if os.path.exists(tujuan):
                    os.remove(tujuan)
                shutil.move(os.path.join(sementara, nama), tujuan)

            shutil.rmtree(sementara, ignore_errors=True)
            logger.info(f"Bobot tersimpan ke {path} lewat direktori sementara.")

    def _scores_dict(self, row) -> Dict[str, float]:
        """Probabilitas per label, dipetakan lewat `label_map` yang sudah diselesaikan.

        Selalu mengembalikan ketiga kunci dalam urutan tetap positive/neutral/
        negative supaya bentuk responsnya stabil bagi pemanggil, berapa pun
        urutan internal checkpoint-nya. Kelas yang tidak dimiliki model bernilai
        0.0 alih-alih hilang, karena Laravel membaca ketiganya langsung.
        """
        skor = {'positive': 0.0, 'neutral': 0.0, 'negative': 0.0}

        for idx, nama in self.label_map.items():
            if nama in skor and 0 <= int(idx) < len(row):
                skor[nama] = round(row[int(idx)].item(), 4)

        return skor

    def _resolve_label_map(self) -> Dict[int, str]:
        """
        Baca urutan label dari config model, bukan mengasumsikannya.

        Urutan label BERBEDA-BEDA antar checkpoint publik dan tidak mengikuti
        konvensi apa pun. `crypter70` dan `mdhugol` memakai 0=positive, tetapi
        `taufiqdp/indonesian-sentiment` memakai 0=negatif - persis kebalikannya.
        Memakai peta yang salah membalik seluruh hasil **tanpa memunculkan galat**,
        jadi kesalahannya hanya terlihat dari mutu prediksi yang anjlok.

        Config yang masih memakai `LABEL_0/1/2` tidak informatif; untuk itu peta
        bawaan dipakai, yang benar untuk model produksi dan sudah diverifikasi.

        Returns:
            Peta id kelas -> nama label kanonik.
        """
        raw = getattr(getattr(self.model, 'config', None), 'id2label', None) or {}
        resolved: Dict[int, str] = {}

        for key, value in raw.items():
            name = _LABEL_ALIASES.get(str(value).strip().lower())
            if name:
                resolved[int(key)] = name

        if len(resolved) == 3 and len(set(resolved.values())) == 3:
            if resolved != LABEL_MAP:
                logger.warning(
                    f"Urutan label {self.model_name} BERBEDA dari bawaan: {resolved}. "
                    f"Peta dari config model yang dipakai."
                )
            else:
                logger.info(f"Peta label dari config model: {resolved}")
            return resolved

        logger.info(
            f"config {self.model_name} tidak menyebutkan label ({raw or 'kosong'}); "
            f"peta bawaan dipakai: {LABEL_MAP}"
        )
        return dict(LABEL_MAP)

    def _load_calibration(self, model_dir: str) -> float:
        """
        Ambil suhu kalibrasi milik bobot yang sedang dipakai.

        Checkpoint hasil retraining menyimpan suhunya sendiri di
        `calibration.json`; bila tidak ada (mis. model dasar dari HuggingFace),
        dipakai nilai terukur di `settings.sentiment_temperature`.

        Args:
            model_dir: Path checkpoint lokal atau nama repo HuggingFace.

        Returns:
            Suhu T (> 0). 1.0 berarti tanpa kalibrasi.
        """
        default = settings.sentiment_temperature
        path = os.path.join(model_dir, CALIBRATION_FILE)

        if not os.path.isfile(path):
            return default

        try:
            with open(path, encoding='utf-8') as handle:
                value = float(json.load(handle).get('temperature', default))
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"calibration.json tidak terbaca ({exc}); T={default} dipakai")
            return default

        if not (0.05 < value < 20.0):
            logger.warning(f"Suhu pada checkpoint tidak wajar ({value}); T={default} dipakai")
            return default

        logger.info(f"Kalibrasi checkpoint dimuat: T={value}")
        return value

    @staticmethod
    def _save_calibration(model_dir: str, temperature: float, extra: Dict[str, Any]) -> None:
        """Simpan suhu bersama checkpoint agar ikut terpakai setelah restart."""
        try:
            os.makedirs(model_dir, exist_ok=True)
            with open(os.path.join(model_dir, CALIBRATION_FILE), 'w', encoding='utf-8') as handle:
                json.dump({'temperature': temperature, **extra}, handle, indent=2)
        except Exception as exc:  # noqa: BLE001
            # Kalibrasi bersifat penyempurnaan; kegagalan menyimpannya tidak
            # boleh membatalkan checkpoint yang sudah terbukti lebih baik.
            logger.warning(f"Gagal menyimpan calibration.json: {exc}")

    @staticmethod
    def _sanitize_config(preprocessing_config) -> Dict[str, Any]:
        """
        Matikan stemming dan stopword removal untuk jalur BERT.

        IndoBERT memakai tokenisasi subword dan dilatih pada teks alami, sehingga
        morfologi dan kata fungsi justru menjadi sinyal. Menerapkan preprocessing
        gaya bag-of-words merusaknya: pada pengujian, 5 dari 6 kalimat bernegasi
        berbalik dari negative menjadi positive dengan confidence ~0.99
        ("pelayanannya tidak bagus sama sekali" -> "layan bagus").

        Normalisasi slang, perbaikan typo, dan karakter berulang tetap dijalankan
        karena membawa teks media sosial mendekati distribusi data pralatih.

        Args:
            preprocessing_config: Konfigurasi dari pemanggil (dict atau model).

        Returns:
            Dict konfigurasi yang aman untuk model transformer.
        """
        config = dict(TextCleaner._normalize_config(preprocessing_config))

        disabled = [k for k in ('stemming', 'remove_stopwords') if config.get(k)]
        if disabled:
            logger.info(
                f"Preprocessing {disabled} dinonaktifkan untuk sentimen: "
                f"merusak morfologi dan negasi yang dibutuhkan IndoBERT"
            )

        # Aturannya sekarang tinggal di satu tempat, PROFILES['transformer'],
        # supaya kebijakan tiap modul analisis tidak tersebar di service
        # masing-masing dan tidak bisa berubah diam-diam.
        config.update(PROFILES['transformer'])
        return config

    @staticmethod
    def _empty_prediction(original: str) -> Dict[str, Any]:
        """
        Hasil untuk teks yang tidak punya isi apa pun untuk dinilai.

        Wajib ada karena kontrak penjajaran mengharuskan N teks masuk = N hasil
        keluar, sehingga baris kosong TIDAK boleh dibuang. Tetapi meneruskannya
        ke IndoBERT juga salah: model mengklasifikasi urutan `[CLS] [SEP]` dan
        mengembalikan `positive` dengan keyakinan 0,9429 - terukur, konsisten,
        dan sepenuhnya tanpa makna.

        Dampaknya nyata di antarmuka: setiap baris kosong pada CSV yang diunggah
        menaikkan persentase positif, yang justru angka utama yang ditampilkan
        Laravel. Modul aspek mengembalikan `[]` dan modul topik `-1` untuk kasus
        yang sama; di sini padanannya adalah netral dengan keyakinan 0.
        """
        return {
            'text': original,
            'processed_text': '',
            'sentiment': 'neutral',
            'confidence': 0.0,
            'scores': {'positive': 0.0, 'neutral': 0.0, 'negative': 0.0},
            'method': 'empty',
        }

    async def analyze(
        self,
        texts: List[str],
        preprocessing_config: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Analisis sentimen sekumpulan teks.

        Args:
            texts: Daftar teks. Baris kosong dipertahankan pada posisinya.
            preprocessing_config: Konfigurasi pembersihan (dict atau model
                pydantic). Stemming/stopword removal selalu dipaksa mati -
                lihat `_sanitize_config`.

        Returns:
            Dict berisi `predictions` (panjang SAMA dengan `texts`, urutan sama),
            `distribution`, `metrics`, dan `summary`.
        """
        try:
            # Convert Pydantic model to dict if needed
            if preprocessing_config and hasattr(preprocessing_config, 'dict'):
                preprocessing_config = preprocessing_config.dict()

            # Preprocessing
            if preprocessing_config:
                cleaner = TextCleaner(self._sanitize_config(preprocessing_config))
                preprocessed_texts = cleaner.clean_texts(texts)
            else:
                preprocessed_texts = [str(t or '') for t in texts]

            # Baris kosong disisihkan sebelum inferensi, lalu hasilnya
            # dikembalikan ke posisi semula. Pembersihan bisa MENGHASILKAN teks
            # kosong dari masukan yang tidak kosong (mis. "a" atau "😀"), jadi
            # pemeriksaan dilakukan setelah preprocessing, bukan sebelumnya.
            scorable = [i for i, t in enumerate(preprocessed_texts) if t and t.strip()]

            predictions: List[Optional[Dict[str, Any]]] = [None] * len(texts)

            if scorable:
                # Lazy-load model if not loaded yet
                self._ensure_loaded()
                subset = [preprocessed_texts[i] for i in scorable]

                if self.model and self.tokenizer:
                    scored = self._predict_with_model(subset)
                else:
                    scored = self._predict_rule_based(subset)

                for slot, prediction in zip(scorable, scored):
                    # `text` mengembalikan teks ASLI pemanggil; teks hasil
                    # pembersihan tetap tersedia terpisah agar antarmuka bisa
                    # menampilkan keduanya tanpa menebak-nebak.
                    prediction['text'] = texts[slot]
                    prediction['processed_text'] = preprocessed_texts[slot]
                    predictions[slot] = prediction

            for i, prediction in enumerate(predictions):
                if prediction is None:
                    predictions[i] = self._empty_prediction(texts[i])

            final: List[Dict[str, Any]] = [p for p in predictions if p is not None]

            # Distribusi dihitung HANYA atas teks yang benar-benar dinilai.
            # Memasukkan baris kosong ke penyebut membuat persentase netral naik
            # semu dan tidak mencerminkan korpus pengguna.
            scored_only = [p for p in final if p['method'] != 'empty']

            distribution = self._calculate_distribution(scored_only)
            metrics = self._calculate_metrics(scored_only)
            metrics['total_texts'] = len(texts)
            metrics['total_empty'] = len(final) - len(scored_only)

            # Pemotongan dilaporkan, bukan dibiarkan senyap. Tidak satu pun
            # korpus proyek ini melebihi 512 token (maksimum terukur: 253 pada
            # YouTube), tetapi API menerima teks sampai 10 000 karakter, jadi
            # dokumen panjang yang diunggah pengguna bisa kehilangan ekornya -
            # dan pada ulasan, justru bagian akhir yang memuat kesimpulan.
            truncated = sum(1 for p in final if p.get('truncated'))
            metrics['total_truncated'] = truncated
            if truncated:
                logger.warning(
                    f"{truncated} teks melebihi {self.MAX_MODEL_TOKENS} token dan "
                    f"dipotong; bagian setelahnya tidak ikut dinilai"
                )

            errors = sum(1 for p in final if p.get('method') == 'error')
            metrics['total_failed'] = errors
            if errors:
                logger.error(f"{errors} teks gagal dinilai model")

            return {
                'predictions': final,
                'distribution': distribution,
                'metrics': metrics,
                'review_queue': self._build_review_queue(final),
                'summary': self._generate_summary(distribution, len(scored_only))
            }

        except Exception as e:
            logger.error(f"Sentiment analysis error: {str(e)}")
            raise

    # Batas token IndoBERT. Teks yang mengisi seluruh jendela ini hampir pasti
    # terpotong, dan bagian yang terbuang tidak pernah ikut dinilai.
    MAX_MODEL_TOKENS = 512

    # Teks per chunk. Tidak besar agar hemat RAM di kontainer Railway.
    MAX_CHUNK = 256

    def _score_chunk(self, chunk: List[str], slots: List[int],
                     predictions: List[Optional[Dict]]) -> None:
        """
        Nilai satu chunk dan tulis hasilnya ke posisi `slots`.

        Bila chunk gagal (lazimnya kehabisan memori pada teks panjang), chunk
        DIBELAH DUA dan dicoba lagi, sampai ukuran satu. Sebelumnya satu
        kegagalan menjatuhkan seluruh 256 teks di dalamnya menjadi
        `neutral` berkeyakinan 0 - padahal yang bermasalah lazimnya hanya satu
        teks. Dengan pembelahan, hanya teks yang benar-benar rusak yang
        ditandai `error`, sisanya tetap memperoleh prediksi sungguhan.
        """
        try:
            inputs = self.tokenizer(
                chunk,
                return_tensors="pt",
                truncation=True,
                max_length=self.MAX_MODEL_TOKENS,
                padding=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                # Temperature scaling: logit dibagi T sebelum softmax.
                # Pembagian skalar positif tidak mengubah urutan logit, jadi
                # argmax - dan karenanya seluruh metrik klasifikasi - persis
                # sama; hanya keyakinannya yang menjadi jujur.
                probs = torch.nn.functional.softmax(
                    outputs.logits / self.temperature, dim=-1
                )

            # Teks yang mengisi seluruh jendela dianggap terpotong. Tanpa
            # penanda ini, ekor dokumen panjang hilang dari analisis tanpa
            # jejak apa pun di keluaran - dan justru bagian akhir ulasan yang
            # sering memuat kesimpulan penilaiannya.
            filled = inputs['attention_mask'].sum(dim=1)

            for j, text in enumerate(chunk):
                predicted_class = torch.argmax(probs[j]).item()
                prediction = {
                    'text': text,
                    'sentiment': self.label_map.get(predicted_class, 'neutral'),
                    'confidence': round(probs[j][predicted_class].item(), 4),
                    # Dibangun lewat label_map yang SUDAH diselesaikan, bukan
                    # indeks 0/1/2 tetap. `sentiment` di atas memang sudah
                    # memakai peta itu, tetapi `scores` dulu memakai indeks
                    # tetap - sehingga untuk checkpoint dengan urutan label
                    # berbeda (`taufiqdp` memakai 0=negatif, kebalikan
                    # `crypter70`) labelnya benar sementara vektor
                    # probabilitasnya TERTUKAR, tanpa galat apa pun. Persis
                    # bahaya yang membuat `_resolve_label_map` ada; separuhnya
                    # saja yang terpakai. Bukan masalah teoretis: SENTIMENT_MODELS
                    # memang dirancang agar checkpoint bisa ditukar.
                    'scores': self._scores_dict(probs[j]),
                    # Setiap hasil menyebut jalur yang memproduksinya. Tanpa
                    # ini, prediksi IndoBERT dan tebakan daftar-kata cadangan
                    # tidak bisa dibedakan pemanggil - degradasi diam-diam
                    # yang persis sama sudah pernah terjadi di modul topik.
                    'method': 'indobert',
                    'temperature': self.temperature,
                }
                if int(filled[j].item()) >= self.MAX_MODEL_TOKENS:
                    prediction['truncated'] = True
                predictions[slots[j]] = prediction

            del inputs, outputs, probs, filled
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as exc:  # noqa: BLE001
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if len(chunk) > 1:
                logger.warning(
                    f"Chunk {len(chunk)} teks gagal ({type(exc).__name__}); "
                    f"dibelah dua dan dicoba ulang"
                )
                mid = len(chunk) // 2
                self._score_chunk(chunk[:mid], slots[:mid], predictions)
                self._score_chunk(chunk[mid:], slots[mid:], predictions)
                return

            logger.error(f"Teks tunggal gagal dinilai: {exc}")
            predictions[slots[0]] = {
                'text': chunk[0],
                'sentiment': 'neutral',
                'confidence': 0.0,
                'scores': {'positive': 0.0, 'neutral': 0.0, 'negative': 0.0},
                'method': 'error',
                'error': str(exc),
            }

    def _predict_with_model(self, texts: List[str]) -> List[Dict]:
        """
        Prediksi sentimen dengan model — inferensi batch per chunk.

        Teks diurutkan berdasarkan panjang sebelum dipotong menjadi batch
        (length bucketing), lalu hasilnya dikembalikan ke urutan semula.

        Alasannya: padding dinamis mem-pad seluruh batch sepanjang teks
        TERPANJANG di dalamnya. Tanpa pengurutan, satu komentar panjang membuat
        255 teks pendek lainnya ikut dihitung pada panjang itu. Diukur pada
        331 klausa aspek, tahap ini menghabiskan 19,6 detik dari total 26 detik
        analisis.

        Urutan keluaran dijamin sama dengan urutan masukan.
        """
        predictions: List[Optional[Dict]] = [None] * len(texts)

        # Indeks diurutkan panjang; teks berukuran mirip dikelompokkan bersama.
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))

        for i in range(0, len(order), self.MAX_CHUNK):
            slots = order[i:i + self.MAX_CHUNK]
            self._score_chunk([texts[k] for k in slots], slots, predictions)

        # Tidak boleh ada lubang: setiap posisi harus terisi.
        return [p if p is not None else
                {'text': texts[i], 'sentiment': 'neutral', 'confidence': 0.0,
                 'scores': {'positive': 0.0, 'neutral': 0.0, 'negative': 0.0},
                 'method': 'error'}
                for i, p in enumerate(predictions)]

    # Daftar kata untuk jalur cadangan. Dipakai HANYA bila bobot IndoBERT gagal
    # dimuat sama sekali (mis. tidak ada jaringan pada kontainer baru).
    POSITIVE_WORDS = {
        'baik', 'bagus', 'hebat', 'mantap', 'keren', 'sempurna', 'memuaskan',
        'senang', 'suka', 'cinta', 'indah', 'cantik', 'ganteng', 'pintar',
        'cerdas', 'ramah', 'sopan', 'berkualitas', 'recommended', 'puas',
        'excellent', 'positif', 'gembira', 'sukses', 'berhasil', 'unggul',
        'terbaik', 'nyaman', 'murah', 'cepat', 'bersih', 'enak', 'lancar',
        'adil', 'jujur', 'aman', 'rapi', 'terjangkau', 'membantu',
    }

    NEGATIVE_WORDS = {
        'buruk', 'jelek', 'parah', 'mengecewakan', 'kecewa', 'benci', 'marah',
        'kesal', 'lambat', 'rusak', 'salah', 'gagal', 'error', 'bermasalah',
        'komplain', 'negatif', 'sedih', 'susah', 'sulit', 'ribet', 'payah',
        'mahal', 'kotor', 'lelet', 'curang', 'korupsi', 'bohong', 'menyiksa',
        'menyebalkan', 'terlambat', 'antri', 'macet', 'bangkrut', 'menderita',
    }

    # Negasi TIDAK dihitung sebagai kata negatif - ia membalik polaritas kata
    # sesudahnya. Versi sebelumnya memasukkan 'tidak' dan 'kurang' ke daftar
    # negatif, sehingga "tidak jelek" menghasilkan dua hitungan negatif dan
    # "tidak bagus" berakhir seri lalu dinilai netral.
    NEGATION_WORDS = {'tidak', 'tak', 'bukan', 'gak', 'nggak', 'enggak', 'ga',
                      'kagak', 'jangan', 'belum', 'tanpa', 'kurang'}

    # Jangkauan pembalikan setelah kata negasi. Bahasa Indonesia menempatkan
    # negasi tepat sebelum kata yang dinegasikan ("tidak begitu bagus"), jadi
    # dua token sudah menutup kasus lazimnya tanpa menyeret klausa berikutnya.
    NEGATION_SCOPE = 2

    def _predict_rule_based(self, texts: List[str]) -> List[Dict]:
        """
        Analisis sentimen berbasis daftar kata - jalur cadangan.

        Bukan pesaing model: akurasinya jauh di bawah IndoBERT (terukur pada
        SmSA test, lihat docs/METODOLOGI_ANALISIS_SENTIMEN.md). Tujuannya agar
        service tetap mengembalikan hasil yang masuk akal ketika bobot model
        tidak bisa dimuat, bukan melempar galat ke pengguna.

        Negasi ditangani dengan membalik polaritas maksimal dua kata setelahnya.
        """
        predictions: List[Dict[str, Any]] = []

        for text in texts:
            if not text or not str(text).strip():
                predictions.append(self._empty_prediction(text))
                continue

            words = str(text).lower().split()
            pos_count = 0
            neg_count = 0
            negate_until = -1

            for i, word in enumerate(words):
                # Tanda baca sudah dibersihkan pada jalur normal, tetapi
                # pemanggil bisa mengirim teks mentah.
                token = word.strip('.,!?;:"()[]')

                if token in self.NEGATION_WORDS:
                    negate_until = i + self.NEGATION_SCOPE
                    continue

                flipped = i <= negate_until
                if token in self.POSITIVE_WORDS:
                    neg_count += 1 if flipped else 0
                    pos_count += 0 if flipped else 1
                elif token in self.NEGATIVE_WORDS:
                    pos_count += 1 if flipped else 0
                    neg_count += 0 if flipped else 1

            total = pos_count + neg_count
            if pos_count > neg_count:
                sentiment = 'positive'
                confidence = min(0.6 + (pos_count * 0.1), 0.95)
            elif neg_count > pos_count:
                sentiment = 'negative'
                confidence = min(0.6 + (neg_count * 0.1), 0.95)
            else:
                sentiment = 'neutral'
                confidence = 0.5

            # `scores` disertakan agar bentuk hasil sama dengan jalur model;
            # antarmuka Laravel membaca kunci ini tanpa membedakan jalur.
            if total:
                scores = {
                    'positive': round(pos_count / total, 4),
                    'negative': round(neg_count / total, 4),
                }
                scores['neutral'] = round(max(0.0, 1.0 - scores['positive'] - scores['negative']), 4)
            else:
                scores = {'positive': 0.0, 'neutral': 1.0, 'negative': 0.0}

            predictions.append({
                'text': text,
                'sentiment': sentiment,
                'confidence': round(confidence, 4),
                'scores': {k: scores[k] for k in ('positive', 'neutral', 'negative')},
                'method': 'rule-based',
            })

        return predictions

    @staticmethod
    def _build_review_queue(predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Tandai prediksi yang paling layak dikoreksi manusia.

        Ini penghubung antara evaluasi dan loop active learning. Confidence
        model terbukti informatif - terukur pada SmSA test, akurasi 0,9847 pada
        pita >= 0,99 tetapi hanya 0,4340 di bawah 0,70 - sehingga meninjau baris
        berkeyakinan rendah jauh lebih produktif daripada meninjau acak.

        Indeksnya mengacu ke posisi pada `predictions`, yang sejajar dengan
        daftar `texts` milik pemanggil, sehingga Laravel bisa langsung memakainya
        untuk mengurutkan antrean koreksi tanpa memetakan ulang apa pun.

        Args:
            predictions: Seluruh prediksi, termasuk yang bertanda `empty`.

        Returns:
            Dict berisi ambang, jumlah, porsi, dan indeks yang perlu ditinjau
            (diurutkan dari yang paling tidak yakin).
        """
        threshold = settings.sentiment_review_threshold

        scorable = [
            (i, p) for i, p in enumerate(predictions) if p.get('method') != 'empty'
        ]
        flagged = [
            (i, p) for i, p in scorable if p.get('confidence', 0.0) < threshold
        ]
        flagged.sort(key=lambda pair: pair[1].get('confidence', 0.0))

        return {
            'threshold': threshold,
            'count': len(flagged),
            'share': round(len(flagged) / len(scorable), 4) if scorable else 0.0,
            # Dibatasi agar respons tidak membengkak pada korpus besar; urutan
            # menaik berarti yang paling meragukan selalu ikut terbawa.
            'indices': [i for i, _ in flagged[:200]],
        }

    def _calculate_distribution(self, predictions: List[Dict]) -> Dict:
        """Calculate sentiment distribution"""
        total = len(predictions)
        if total == 0:
            return {'positive': 0, 'neutral': 0, 'negative': 0}
        
        sentiments = [p['sentiment'] for p in predictions]
        
        return {
            'positive': round((sentiments.count('positive') / total) * 100, 2),
            'neutral': round((sentiments.count('neutral') / total) * 100, 2),
            'negative': round((sentiments.count('negative') / total) * 100, 2)
        }
    
    def _calculate_metrics(self, predictions: List[Dict]) -> Dict:
        """Calculate performance metrics"""
        confidences = [p['confidence'] for p in predictions if 'confidence' in p]
        
        return {
            'avg_confidence': round(np.mean(confidences), 4) if confidences else 0.0,
            'min_confidence': round(min(confidences), 4) if confidences else 0.0,
            'max_confidence': round(max(confidences), 4) if confidences else 0.0,
            'total_analyzed': len(predictions)
        }
    
    def _generate_summary(self, distribution: Dict, total: int) -> str:
        """Generate analysis summary"""
        dominant = max(distribution, key=distribution.get)
        percentage = distribution[dominant]
        
        sentiment_labels = {
            'positive': 'positif',
            'negative': 'negatif',
            'neutral': 'netral'
        }
        
        return (
            f"Dari {total} teks yang dianalisis, {percentage}% menunjukkan sentimen "
            f"{sentiment_labels[dominant]}. Distribusi lengkap: "
            f"{distribution['positive']}% positif, "
            f"{distribution['neutral']}% netral, "
            f"{distribution['negative']}% negatif."
        )

    # ── Retraining (active learning) ────────────────────────────────────────

    def _encode(self, texts: List[str], max_length: Optional[int] = None):
        """
        Tokenisasi batch teks menjadi tensor input_ids + attention_mask.

        Panjang padding DITURUNKAN dari data, bukan dipatok 512. Data koreksi
        berisi komentar pendek: pada korpus produksi panjang persentil ke-99
        hanya 116 token, sehingga padding ke 512 membuat setiap langkah maju
        dan mundur menghitung ~4x token semu. Self-attention berskala kuadratik
        terhadap panjang urutan, jadi biayanya jauh lebih besar dari sekadar 4x.

        Batas atas tetap 512 (batas posisi IndoBERT) dan dibulatkan ke kelipatan
        8 agar ramah terhadap kernel tensor.

        Args:
            texts: Teks yang akan ditokenisasi.
            max_length: Paksa panjang tertentu; None berarti diturunkan dari data.

        Returns:
            Tuple (input_ids, attention_mask).
        """
        if max_length is None:
            lengths = [len(self.tokenizer.encode(t, truncation=True, max_length=512))
                       for t in texts]
            longest = max(lengths) if lengths else 8
            max_length = min(512, max(8, ((longest + 7) // 8) * 8))
            logger.info(
                f"Panjang padding retraining: {max_length} token "
                f"(terpanjang {longest}, sebelumnya selalu 512)"
            )

        encoding = self.tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        return encoding['input_ids'], encoding['attention_mask']

    def _evaluate(self, input_ids, attention_masks, labels, batch_size: int = 16) -> Dict[str, Any]:
        """
        Jalankan inference pada satu set dan kembalikan metrik klasifikasi.
        Dipakai dua kali: sebelum dan sesudah fine-tuning, pada set validasi yang sama.
        """
        from torch.utils.data import DataLoader, TensorDataset

        self.model.eval()
        loader = DataLoader(TensorDataset(input_ids, attention_masks, labels), batch_size=batch_size)

        y_true: List[int] = []
        y_pred: List[int] = []

        with torch.no_grad():
            for batch in loader:
                b_ids = batch[0].to(self.device)
                b_mask = batch[1].to(self.device)
                outputs = self.model(input_ids=b_ids, attention_mask=b_mask)
                preds = torch.argmax(outputs.logits, dim=1).cpu().tolist()

                y_pred.extend(preds)
                y_true.extend(batch[2].cpu().tolist())

                del b_ids, b_mask, outputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        return classification_metrics(y_true, y_pred, self.label_map)

    def _collect_logits(self, input_ids, attention_masks, batch_size: int = 16):
        """
        Kumpulkan logit MENTAH (sebelum softmax) untuk satu set.

        Dipakai mencari suhu kalibrasi setelah retraining. Harus logit mentah,
        bukan probabilitas: temperature scaling bekerja dengan membagi logit,
        dan probabilitas yang sudah di-softmax tidak bisa dikembalikan tepat.
        """
        from torch.utils.data import DataLoader, TensorDataset

        self.model.eval()
        loader = DataLoader(TensorDataset(input_ids, attention_masks), batch_size=batch_size)
        chunks = []

        with torch.no_grad():
            for batch in loader:
                outputs = self.model(
                    input_ids=batch[0].to(self.device),
                    attention_mask=batch[1].to(self.device),
                )
                chunks.append(outputs.logits.detach().cpu())
                del outputs

        return torch.cat(chunks) if chunks else None

    async def retrain(
        self,
        training_data: List[Dict[str, Any]],
        epochs: int = 3,
        learning_rate: float = 2e-5,
        save_path: Optional[str] = None,
        force: bool = False
    ) -> Dict[str, Any]:
        """
        Pembungkus async tipis untuk `_retrain_sync`.

        Seluruh loop pelatihan PyTorch berjalan sinkron dan memakan waktu
        menit. Menjalankannya langsung di coroutine ini akan memblokir event
        loop, sehingga SETIAP permintaan lain ke layanan menggantung sampai
        pelatihan selesai - termasuk `/health` dan `/api/analyze/*`.

        Dulu tidak terlihat karena pelatihan hanya jalan saat admin menekan
        tombol, biasanya di saat sepi. Sejak sisi web memicu retraining
        otomatis begitu koreksi pengguna mencapai ambang, waktunya tidak lagi
        dipilih manusia dan bisa jatuh tepat ketika pengguna lain menganalisis.

        `asyncio.to_thread` memindahkan kerja beratnya ke thread lain. PyTorch
        melepaskan GIL selama operasi tensor, jadi inference tetap terlayani -
        lebih lambat karena berbagi CPU, tetapi tidak menggantung sama sekali.

        `lepas_memori` dipanggil SETELAH thread kembali, karena baru saat itu
        optimizer dan gradien milik frame `_retrain_sync` benar-benar bebas.
        Di `finally` supaya pelatihan yang gagal di tengah jalan pun tidak
        meninggalkan proses sebesar puncaknya.
        """
        try:
            return await asyncio.to_thread(
                self._retrain_sync,
                training_data,
                epochs,
                learning_rate,
                save_path,
                force,
            )
        finally:
            lepas_memori()

    def _retrain_sync(
        self,
        training_data: List[Dict[str, Any]],
        epochs: int = 3,
        learning_rate: float = 2e-5,
        save_path: Optional[str] = None,
        force: bool = False
    ) -> Dict[str, Any]:
        """
        Retrain sentiment model dengan data berlabel hasil koreksi user.

        Alur mengikuti fase Evaluation CRISP-ML(Q): ukur model lama pada set
        validasi, latih, ukur lagi pada set validasi yang sama, lalu checkpoint
        hanya disimpan bila tidak menurunkan weighted F1.

        Args:
            training_data: List of {text, label} dengan label positive|neutral|negative.
            epochs: Jumlah epoch fine-tuning.
            learning_rate: Learning rate AdamW.
            save_path: Lokasi simpan checkpoint; default settings.sentiment_checkpoint_path.
            force: Simpan checkpoint walaupun metriknya menurun.

        Returns:
            Dict berisi metrik sebelum/sesudah, distribusi label, dan status penyimpanan.
        """
        from torch.utils.data import DataLoader, TensorDataset
        from torch.optim import AdamW

        save_path = save_path or settings.sentiment_checkpoint_path

        # Seed dikunci sebelum apa pun yang acak (split, shuffle, dropout).
        set_seed(settings.retrain_seed)

        self._ensure_loaded()
        if not self.model or not self.tokenizer:
            raise ValueError("Model is not loaded")

        # Protokol kolam: latih dari bobot DASAR, bukan dari checkpoint putaran
        # sebelumnya. Lihat `settings.retrain_from_baseline` untuk alasannya.
        #
        # Aman dilakukan di sini: jalur inferensi memegang lock yang sama
        # (`_use_models`), jadi tidak ada permintaan yang membaca model selagi
        # bobotnya ditukar. Setelah pelatihan, model yang hidup adalah hasil
        # pelatihan itu sendiri.
        dari_dasar = settings.retrain_from_baseline and self.using_retrained

        # Yang dilaporkan ke pemanggil adalah "pelatihan ini BERANGKAT dari
        # bobot dasar", bukan "kami menukar bobotnya". Keduanya berbeda ketika
        # belum ada checkpoint sama sekali: `dari_dasar` False karena tidak ada
        # yang perlu ditukar, padahal bobot yang dilatih memang bobot dasar.
        #
        # Terjadi pada iterasi 2 studi kasus: `models/sentiment_retrained/`
        # tidak ada, pelatihan berangkat dari crypter70 (terbukti dari suhu
        # sebelumnya 2,7748 milik bobot dasar), tetapi catatannya berbunyi
        # `trained_from_baseline: false` - persis kebalikan dari yang terjadi.
        # Untuk skripsi ini jejak auditnya yang salah, bukan pelatihannya.
        mulai_dari_dasar = dari_dasar or not self.using_retrained
        if dari_dasar:
            logger.info(
                f"Protokol kolam: melatih dari bobot dasar {self.base_model_name}, "
                f"bukan melanjutkan {self.model_name}"
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.base_model_name, low_cpu_mem_usage=True,
                **hub_kwargs(self.base_model_name)
            )
            self.model.to(self.device)
            # Peta label milik bobot yang BARU dimuat - checkpoint hasil
            # retraining dan model dasar tidak dijamin memakai urutan yang sama.
            self.label_map = self._resolve_label_map()

        logger.info(f"Starting sentiment model retrain with {len(training_data)} samples")

        # ── Validasi & parsing data ─────────────────────────────────────────
        texts: List[str] = []
        label_ids: List[int] = []
        skipped: List[Dict[str, str]] = []

        for item in training_data:
            text = (item.get('text') or '').strip()
            label_str = item.get('label')

            if not text:
                skipped.append({'reason': 'empty_text', 'label': str(label_str)})
                continue
            if label_str not in self.label_to_id:
                skipped.append({'reason': 'unknown_label', 'label': str(label_str)})
                continue

            texts.append(text)
            label_ids.append(self.label_to_id[label_str])

        if not texts:
            raise ValueError("No valid training data after processing")

        distribution = label_distribution([self.label_map[i] for i in label_ids])
        logger.info(
            f"Distribusi label training: {distribution['counts']} "
            f"(imbalance ratio {distribution['imbalance_ratio']})"
        )

        # ── Split stratifikasi ──────────────────────────────────────────────
        train_idx, val_idx = stratified_split(
            label_ids, settings.retrain_val_ratio, settings.retrain_seed
        )
        if not val_idx:
            raise ValueError("Set validasi kosong - tambahkan data koreksi terlebih dahulu")

        input_ids, attention_masks = self._encode(texts)
        labels = torch.tensor(label_ids)

        train_ds = TensorDataset(
            input_ids[train_idx], attention_masks[train_idx], labels[train_idx]
        )
        val_input_ids = input_ids[val_idx]
        val_masks = attention_masks[val_idx]
        val_labels = labels[val_idx]

        # ── Baseline: ukur model saat ini sebelum disentuh ──────────────────
        metrics_before = self._evaluate(val_input_ids, val_masks, val_labels)
        logger.info(
            f"Metrik sebelum retraining: acc={metrics_before['accuracy']} "
            f"weighted_f1={metrics_before['weighted_f1']}"
        )

        # ── Bobot kelas melawan ketimpangan data koreksi ────────────────────
        weights = compute_class_weights([label_ids[i] for i in train_idx], len(self.label_map))
        weight_tensor = torch.tensor(weights, dtype=torch.float).to(self.device)
        loss_fn = torch.nn.CrossEntropyLoss(weight=weight_tensor)
        logger.info(f"Bobot kelas (positive, neutral, negative): {weights}")

        train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
        optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)

        train_losses: List[float] = []

        # ── Pemilihan epoch terbaik ─────────────────────────────────────────
        # Sebelumnya bobot epoch TERAKHIR yang dipakai. Pada data koreksi yang
        # kecil dan timpang, weighted F1 validasi lazim memuncak di epoch 1-2
        # lalu turun karena overfitting, sehingga menjalankan 3 epoch justru
        # membuang hasil terbaik yang sudah dicapai.
        #
        # Epoch terbaik disimpan ke DISK, bukan disalin ke RAM: menyimpan
        # state_dict IndoBERT base menambah ~500 MB, dan seluruh service memang
        # dirancang muat di kuota RAM Railway (alasan yang sama membuat rollback
        # dilakukan dengan memuat ulang dari disk, bukan menyalin bobot).
        best_dir = f"{save_path}.best"
        best_f1 = -1.0
        best_epoch = 0
        epoch_history: List[Dict[str, Any]] = []

        for epoch in range(epochs):
            self.model.train()
            total_train_loss = 0.0

            for batch in train_loader:
                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_labels = batch[2].to(self.device)

                self.model.zero_grad()

                outputs = self.model(input_ids=b_input_ids, attention_mask=b_input_mask)
                loss = loss_fn(outputs.logits, b_labels)
                total_train_loss += loss.item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                del b_input_ids, b_input_mask, b_labels, outputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            avg_train_loss = total_train_loss / len(train_loader) if len(train_loader) else 0.0
            train_losses.append(round(avg_train_loss, 4))

            epoch_metrics = self._evaluate(val_input_ids, val_masks, val_labels)
            epoch_history.append({
                'epoch': epoch + 1,
                'train_loss': round(avg_train_loss, 4),
                'val_accuracy': epoch_metrics['accuracy'],
                'val_weighted_f1': epoch_metrics['weighted_f1'],
            })
            logger.info(
                f"Epoch {epoch + 1}/{epochs} - Train loss: {avg_train_loss:.4f} "
                f"- Val acc: {epoch_metrics['accuracy']:.4f} "
                f"- Val weighted F1: {epoch_metrics['weighted_f1']:.4f}"
            )

            if epoch_metrics['weighted_f1'] > best_f1:
                best_f1 = epoch_metrics['weighted_f1']
                best_epoch = epoch + 1
                self._save_weights(best_dir)
                self.tokenizer.save_pretrained(best_dir)

        # Kembali ke epoch terbaik bila bukan yang terakhir. Perbandingan
        # terhadap model lama karena itu selalu memakai kandidat TERBAIK, bukan
        # kandidat kebetulan-terakhir.
        if best_epoch and best_epoch != epochs and os.path.isdir(best_dir):
            logger.info(
                f"Epoch terbaik adalah {best_epoch}/{epochs} "
                f"(weighted F1 {best_f1:.4f}); bobot dikembalikan ke sana."
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(best_dir)
            self.model.to(self.device)
            self.model.eval()

        metrics_after = self._evaluate(val_input_ids, val_masks, val_labels)

        # ── Kalibrasi ulang ─────────────────────────────────────────────────
        # Suhu adalah milik BOBOT tertentu. Setelah fine-tuning, distribusi
        # logit berubah, sehingga suhu model dasar tidak lagi berlaku - dan
        # memakainya membuat `review_queue` memilih baris yang salah.
        #
        # Suhu dicari pada set validasi yang sama, yaitu data yang tidak ikut
        # dilatih. Bila set itu terlalu kecil atau modelnya benar semua,
        # `fit_temperature` mengembalikan 1.0 (setara tanpa kalibrasi) alih-alih
        # angka ekstrem yang tidak bermakna.
        val_logits = self._collect_logits(val_input_ids, val_masks)
        fitted_temperature = fit_temperature(val_logits, val_labels)

        # "Tidak bisa menghitung T baru" TIDAK berarti "tanpa kalibrasi".
        #
        # `fit_temperature` mengembalikan 1.0 ketika kalibrasi tidak layak -
        # set validasi terlalu kecil, tidak ada kesalahan, atau hasilnya di luar
        # rentang wajar. Menuliskan 1.0 apa adanya MEMBUANG suhu yang sudah
        # terukur pada ribuan sampel, dan menggantinya dengan asumsi bahwa model
        # ini terkalibrasi sempurna - asumsi yang justru diketahui salah untuk
        # keluarga model ini (ECE mentah 0,0827 pada model dasar).
        #
        # Terukur pada studi kasus: retrain 218 koreksi menyisakan set validasi
        # 44 baris (<50), penjaga menyala, T 2,7748 -> 1,0, dan rerata keyakinan
        # melonjak 0,8929 -> 0,9497. Akibatnya antrean tinjauan menyusut dari
        # 109 menjadi 58 baris dan kesalahan yang tertangkap turun 66,7% -> 53,7%
        # - persis mekanisme yang seharusnya memberi makan iterasi berikutnya.
        #
        # Suhu lama memang belum tentu optimal untuk bobot yang baru, tetapi ia
        # angka terukur dari arsitektur yang sama; 1.0 adalah angka yang sudah
        # diketahui keliru. Diwarisi sampai ada cukup data untuk mengukur ulang.
        diwarisi = (
            fitted_temperature == 1.0
            and self.temperature != 1.0
            and len(val_idx) < settings.calibration_min_samples
        )
        new_temperature = self.temperature if diwarisi else fitted_temperature

        if diwarisi:
            logger.warning(
                f"Kalibrasi dilewati: set validasi {len(val_idx)} < "
                f"{settings.calibration_min_samples} sampel. Suhu lama "
                f"T={self.temperature} DIWARISI, bukan direset ke 1.0. "
                f"Kumpulkan lebih banyak koreksi untuk mengukur ulang."
            )
        else:
            logger.info(
                f"Suhu kalibrasi hasil retraining: T={new_temperature} "
                f"(sebelumnya {self.temperature})"
            )

        # ── Keputusan simpan / tolak ────────────────────────────────────────
        improved, delta = should_accept_checkpoint(
            metrics_before['weighted_f1'],
            metrics_after['weighted_f1'],
            settings.retrain_min_delta,
            force,
        )
        saved = False

        if improved:
            os.makedirs(save_path, exist_ok=True)
            self._save_weights(save_path)
            self.tokenizer.save_pretrained(save_path)
            self._save_calibration(save_path, new_temperature, {
                'fitted_on': ('diwarisi dari bobot sebelumnya (set validasi '
                              'terlalu kecil untuk mengukur ulang)'
                              if diwarisi else 'validation split retraining'),
                'inherited': diwarisi,
                'val_size': len(val_idx),
                'min_samples': settings.calibration_min_samples,
                'seed': settings.retrain_seed,
            })
            write_provenance(save_path, {
                'modul': 'sentiment',
                'bobot_awal': (self.base_model_name if mulai_dari_dasar
                               else self.model_name),
                'trained_from_baseline': mulai_dari_dasar,
                'jumlah_sampel': len(training_data),
                'sidik_jari_kolam': pool_fingerprint([d['text'] for d in training_data]),
                'distribusi_label': distribution,
                'val_size': len(val_idx),
                'epochs': epochs,
                'best_epoch': best_epoch,
                'weighted_f1_sebelum': metrics_before['weighted_f1'],
                'weighted_f1_sesudah': metrics_after['weighted_f1'],
                'temperature': new_temperature,
                'temperature_inherited': diwarisi,
                'seed': settings.retrain_seed,
            })
            saved = True

            # Hot reload dari checkpoint yang baru disimpan
            self.model_name = save_path
            self.using_retrained = True
            self._loaded = False
            self._ensure_loaded()

            logger.info(f"Checkpoint disimpan ke {save_path} (delta weighted F1 {delta:+})")
        else:
            # Bobot hasil training dibuang dengan memuat ulang dari sumber lama.
            # Checkpoint lama di disk tidak pernah ditimpa sebelum keputusan ini,
            # sehingga rollback cukup dengan reload - tanpa menyalin state_dict.
            logger.warning(
                f"Retraining menurunkan weighted F1 ({delta:+}); checkpoint TIDAK disimpan. "
                f"Model dikembalikan ke {self.model_name}."
            )
            self._loaded = False
            self._ensure_loaded()

        # Direktori sementara epoch terbaik tidak boleh tertinggal: ia berisi
        # salinan penuh bobot model dan akan menumpuk tiap kali retrain dijalankan.
        if os.path.isdir(best_dir):
            shutil.rmtree(best_dir, ignore_errors=True)

        return {
            'epochs_completed': epochs,
            'best_epoch': best_epoch,
            'epoch_history': epoch_history,
            'train_loss': train_losses,
            # val_loss dipertahankan demi kompatibilitas respons lama
            'val_loss': [],
            'val_accuracy': metrics_after['accuracy'],
            'model_path': save_path,
            'saved': saved,
            'rejected_for_regression': not saved,
            'seed': settings.retrain_seed,
            'samples_used': len(texts),
            'samples_skipped': len(skipped),
            'skipped_detail': skipped[:20],
            'label_distribution': distribution,
            'class_weights': {self.label_map[i]: w for i, w in enumerate(weights)},
            'train_size': len(train_idx),
            'val_size': len(val_idx),
            'metrics_before': metrics_before,
            'metrics_after': metrics_after,
            'weighted_f1_delta': delta,
            'temperature': self.temperature,
            'temperature_inherited': diwarisi,
            'trained_from_baseline': mulai_dari_dasar,
            'baseline_swapped': dari_dasar,
            'temperature_fitted': new_temperature,
        }
