from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import asyncio
import json
import time
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.config import SENTIMENT_MODELS, settings
from app.schemas.request_schemas import (
    TextAnalysisRequest,
    PreprocessRequest,
    HealthResponse,
    AspectRetrainRequest,
    SentimentRetrainRequest,
    AssociationRequest,
    AspectNormalizeRequest,
    RetrainPreviewRequest
)
from app.services.sentiment_service import SentimentService, LABEL_MAP, LABEL_TO_ID
from app.services.aspect_service import AspectService
from app.services.topic_service import TopicService
from app.services.association_service import AssociationService
from app.security import (
    JALUR_TERBUKA,
    PembatasLaju,
    kunci_sah,
    alamat_klien,
    verifikasi_startup,
    wajib_kunci_api,
)
from app.utils.logger import setup_logger
from app import weights_store
from app.utils.training import (
    compute_class_weights,
    label_distribution,
    stratified_split,
)

# Setup logger
logger = setup_logger(__name__)

# Global services
sentiment_service = None
aspect_service = None
topic_service = None
association_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for loading models on startup"""
    global sentiment_service, aspect_service, topic_service, association_service
    
    logger.info("🚀 Starting NLP API Service...")

    # Dijalankan SEBELUM model dimuat, dan sengaja di luar `try` di bawah:
    # konfigurasi yang tidak aman harus menghentikan start, bukan ditelan
    # penangan galat lalu berjalan tanpa autentikasi.
    verifikasi_startup()

    # Bobot ditarik SEBELUM service dibangun: tiap service menyelesaikan path
    # checkpoint-nya di konstruktor dan jatuh ke bobot publik bila berkasnya
    # belum ada. Menariknya setelah itu tidak akan terbaca.
    weights_store.unduh_bobot()

    try:
        # Initialize services
        logger.info("📦 Loading models...")
        sentiment_service = SentimentService()
        # SATU instance dipakai bersama. AspectService memakainya untuk menilai
        # polaritas per-aspek; membiarkannya membuat instance sendiri berarti
        # bobot sentimen dimuat dua kali (~500 MB tambahan) dan bobot hasil
        # retraining tidak pernah sampai ke jalur aspek.
        aspect_service = AspectService(sentiment_service=sentiment_service)
        topic_service = TopicService()
        association_service = AssociationService()
        
        logger.info("✅ All models loaded successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error loading models: {str(e)}")
        raise
    
    yield
    
    # Cleanup
    logger.info("🛑 Shutting down NLP API Service...")

# Create FastAPI app
# Skema OpenAPI disembunyikan di produksi: ia memberi tahu persis bentuk
# muatan /api/retrain/*, yang merupakan endpoint paling berbahaya di sini.
_docs = "/docs" if settings.expose_docs or settings.app_env != "production" else None
_redoc = "/redoc" if settings.expose_docs or settings.app_env != "production" else None

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="API untuk analisis teks menggunakan NLP dan Machine Learning",
    lifespan=lifespan,
    docs_url=_docs,
    redoc_url=_redoc,
    openapi_url=("/openapi.json" if _docs else None),
    # Dependency dipasang di tingkat aplikasi, bukan per endpoint.
    # Alasannya bukan kerapian melainkan kegagalan-tertutup: endpoint baru
    # otomatis terlindungi. Memasang `Depends` satu per satu berarti endpoint
    # yang ditambahkan enam bulan lagi terbuka sampai ada yang ingat - dan
    # daftar endpoint di sini sudah pernah kehilangan `except HTTPException`
    # di tujuh tempat karena persis pola itu.
    dependencies=[Depends(wajib_kunci_api)],
)

# CORS Middleware - dynamic to support Railway domains
def _build_allowed_origins() -> list:
    origins = list(settings.allowed_origins)
    # Add Railway/Laravel public domain from env if set
    laravel_url = os.getenv("LARAVEL_PUBLIC_URL")
    if laravel_url:
        origins.append(laravel_url.rstrip("/"))
    # Allow any Railway preview/production domain
    railway_domain = os.getenv("RAILWAY_PUBLIC_DOMAIN")
    if railway_domain:
        origins.extend([
            f"https://{railway_domain}",
            f"http://{railway_domain}",
        ])
    # In production, allow wildcard subdomains on .up.railway.app
    # Dibaca lewat settings agar satu sumber kebenaran; `settings.app_env`
    # sebelumnya terdefinisi tetapi tidak pernah dipakai sementara kode ini
    # membaca env langsung.
    if settings.app_env == "production":
        origins.append("https://*.up.railway.app")
    return origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=_build_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Batas laju ──────────────────────────────────────────────────────────────
_pembatas = PembatasLaju(settings.rate_limit_max, settings.rate_limit_window)


@app.middleware("http")
async def batasi_laju(request: Request, call_next):
    """
    Pagar terhadap satu pemanggil yang menghabiskan CPU.

    Healthcheck dikecualikan: platform hosting memanggilnya sesering yang ia
    mau, dan membuatnya kena 429 akan membuat kontainer dianggap mati lalu
    di-restart - mengubah pembatas laju menjadi penyebab downtime.

    Pembersihan dijalankan di sini, bukan lewat tugas latar, supaya tidak ada
    coroutine tambahan yang harus dihentikan saat shutdown.

    **Pemanggil dengan kunci sah tidak dibatasi.** Satu-satunya pemanggil sah
    adalah Laravel, dari SATU alamat IP, dan ia memecah analisis per 50 teks
    menjadi permintaan terpisah: analisis TNI (3.366 baris) berarti ~68
    permintaan sentimen dan ~68 aspek. Batas 30/menit per IP akan menjatuhkan
    analisis besar di tengah jalan dengan 429. Yang perlu dibatasi adalah orang
    luar - permintaan tanpa kunci atau dengan kunci salah tetap kena batas,
    jadi perlindungan dari penebakan kunci tidak berkurang.
    """
    if request.url.path in JALUR_TERBUKA:
        return await call_next(request)

    if kunci_sah(request.headers.get("x-api-key")):
        return await call_next(request)

    kunci = alamat_klien(request)
    if not _pembatas.izinkan(kunci):
        logger.warning(f"Batas laju terlampaui oleh {kunci} pada {request.url.path}")
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "detail": (
                    f"Terlalu banyak permintaan. Maksimum "
                    f"{settings.rate_limit_max} per {settings.rate_limit_window} detik."
                )
            },
            headers={"Retry-After": str(settings.rate_limit_window)},
        )

    _pembatas.bersihkan()
    return await call_next(request)

# Root endpoint
@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "message": f"{settings.app_name} is running",
        "version": settings.app_version,
        "timestamp": datetime.now().isoformat()
    }

# Health check
def _sentiment_profile() -> Dict[str, Any]:
    """Profil kandidat model sentimen yang sedang dipakai (kosong bila tak terdaftar)."""
    nama = getattr(sentiment_service, "base_model_name", None) or settings.sentiment_base_model
    return SENTIMENT_MODELS.get(nama, {})


def _identitas_bobot(berkas_provenance: Optional[str]) -> Dict[str, Any]:
    """
    Identitas bobot yang sedang hidup, dibaca dari `provenance.json`-nya.

    Path checkpoint saja tidak cukup: setiap retraining menimpa berkas yang
    sama, sehingga bobot iterasi 1 dan iterasi 2 sama-sama tampil sebagai
    `models/aspect_retrained.pt`. Terjadi 27 Sep: aspek dikembalikan ke
    iterasi 1, tetapi dari luar tidak ada yang membedakannya dari iterasi 2,
    dan halaman depan Laravel menampilkan angka model yang sudah tidak dipakai.
    `sidik_jari_kolam` sama persis dengan yang dicatat Laravel pada
    `model_trainings.result`, jadi keduanya bisa dicocokkan. None berarti
    bobot dasar (belum pernah dilatih ulang) atau catatan tidak terbaca.
    """
    kosong = {"pool_fingerprint": None, "samples": None, "restored_from": None}
    if not berkas_provenance or not os.path.isfile(berkas_provenance):
        return kosong
    try:
        with open(berkas_provenance, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return kosong
    dipulihkan = data.get("dipulihkan") or {}
    return {
        "pool_fingerprint": data.get("sidik_jari_kolam"),
        "samples": data.get("jumlah_sampel"),
        "restored_from": dipulihkan.get("dari_commit_hf") if isinstance(dipulihkan, dict) else None,
    }


def _identitas_sentimen() -> Dict[str, Any]:
    if not getattr(sentiment_service, "using_retrained", False):
        return _identitas_bobot(None)
    return _identitas_bobot(os.path.join(str(sentiment_service.model_name), "provenance.json"))


def _identitas_aspek() -> Dict[str, Any]:
    jalur = getattr(aspect_service, "_model_path", None)
    return _identitas_bobot(f"{jalur}.provenance.json" if jalur else None)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check"""
    sentimen = _identitas_sentimen()
    aspek = _identitas_aspek()
    return {
        "status": "healthy",
        "message": "All systems operational",
        "version": settings.app_version,
        "timestamp": datetime.now().isoformat(),
        "services": {
            "sentiment": sentiment_service is not None,
            "aspect": aspect_service is not None,
            "topic": topic_service is not None,
            "association": association_service is not None
        },
        # Terlihat dari Laravel apakah bobot hasil active learning benar-benar
        # dipakai, bukan diam-diam kembali ke model dasar setelah restart.
        "models": {
            "sentiment_source": getattr(sentiment_service, "model_name", None),
            "sentiment_retrained": getattr(sentiment_service, "using_retrained", False),
            # Provenance model sentimen. Beberapa kandidat sudah diukur dan
            # disimpan di SENTIMENT_MODELS agar bisa ditukar lewat konfigurasi
            # setelah evaluasi studi kasus; blok ini menyatakan mana yang hidup,
            # dari bobot dasar apa, dan dengan kalibrasi berapa - sehingga hasil
            # analisis lama bisa ditelusuri ke model yang menghasilkannya.
            "sentiment_base": _sentiment_profile().get("base"),
            "sentiment_temperature": getattr(sentiment_service, "temperature", None),
            "sentiment_review_threshold": settings.sentiment_review_threshold,
            "sentiment_label_map": getattr(sentiment_service, "label_map", None),
            "aspect_checkpoint": getattr(aspect_service, "_model_path", None),
            "aspect_retrained": bool(getattr(aspect_service, "_model_path", None)),
            # Bobot MANA yang hidup - lihat `_identitas_bobot`.
            "sentiment_pool_fingerprint": sentimen["pool_fingerprint"],
            "sentiment_pool_samples": sentimen["samples"],
            "aspect_pool_fingerprint": aspek["pool_fingerprint"],
            "aspect_pool_samples": aspek["samples"],
            "aspect_restored_from": aspek["restored_from"],
        },
        # Bobot dimuat MALAS: permintaan pertama tiap jenis membayar biaya unduh
        # dan muat model, yang di kontainer baru bisa memakan menit. Tanpa blok
        # ini Laravel tidak punya cara membedakan "service lambat" dari "service
        # sedang memuat model", sehingga cold start tampak seperti timeout.
        # Laravel bisa memanggil POST /api/warmup lebih dulu, atau menampilkan
        # keterangan yang benar kepada pengguna.
        "weights_loaded": {
            "sentiment": bool(getattr(sentiment_service, "_loaded", False)),
            "aspect": bool(getattr(aspect_service, "_loaded", False)),
            "topic": bool(getattr(topic_service, "_loaded", False)),
        }
    }

@app.post("/api/warmup")
async def warmup():
    """
    Muat bobot seluruh model lebih dulu, sebelum permintaan pengguna pertama.

    Bobot sengaja dimuat malas agar RAM saat start tetap rendah (syarat kuota
    Railway), tetapi akibatnya permintaan PERTAMA tiap jenis membayar biaya
    unduh dan muat model - di kontainer baru bisa memakan menit, dan dari sisi
    Laravel itu tidak bisa dibedakan dari analisis yang menggantung.

    Panggil endpoint ini setelah deploy, atau sebelum menjalankan analisis besar.
    Aman dipanggil berulang: `_ensure_loaded()` langsung kembali bila sudah
    dimuat, dan model yang gagal dimuat tetap menempuh jalur cadangan alih-alih
    melempar galat.

    Returns:
        Status per model beserta lama proses pemuatan.
    """
    import time

    hasil: Dict[str, Any] = {}
    mulai_total = time.time()

    for nama, service in (
        ("sentiment", sentiment_service),
        ("aspect", aspect_service),
        ("topic", topic_service),
    ):
        if service is None:
            hasil[nama] = {"loaded": False, "error": "service tidak tersedia"}
            continue

        mulai = time.time()
        try:
            if hasattr(service, "_ensure_loaded"):
                service._ensure_loaded()
            hasil[nama] = {
                "loaded": bool(getattr(service, "_loaded", False)),
                "seconds": round(time.time() - mulai, 1),
            }
        except Exception as exc:  # noqa: BLE001
            # Pemanasan tidak boleh menjatuhkan service: jalur analisis punya
            # rantai cadangan sendiri dan tetap bisa berjalan.
            logger.error(f"Warmup {nama} gagal: {exc}")
            hasil[nama] = {"loaded": False, "error": str(exc),
                           "seconds": round(time.time() - mulai, 1)}

    return {
        "status": "success",
        "models": hasil,
        "total_seconds": round(time.time() - mulai_total, 1),
    }


# Preprocessing endpoint
@app.post("/api/preprocess")
async def preprocess_text(request: PreprocessRequest):
    """
    Preprocess text data
    
    - **texts**: List of texts to preprocess
    - **config**: Preprocessing configuration
    """
    try:
        from app.preprocessing.text_cleaner import PROFILES, TextCleaner

        if request.task:
            cleaner = TextCleaner.for_task(request.task, request.config)
            # Profil 'span' (aspek) memang tidak membersihkan apa pun: ekstraksi
            # berbasis offset karakter, sehingga teks harus utuh.
            preprocessed = (
                list(request.texts) if cleaner is None
                else cleaner.clean_texts(request.texts)
            )
        else:
            cleaner = TextCleaner(config=request.config)
            preprocessed = cleaner.clean_texts(request.texts)

        return {
            "status": "success",
            "preprocessed": preprocessed,
            "original_count": len(request.texts),
            "processed_count": len(preprocessed),
            # Pengguna bisa melihat opsi mana yang ditimpa kebijakan modul,
            # sehingga pratinjaunya jujur alih-alih diam-diam berbeda.
            "task": request.task,
            "applied_policy": PROFILES.get(request.task) if request.task else None,
        }
        
    except HTTPException:
        # 503/409 yang kita bangkitkan sendiri harus sampai ke pemanggil
        # apa adanya. Tanpa ini `except Exception` di bawah menelannya dan
        # mengubahnya jadi 500, sehingga Laravel tidak bisa membedakan
        # "layanan belum siap" (bisa dicoba lagi) dari kegagalan sungguhan.
        raise
    except Exception as e:
        logger.error(f"Preprocessing error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Preprocessing failed: {str(e)}"
        )

# Sentiment Analysis
@app.post("/api/normalize/aspects")
async def normalize_aspects(request: AspectNormalizeRequest):
    """
    Kembalikan bentuk dasar tiap nama aspek, memakai stemmer yang sama persis
    dengan yang dipakai ekstraksi aspek.

    Ada demi evaluasi hold-out. `document_aspects` mode automatic berisi bentuk
    permukaan seperti yang tertulis di teks (`pelayanannya`, `pajaknya`),
    sementara anotator manusia menulis bentuk dasar (`pelayanan`, `pajak`).
    Bahasa Indonesia aglutinatif, jadi membandingkan keduanya sebagai string
    persis menghitung pasangan yang BENAR sebagai false positive sekaligus
    false negative. Terukur pada 6 kalimat / 8 aspek emas dari keluaran yang
    sama: micro-F1 0,118 dengan string persis, 0,941 setelah dinormalkan.

    Respons analisis aspek sudah membawa `document_aspects_normalized`; endpoint
    ini melengkapi sisi lainnya, yaitu label emas. Cukup dipanggil sekali saat
    golden dataset diimpor - datasetnya beku, jadi hasilnya bisa disimpan.

    Tidak memuat bobot model apa pun (hanya Sastrawi), jadi murah dan tidak
    ikut mengantre di belakang retraining.
    """
    if not aspect_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Aspect service not available"
        )

    try:
        hasil = {
            nama: aspect_service.normalize_aspect_name(nama)
            for nama in request.aspects
        }

        return {
            "status": "success",
            "results": {
                "normalized": hasil,
                "aspects": [hasil[nama] for nama in request.aspects],
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Aspect normalize error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Aspect normalize failed: {str(e)}"
        )


# ── Penjaga pemakaian bobot model ────────────────────────────────────────────
# Satu lock per model, dipakai BERSAMA oleh jalur inference dan jalur pelatihan.
#
# Dua masalah berbeda yang ditutup oleh lock yang sama:
#
# 1. Dua retraining serentak akan sama-sama memuat model, sama-sama melatih, dan
#    sama-sama menulis checkpoint ke lokasi yang SAMA - hasilnya tidak dapat
#    diprediksi dan pemakaian RAM-nya berlipat, cukup untuk membuat container
#    Railway dimatikan. Permintaan kedua dibalas 409, tidak diantrekan.
#
# 2. Sejak pelatihan dipindah ke thread terpisah (agar tidak memblokir event
#    loop), pelatihan dan inference bisa berjalan BERSAMAAN di atas objek model
#    yang sama. `_retrain_sync` memanggil `self.model.train()` - yang menyalakan
#    dropout - lalu memperbarui bobotnya batch demi batch. Analisis yang
#    kebetulan jatuh di tengah itu akan membaca bobot setengah terlatih dengan
#    dropout aktif, dan mengembalikan angka yang salah TANPA galat apa pun.
#    Persis kelas kegagalan diam-diam yang dihindari di seluruh layanan ini.
#
# Kenapa `asyncio.Lock` dan bukan `threading.Lock`: inference berjalan di event
# loop, pelatihan di thread pekerja. Lock threading akan MEMBLOKIR event loop
# ketika pelatihan sedang memegangnya, sehingga /health menggantung lagi - yaitu
# masalah yang baru saja diperbaiki. `asyncio.Lock` membuat coroutine inference
# menunggu tanpa memblokir loop, jadi /health tetap menjawab seketika.
#
# Biaya nyatanya nol: inference memang sudah berjalan berurutan di event loop
# (badan `analyze()` seluruhnya sinkron), jadi lock ini tidak menambah antrean
# yang belum ada. Yang ditambahkan hanyalah penundaan analisis selama pelatihan
# berlangsung - jauh lebih baik daripada hasil yang salah diam-diam.
#
# Lock per proses sudah cukup KARENA layanan wajib jalan `--workers 1` (dipatok
# di Procfile, nixpacks.toml, dan railway.json), alasan yang sama dengan kenapa
# bobot model disimpan di global per proses.
#
# Topic tidak punya lock: modelnya tidak pernah dilatih ulang lewat API.
_model_locks: Dict[str, asyncio.Lock] = {
    "aspect": asyncio.Lock(),
    "sentiment": asyncio.Lock(),
}


# Model yang sedang dilatih, beserta kapan mulai dan perkiraan lamanya.
# Diisi oleh endpoint retraining SELAMA ia memegang lock modelnya.
_pelatihan: Dict[str, Dict[str, float]] = {}


def _detik_per_sampel(model: str) -> float:
    return (settings.retrain_sec_per_sample_sentiment if model == "sentiment"
            else settings.retrain_sec_per_sample_aspect)


@asynccontextmanager
async def _sedang_melatih(model: str, jumlah_sampel: int):
    """Tandai `model` sedang dilatih selama blok berjalan."""
    _pelatihan[model] = {
        "mulai": time.monotonic(),
        "perkiraan": jumlah_sampel * _detik_per_sampel(model),
    }
    try:
        yield
    finally:
        _pelatihan.pop(model, None)


def _tolak_bila_sedang_dilatih(names) -> None:
    """
    Balas 503 + `Retry-After` bila analisis butuh model yang sedang dilatih.

    Menunggu lock hanya benar bila tunggunya singkat. Pelatihan memakan
    belasan sampai puluhan menit, sementara Laravel menyerah setelah 600
    detik - dan menyerahnya klien TIDAK membatalkan penantian di sini.
    Terukur dengan server sungguhan: klien timeout, lalu analisisnya tetap
    dijalankan begitu lock lepas, tanpa ada yang menunggu hasilnya. Setiap
    penyerahan jadi pekerjaan hantu yang menumpuk tepat sesudah pelatihan,
    sementara Laravel mengirim salinan baru.

    Hanya pelatihan yang ditolak. Analisis yang menunggu analisis lain tetap
    mengantre, karena itu memang singkat.

    `Retry-After` diperkirakan dari jumlah sampel x detik per sampel, dibatasi
    30-600 detik. Perkiraannya boleh meleset: mencoba terlalu cepat hanya
    menghasilkan 503 lagi, yang murah.

    Mati secara bawaan (`retrain_reject_inference`). Nyalakan hanya bila
    pemanggil membaca `Retry-After` dan mengantrekan ulang - tanpa itu,
    analisis yang tadinya berhasil menunggu pelatihan singkat justru gagal.
    """
    if not settings.retrain_reject_inference:
        return
    sibuk = [n for n in set(names) if n in _pelatihan]
    if not sibuk:
        return
    sekarang = time.monotonic()
    sisa = max(
        _pelatihan[n]["mulai"] + _pelatihan[n]["perkiraan"] - sekarang for n in sibuk
    )
    tunggu = int(min(max(sisa, 30), 600))
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail=(
            f"Model {', '.join(sorted(sibuk))} sedang dilatih ulang; "
            f"coba lagi dalam {tunggu} detik"
        ),
        headers={"Retry-After": str(tunggu)},
    )


@asynccontextmanager
async def _use_models(*names: str):
    """Pegang bobot model yang disebut selama blok berjalan.

    Lock diambil dalam urutan abjad yang TETAP. Itu yang mencegah deadlock:
    analisis aspek butuh 'aspect' lalu 'sentiment' (AspectService memakai
    SentimentService bersama untuk polaritas per aspek), sementara tiap jalur
    retraining hanya butuh satu. Tanpa urutan tetap, dua jalur yang mengambil
    lock dengan urutan berbeda bisa saling menunggu selamanya.
    """
    _tolak_bila_sedang_dilatih(names)
    diambil = []
    try:
        for nama in sorted(set(names)):
            await _model_locks[nama].acquire()
            diambil.append(nama)
        yield
    finally:
        for nama in reversed(diambil):
            _model_locks[nama].release()


def _claim_retrain_slot(model: str) -> asyncio.Lock:
    """Balas 409 bila model ini sedang dipakai eksklusif - bukan mengantre.

    Mengantre akan membuat permintaan kedua menggantung selama menit-menit,
    lalu melatih dengan data yang sudah usang. `RetrainModel` di Laravel
    menangani respons non-2xx dengan rapi: statusnya dicatat gagal beserta
    pesannya, dan tidak ada pengulangan otomatis (`$tries = 1`).

    Pemeriksaan `locked()` lalu `async with` aman tanpa race: jalur cepat
    `asyncio.Lock.acquire()` tidak pernah menyerahkan kendali ketika lock
    sedang bebas, jadi tidak ada coroutine lain yang bisa menyelip di antaranya.
    """
    lock = _model_locks[model]
    if lock.locked():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Pelatihan model {model} sedang berjalan; coba lagi nanti",
        )
    return lock


@app.post("/api/analyze/sentiment")
async def analyze_sentiment(request: TextAnalysisRequest):
    """
    Analyze sentiment of texts
    
    - **texts**: List of texts to analyze
    - **preprocessing_config**: Optional preprocessing configuration
    """
    try:
        if not sentiment_service:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Sentiment service not available"
            )
        
        async with _use_models('sentiment'):
            result = await sentiment_service.analyze(
                texts=request.texts,
                preprocessing_config=request.preprocessing_config
            )
        
        return {
            "status": "success",
            "analysis_type": "sentiment",
            "results": result
        }
        
    except HTTPException:
        # 503/409 yang kita bangkitkan sendiri harus sampai ke pemanggil
        # apa adanya. Tanpa ini `except Exception` di bawah menelannya dan
        # mengubahnya jadi 500, sehingga Laravel tidak bisa membedakan
        # "layanan belum siap" (bisa dicoba lagi) dari kegagalan sungguhan.
        raise
    except Exception as e:
        logger.error(f"Sentiment analysis error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Sentiment analysis failed: {str(e)}"
        )

# Aspect Extraction
@app.post("/api/analyze/aspect")
async def analyze_aspect(request: TextAnalysisRequest):
    """
    Extract aspects from texts
    
    - **texts**: List of texts to analyze
    - **preprocessing_config**: Optional preprocessing configuration
    - **predefined_aspects**: Optional list of aspects to look for (rule-based)
    - **mode**: 'automatic' or 'rule-based'
    """
    try:
        if not aspect_service:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Aspect service not available"
            )
        
        async with _use_models('aspect', 'sentiment'):
            result = await aspect_service.analyze(
                texts=request.texts,
                preprocessing_config=request.preprocessing_config,
                predefined_aspects=request.predefined_aspects,
                mode=request.mode
            )
        
        return {
            "status": "success",
            "analysis_type": "aspect",
            "results": result
        }
        
    except HTTPException:
        # 503/409 yang kita bangkitkan sendiri harus sampai ke pemanggil
        # apa adanya. Tanpa ini `except Exception` di bawah menelannya dan
        # mengubahnya jadi 500, sehingga Laravel tidak bisa membedakan
        # "layanan belum siap" (bisa dicoba lagi) dari kegagalan sungguhan.
        raise
    except Exception as e:
        logger.error(f"Aspect analysis error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Aspect analysis failed: {str(e)}"
        )

# Topic Modeling
@app.post("/api/analyze/topic")
async def analyze_topic(request: TextAnalysisRequest):
    """
    Identify topics from texts
    
    - **texts**: List of texts to analyze
    - **preprocessing_config**: Optional preprocessing configuration
    - **num_topics**: Number of topics to identify (default: 5)
    """
    try:
        if not topic_service:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Topic service not available"
            )
        
        result = await topic_service.analyze(
            texts=request.texts,
            preprocessing_config=request.preprocessing_config,
            num_topics=(
                # `or 5` menelan nilai 0, padahal 0 berarti "pilih otomatis".
                # Akibatnya permintaan mode otomatis diam-diam dijalankan
                # sebagai 5 topik - terukur: hasilnya identik persis dengan
                # num_topics=5, sehingga bug ini tidak terlihat dari luar.
                5 if request.num_topics is None else request.num_topics
            )
        )
        
        return {
            "status": "success",
            "analysis_type": "topic",
            "results": result
        }
        
    except HTTPException:
        # 503/409 yang kita bangkitkan sendiri harus sampai ke pemanggil
        # apa adanya. Tanpa ini `except Exception` di bawah menelannya dan
        # mengubahnya jadi 500, sehingga Laravel tidak bisa membedakan
        # "layanan belum siap" (bisa dicoba lagi) dari kegagalan sungguhan.
        raise
    except Exception as e:
        logger.error(f"Topic analysis error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Topic analysis failed: {str(e)}"
        )

# Combined Analysis
@app.post("/api/analyze/combined")
async def analyze_combined(request: TextAnalysisRequest):
    """
    Perform combined analysis (sentiment + aspect + topic)

    - **texts**: List of texts to analyze
    - **preprocessing_config**: Optional preprocessing configuration
    - **predefined_aspects**: Kategori aspek untuk mode rule-based
    - **mode**: "automatic" atau "rule-based"

    Catatan: predefined_aspects/mode dulu diterima skema tapi TIDAK diteruskan
    ke aspect_service, sehingga analisis gabungan selalu memakai mode automatic
    dan daftar aspek yang dideklarasikan pengguna dibuang diam-diam - padahal
    antarmuka Laravel menampilkan pilihan itu juga untuk tipe combined.
    """
    try:
        if not all([sentiment_service, aspect_service, topic_service]):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="One or more services not available"
            )
        
        # Kedua bobot dipegang untuk SELURUH rangkaian, bukan per panggilan.
        # Analisis gabungan menyimpan asosiasi aspek-topik yang dihitung dari
        # ketiga hasil sekaligus; bila retraining menyelip di antara panggilan
        # sentimen dan aspek, satu respons akan memuat dua model berbeda.
        async with _use_models('aspect', 'sentiment'):
            sentiment_result = await sentiment_service.analyze(
                texts=request.texts,
                preprocessing_config=request.preprocessing_config
            )

            aspect_result = await aspect_service.analyze(
                texts=request.texts,
                preprocessing_config=request.preprocessing_config,
                predefined_aspects=request.predefined_aspects,
                mode=request.mode or "automatic"
            )

        topic_result = await topic_service.analyze(
            texts=request.texts,
            preprocessing_config=request.preprocessing_config,
            num_topics=(
                # `or 5` menelan nilai 0, padahal 0 berarti "pilih otomatis".
                # Akibatnya permintaan mode otomatis diam-diam dijalankan
                # sebagai 5 topik - terukur: hasilnya identik persis dengan
                # num_topics=5, sehingga bug ini tidak terlihat dari luar.
                5 if request.num_topics is None else request.num_topics
            )
        )
        
        association_result = None
        if association_service and "document_aspects" in aspect_result and "document_topics" in topic_result:
            association_result = association_service.analyze(
                document_aspects=aspect_result["document_aspects"],
                document_topics=topic_result["document_topics"],
                num_topics=topic_result.get("num_topics", 0),
                min_mentions=3
            )
        
        return {
            "status": "success",
            "analysis_type": "combined",
            "results": {
                "sentiment": sentiment_result,
                "aspect": aspect_result,
                "topic": topic_result,
                "association": association_result
            }
        }
        
    except HTTPException:
        # 503/409 yang kita bangkitkan sendiri harus sampai ke pemanggil
        # apa adanya. Tanpa ini `except Exception` di bawah menelannya dan
        # mengubahnya jadi 500, sehingga Laravel tidak bisa membedakan
        # "layanan belum siap" (bisa dicoba lagi) dari kegagalan sungguhan.
        raise
    except Exception as e:
        logger.error(f"Combined analysis error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Combined analysis failed: {str(e)}"
        )

@app.post("/api/analyze/association")
async def analyze_association(request: AssociationRequest):
    """
    Hitung asosiasi aspek-topik (PMI) dari hasil analisis yang sudah ada.

    Endpoint terpisah ini dipakai Laravel untuk dataset besar, di mana
    sentiment/aspect/topic dijalankan per batch sehingga /api/analyze/combined
    tidak bisa menghitung PMI-nya sendiri.
    """
    try:
        if not association_service:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Association service not available"
            )

        result = association_service.analyze(
            document_aspects=request.document_aspects,
            document_topics=request.document_topics,
            num_topics=request.num_topics,
            min_mentions=request.min_mentions or 3
        )

        return {
            "status": "success",
            "analysis_type": "association",
            "results": result
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Association analysis error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Association analysis failed: {str(e)}"
        )


# Retrain Endpoints
@app.post("/api/retrain/aspect")
async def retrain_aspect(request: AspectRetrainRequest):
    """
    Retrain aspect extraction model with labeled data.
    Data format: [{text: str, aspects: [str, ...]}, ...]
    Auto-generates BIO tags internally.
    """
    if not aspect_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Aspect service not available"
        )

    lock = _claim_retrain_slot("aspect")
    try:
        async with lock, _sedang_melatih("aspect", len(request.training_data)):
            result = await aspect_service.retrain(
                training_data=request.training_data,
                epochs=request.epochs,
                learning_rate=request.learning_rate,
                force=request.force
            )

        # Unggah HANYA bila checkpoint benar-benar diterima. Pelatihan yang
        # ditolak (`saved: False`) tidak menulis bobot baru, jadi mengunggah
        # akan mendorong checkpoint LAMA sebagai commit baru dan membuat
        # riwayat Hub berbohong tentang apa yang berubah.
        if result.get('saved'):
            await asyncio.to_thread(
                weights_store.unggah_bobot,
                'aspect',
                f"Retrain aspek: {len(request.training_data)} sampel",
            )

        return {
            "status": "success",
            "results": result
        }
    except HTTPException:
        # 503/409 yang kita bangkitkan sendiri harus sampai ke pemanggil
        # apa adanya. Tanpa ini `except Exception` di bawah menelannya dan
        # mengubahnya jadi 500, sehingga Laravel tidak bisa membedakan
        # "layanan belum siap" (bisa dicoba lagi) dari kegagalan sungguhan.
        raise
    except Exception as e:
        logger.error(f"Aspect retrain error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Aspect retrain failed: {str(e)}"
        )

@app.post("/api/retrain/sentiment")
async def retrain_sentiment(request: SentimentRetrainRequest):
    """
    Retrain sentiment model with labeled data.
    Data format: [{text: str, label: positive|neutral|negative}, ...]
    """
    if not sentiment_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Sentiment service not available"
        )

    lock = _claim_retrain_slot("sentiment")
    try:
        async with lock, _sedang_melatih("sentiment", len(request.training_data)):
            result = await sentiment_service.retrain(
                training_data=request.training_data,
                epochs=request.epochs,
                learning_rate=request.learning_rate,
                force=request.force
            )

        # Unggah HANYA bila checkpoint benar-benar diterima. Pelatihan yang
        # ditolak (`saved: False`) tidak menulis bobot baru, jadi mengunggah
        # akan mendorong checkpoint LAMA sebagai commit baru dan membuat
        # riwayat Hub berbohong tentang apa yang berubah.
        if result.get('saved'):
            await asyncio.to_thread(
                weights_store.unggah_bobot,
                'sentiment',
                f"Retrain sentimen: {len(request.training_data)} sampel",
            )

        return {
            "status": "success",
            "results": result
        }
    except HTTPException:
        # 503/409 yang kita bangkitkan sendiri harus sampai ke pemanggil
        # apa adanya. Tanpa ini `except Exception` di bawah menelannya dan
        # mengubahnya jadi 500, sehingga Laravel tidak bisa membedakan
        # "layanan belum siap" (bisa dicoba lagi) dari kegagalan sungguhan.
        raise
    except Exception as e:
        logger.error(f"Sentiment retrain error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Sentiment retrain failed: {str(e)}"
        )

# Retrain Preview
@app.post("/api/retrain/preview")
async def retrain_preview(request: RetrainPreviewRequest):
    """
    Laporkan kesiapan data koreksi TANPA melatih model.

    Fine-tuning pada data yang sangat timpang bisa menghasilkan model yang hanya
    menebak kelas mayoritas, dengan akurasi yang justru terlihat naik. Endpoint
    ini dipanggil sebelum retraining agar komposisi data terlihat lebih dulu.

    - **model_type**: 'sentiment' atau 'aspect'
    - **training_data**: payload yang sama dengan endpoint retraining terkait
    """
    try:
        if request.model_type == "sentiment":
            labels = [
                item.get("label") for item in request.training_data
                if item.get("text") and item.get("label") in LABEL_TO_ID
            ]
            invalid = len(request.training_data) - len(labels)
            distribution = label_distribution(labels)

            ids = [LABEL_TO_ID[l] for l in labels]
            train_idx, val_idx = stratified_split(
                ids, settings.retrain_val_ratio, settings.retrain_seed
            )

            val_counts: Dict[str, int] = {}
            for i in val_idx:
                name = LABEL_MAP[ids[i]]
                val_counts[name] = val_counts.get(name, 0) + 1

            weights = compute_class_weights([ids[i] for i in train_idx], len(LABEL_MAP))

            # Akurasi yang dicapai hanya dengan menebak kelas mayoritas terus.
            majority_baseline = (
                max(distribution["counts"].values()) / distribution["total"]
                if distribution["total"] else 0.0
            )

            return {
                "status": "success",
                "model_type": "sentiment",
                "samples_valid": len(labels),
                "samples_invalid": invalid,
                "label_distribution": distribution,
                "split": {
                    "seed": settings.retrain_seed,
                    "train_size": len(train_idx),
                    "val_size": len(val_idx),
                    "val_composition": val_counts,
                },
                "class_weights": dict(zip(LABEL_MAP.values(), weights)),
                "majority_baseline_accuracy": round(majority_baseline, 4),
                "warnings": _sentiment_warnings(distribution, val_counts),
            }

        if not aspect_service:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Aspect service not available"
            )

        # Aspek: pakai jalur persiapan yang sama dengan retraining sebenarnya,
        # sehingga laporan cocok dengan apa yang nanti benar-benar dilatih.
        aspect_service._ensure_loaded()
        _, _, _, report = aspect_service._prepare_bio_dataset(request.training_data)

        return {
            "status": "success",
            "model_type": "aspect",
            "bio_report": report,
            "warnings": _aspect_warnings(report),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Retrain preview error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retrain preview failed: {str(e)}"
        )


def _sentiment_warnings(distribution: Dict[str, Any], val_counts: Dict[str, int]) -> List[str]:
    """Peringatan yang layak ditampilkan ke admin sebelum menekan tombol retrain."""
    warnings: List[str] = []

    if distribution["total"] == 0:
        return ["Tidak ada sampel berlabel valid."]

    if distribution["imbalance_ratio"] >= 10:
        warnings.append(
            f"Data sangat timpang (rasio {distribution['imbalance_ratio']}:1). "
            f"Model berisiko hanya menebak kelas '{distribution['majority_class']}'."
        )

    for label, count in distribution["counts"].items():
        if count < 10:
            warnings.append(
                f"Kelas '{label}' hanya punya {count} sampel — terlalu sedikit untuk dilatih."
            )

    for label, count in val_counts.items():
        if count < 5:
            warnings.append(
                f"Set validasi hanya berisi {count} sampel '{label}'; "
                f"metrik untuk kelas ini tidak dapat dipercaya."
            )

    return warnings


def _aspect_warnings(report: Dict[str, Any]) -> List[str]:
    """Peringatan kualitas pelabelan BIO."""
    warnings: List[str] = []

    if report["aspects_unmatched_pct"] >= 10:
        warnings.append(
            f"{report['aspects_unmatched_pct']}% anotasi aspek tidak ditemukan persis "
            f"di teksnya (mis. koreksi 'pelayanan' untuk kalimat 'layanannya')."
        )

    if report["samples_all_O"]:
        warnings.append(
            f"{report['samples_all_O']} sampel tidak punya satu pun aspek yang cocok, "
            f"sehingga akan mengajarkan model bahwa kalimat itu tidak beraspek."
        )

    if report["samples_used"] < 10:
        warnings.append("Sampel valid kurang dari 10; retraining akan ditolak.")

    return warnings


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    # Header ikut diteruskan. Tanpa ini `Retry-After` pada 503 (lihat
    # `_tolak_bila_sedang_dilatih`) hilang di jalan dan pemanggil tidak tahu
    # kapan harus mencoba lagi - penangan ini dulu membangun JSONResponse baru
    # dan membuang `exc.headers` diam-diam.
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "message": exc.detail,
            "timestamp": datetime.now().isoformat()
        },
        headers=getattr(exc, "headers", None),
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "status": "error",
            "message": "Internal server error",
            "timestamp": datetime.now().isoformat()
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )