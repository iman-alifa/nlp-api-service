from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import List, Optional, Dict, Any

from app.config import settings

class PreprocessingConfig(BaseModel):
    """Preprocessing configuration"""
    case_folding: bool = True
    remove_punctuation: bool = True
    remove_numbers: bool = False
    remove_stopwords: bool = True
    stemming: bool = True
    lemmatization: bool = False
    custom_stopwords: List[str] = []

class PreprocessRequest(BaseModel):
    """Request schema for preprocessing"""
    texts: List[str] = Field(..., min_length=1, max_length=settings.max_batch_size)
    config: Optional[PreprocessingConfig] = None
    # Pratinjau memakai kebijakan modul analisis tertentu, bukan konfigurasi
    # mentah. Tanpa ini, pengguna yang mengaktifkan stemming melihat teks
    # ter-stem lalu menjalankan analisis sentimen yang diam-diam mematikannya -
    # pratinjaunya menyesatkan. None = tampilkan efek konfigurasi apa adanya.
    task: Optional[str] = Field(
        default=None,
        description="'transformer' (sentimen), 'bag_of_words' (topik), "
                    "'span' (aspek), atau kosong untuk konfigurasi apa adanya",
    )

    @field_validator('task')
    @classmethod
    def validate_task(cls, v):
        if v is not None and v not in ('transformer', 'bag_of_words', 'span'):
            raise ValueError(
                "task harus 'transformer', 'bag_of_words', 'span', atau kosong"
            )
        return v

class AspectNormalizeRequest(BaseModel):
    """Request untuk menormalkan nama aspek ke bentuk dasar.

    Dipakai sisi evaluasi: label emas ditulis manusia dalam bentuk dasar
    (`pelayanan`) sedangkan ekstraksi mode automatic mengembalikan bentuk
    permukaan (`pelayanannya`). Membandingkannya sebagai string persis
    menghitung pasangan yang benar sebagai salah dua kali.
    """
    aspects: List[str] = Field(..., min_length=1, max_length=settings.max_batch_size)


class TextAnalysisRequest(BaseModel):
    """Request schema for text analysis"""
    texts: List[str] = Field(..., min_length=1, max_length=settings.max_batch_size)
    preprocessing_config: Optional[PreprocessingConfig] = None
    predefined_aspects: Optional[List[str]] = None
    mode: Optional[str] = "automatic"  # 'automatic' or 'rule-based'
    # 0 berarti "pilihkan otomatis": jumlah topik dicari dengan memaksimalkan
    # coherence C_v, dibatasi agar tidak ada topik yang mendominasi korpus.
    # Lihat TopicService._select_num_topics.
    num_topics: Optional[int] = Field(default=5, ge=0, le=20)

    @field_validator('num_topics')
    @classmethod
    def validate_num_topics(cls, v):
        """Tolak 1 topik: tidak bermakna, dan 0 sudah dipakai untuk mode otomatis."""
        if v == 1:
            raise ValueError('num_topics harus 0 (otomatis) atau minimal 2')
        return v

    @field_validator('texts')
    @classmethod
    def validate_texts(cls, v):
        """
        Rapikan spasi tetapi PERTAHANKAN jumlah dan urutan elemen.

        Versi sebelumnya membuang teks kosong dari daftar. Akibatnya kontrak
        keselarasan indeks putus: pemanggil mengirim N teks dan menerima N-k
        hasil, sehingga predictions[i], document_aspects[i], dan
        document_topics[i] tidak lagi menunjuk teks ke-i milik pemanggil.
        Pada Laravel hal itu membuat setiap baris setelah baris kosong
        tersimpan dengan teks asli yang salah.

        Baris kosong tetap diproses dan menghasilkan entri kosong pada
        posisinya, sehingga penjajaran terjaga.
        """
        if not v:
            raise ValueError('Texts cannot be empty')

        v = [text.strip() for text in v]
        if not any(v):
            raise ValueError('No valid texts provided')

        # Batas panjang per teks. `max_text_length` sebelumnya terdefinisi tetapi
        # tidak pernah dibaca, sehingga tidak ada batas ukuran payload sama
        # sekali. Ini menjaga hal yang BERBEDA dari pemotongan token: teks yang
        # melebihi jendela model tetap dipotong dan dilaporkan lewat
        # metrics.total_truncated, sedangkan batas ini melindungi service dari
        # payload raksasa yang menghabiskan memori sebelum sempat dinilai.
        terlalu_panjang = [
            i for i, text in enumerate(v) if len(text) > settings.max_text_length
        ]
        if terlalu_panjang:
            raise ValueError(
                f'{len(terlalu_panjang)} teks melebihi {settings.max_text_length} '
                f'karakter (indeks pertama: {terlalu_panjang[0]})'
            )
        return v

    @field_validator('mode')
    @classmethod
    def validate_mode(cls, v):
        if v not in ['automatic', 'rule-based']:
            raise ValueError('Mode must be either "automatic" or "rule-based"')
        return v

class AssociationRequest(BaseModel):
    """
    Request schema untuk perhitungan asosiasi aspek-topik (PMI).

    Dipakai Laravel saat dataset besar: sentiment/aspect/topic dipanggil
    terpisah per batch, sehingga PMI tidak bisa dihitung di endpoint combined.
    """
    document_aspects: List[List[Any]] = Field(
        ..., description="Aspek per dokumen (nama aspek atau objek {'aspect': ...})"
    )
    document_topics: List[int] = Field(..., description="Topic id per dokumen, -1 untuk outlier")
    num_topics: int = Field(..., ge=0)
    min_mentions: Optional[int] = Field(default=3, ge=1)


class HealthResponse(BaseModel):
    """Response schema for health check"""
    status: str
    message: str
    version: str
    timestamp: str
    services: Optional[Dict[str, bool]] = None
    models: Optional[Dict[str, Any]] = None
    # Bobot dimuat malas, jadi "service tersedia" tidak sama dengan "model siap".
    # Tanpa field ini `response_model` membuang blok tersebut dari respons dan
    # Laravel tidak bisa membedakan cold start dari analisis yang menggantung.
    weights_loaded: Optional[Dict[str, bool]] = None


class RetrainRequestBase(BaseModel):
    """Field yang sama untuk kedua endpoint retraining."""
    epochs: Optional[int] = Field(default=3, ge=1, le=20)
    learning_rate: Optional[float] = Field(default=2e-5, gt=0, lt=1)
    # Secara default checkpoint yang menurunkan metrik akan ditolak. force=true
    # memaksa penyimpanan - hanya dipakai saat sengaja mengganti baseline.
    force: bool = Field(
        default=False,
        description="Simpan checkpoint walaupun metrik validasi menurun"
    )


class AspectRetrainRequest(RetrainRequestBase):
    """Request schema for aspect model retraining"""
    training_data: List[Dict[str, Any]] = Field(
        ..., min_length=10,
        description="List of {text: str, aspects: [str, ...]}. "
                    "Format dari Laravel training_items.corrected_aspects."
    )

class SentimentRetrainRequest(RetrainRequestBase):
    """Request schema for sentiment model retraining"""
    training_data: List[Dict[str, Any]] = Field(
        ..., min_length=10,
        description="List of {text: str, label: positive|neutral|negative}. "
                    "Format dari Laravel training_items.corrected_sentiment."
    )


class RetrainPreviewRequest(BaseModel):
    """
    Request untuk memeriksa kesiapan data koreksi sebelum retraining dijalankan.

    Memakai payload yang sama dengan endpoint retraining, tetapi tanpa batas
    minimal 10 sampel supaya admin tetap bisa melihat kondisi data yang belum cukup.
    """
    # "model_type" bentrok dengan protected namespace "model_" milik pydantic v2
    model_config = ConfigDict(protected_namespaces=())

    model_type: str = Field(..., description="'sentiment' atau 'aspect'")
    training_data: List[Dict[str, Any]] = Field(default_factory=list)

    @field_validator('model_type')
    @classmethod
    def validate_model_type(cls, v):
        if v not in ('sentiment', 'aspect'):
            raise ValueError("model_type harus 'sentiment' atau 'aspect'")
        return v
