"""
Kontrol akses untuk deployment publik.

Sebelum modul ini ada, SELURUH endpoint terbuka. `CORSMiddleware` yang sudah
terpasang hanya membatasi peramban - `curl` mengabaikannya sepenuhnya - jadi
siapa pun yang menemukan URL layanan bisa memanggil `/api/retrain/sentiment`
dan menimpa bobot model. Endpoint-nya bahkan mendokumentasikan dirinya sendiri
di `/docs`. Itu model poisoning tanpa perlu keahlian khusus, dan pada proyek
ini ia akan menghapus hasil enam iterasi active learning.

Dua kontrol di sini, keduanya sengaja sederhana:

1. **Kunci API** pada semua endpoint kecuali `/` dan `/health`. Laravel adalah
   satu-satunya pemanggil sah, jadi tidak ada alasan endpoint analisis pun
   terbuka. `/health` harus tetap terbuka karena platform hosting memakainya
   sebagai healthcheck dan tidak bisa mengirim header.

2. **Batas laju per alamat IP** pada endpoint yang memakan CPU. Analisis
   menerima ratusan teks 10.000 karakter dan berjalan di CPU; beberapa
   permintaan sekaligus cukup membuat layanan tidak responsif, dan karena
   inferensi berjalan serial di event loop antreannya menumpuk.
"""
import time
from collections import deque
from typing import Deque, Dict, Optional

from fastapi import Header, HTTPException, Request, status

from app.config import settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

# Endpoint yang tidak pernah butuh kunci: healthcheck platform hosting tidak
# bisa mengirim header kustom, dan root dipakai untuk pemeriksaan hidup-mati.
JALUR_TERBUKA = frozenset({'/', '/health'})


def auth_aktif() -> bool:
    """Autentikasi menyala begitu `api_key` diisi."""
    return bool(settings.api_key)


def verifikasi_startup() -> None:
    """
    Menolak start di produksi tanpa kunci API.

    Sengaja gagal-tertutup. Alternatifnya - mencatat peringatan lalu tetap
    jalan - menghasilkan layanan yang terbuka dengan satu baris log yang tidak
    dibaca siapa pun, dan itu persis mode kegagalan yang berulang kali muncul
    di proyek ini: degradasi diam-diam yang baru ketahuan setelah terlambat.

    Di pengembangan kunci boleh kosong, supaya tes dan `python run.py` lokal
    tidak perlu konfigurasi tambahan.
    """
    if settings.app_env == 'production' and not auth_aktif():
        raise RuntimeError(
            "API_KEY wajib diisi ketika APP_ENV=production. Tanpa itu endpoint "
            "/api/retrain/* terbuka untuk siapa pun yang tahu URL layanan, dan "
            "bobot model bisa ditimpa oleh pemanggil mana pun."
        )
    if not auth_aktif():
        logger.warning(
            "API_KEY kosong - autentikasi DIMATIKAN. Ini hanya boleh untuk "
            "pengembangan lokal."
        )


def kunci_sah(nilai: Optional[str]) -> bool:
    """
    Apakah `nilai` adalah kunci API yang benar.

    False ketika autentikasi mati: tanpa kunci terkonfigurasi tidak ada
    pemanggil yang bisa dianggap "terpercaya", jadi tidak ada yang boleh
    dibebaskan dari batas laju atas dasar kunci.
    """
    if not auth_aktif() or not nilai:
        return False

    import secrets

    return secrets.compare_digest(nilai, settings.api_key)


async def wajib_kunci_api(
    request: Request,
    x_api_key: Optional[str] = Header(default=None, alias='X-API-Key'),
) -> None:
    """
    Dependency FastAPI: menolak permintaan tanpa kunci yang benar.

    Dibandingkan dengan `secrets.compare_digest` supaya waktu bandingnya tidak
    bergantung pada berapa karakter awal yang cocok.
    """
    if not auth_aktif() or request.url.path in JALUR_TERBUKA:
        return

    import secrets

    diberikan = x_api_key or ''
    if not secrets.compare_digest(diberikan, settings.api_key):
        # Pesannya sengaja tidak membedakan "tidak ada kunci" dari "kunci
        # salah" - keduanya tidak perlu dibantu.
        logger.warning(f"Kunci API ditolak untuk {request.url.path}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Kunci API tidak valid. Sertakan header X-API-Key.",
        )


class PembatasLaju:
    """
    Pembatas laju jendela-geser, disimpan di memori proses.

    Memori proses **benar** di sini, bukan jalan pintas: layanan ini wajib
    berjalan `--workers 1` (keadaan model hidup di global tingkat proses), jadi
    satu proses melihat seluruh lalu lintas kontainer ini. Penyimpanan bersama
    seperti Redis baru diperlukan kalau replika inferensi ditambah - dan itu
    Tahap 2 pada `docs/RENCANA_DEPLOY.md`.
    """

    def __init__(self, maks: int, jendela_detik: int) -> None:
        self.maks = maks
        self.jendela = jendela_detik
        self._jejak: Dict[str, Deque[float]] = {}

    def izinkan(self, kunci: str, sekarang: Optional[float] = None) -> bool:
        # maks <= 0 mematikan pembatas sepenuhnya. Dipakai oleh suite tes, dan
        # sah di produksi ketika sudah ada pembatas laju di lapisan depan
        # (CDN, nginx) - dua pembatas bertingkat hanya membuat 429 sulit
        # dilacak asalnya.
        if self.maks <= 0:
            return True

        sekarang = time.monotonic() if sekarang is None else sekarang
        batas_bawah = sekarang - self.jendela

        antrean = self._jejak.setdefault(kunci, deque())
        while antrean and antrean[0] <= batas_bawah:
            antrean.popleft()

        if len(antrean) >= self.maks:
            return False

        antrean.append(sekarang)
        return True

    def reset(self) -> None:
        """Kosongkan seluruh jejak. Dipakai suite tes agar tiap tes berdiri
        sendiri - tanpa ini kuota bocor antar-tes dan kegagalannya bergantung
        pada urutan eksekusi."""
        self._jejak.clear()

    def bersihkan(self, sekarang: Optional[float] = None) -> None:
        """
        Buang pelacak IP yang sudah kedaluwarsa seluruhnya.

        Tanpa ini `_jejak` tumbuh selamanya seiring jumlah IP unik - kebocoran
        memori lambat pada layanan yang berumur panjang.
        """
        sekarang = time.monotonic() if sekarang is None else sekarang
        batas_bawah = sekarang - self.jendela
        mati = [k for k, v in self._jejak.items() if not v or v[-1] <= batas_bawah]
        for k in mati:
            del self._jejak[k]


def alamat_klien(request: Request) -> str:
    """
    Alamat IP pemanggil, menembus satu lapis proxy bila dikonfigurasi.

    `X-Forwarded-For` **hanya** dipercaya ketika `settings.trust_proxy` menyala.
    Header itu ditulis klien dan bisa dipalsukan; mempercayainya tanpa syarat
    membuat batas laju bisa dilewati hanya dengan mengarang header. Railway dan
    HF Spaces sama-sama menaruh layanan di belakang satu proxy, jadi entri
    PERTAMA adalah klien aslinya pada topologi itu.
    """
    if settings.trust_proxy:
        diteruskan = request.headers.get('x-forwarded-for')
        if diteruskan:
            return diteruskan.split(',')[0].strip()
    return request.client.host if request.client else 'tidak diketahui'
