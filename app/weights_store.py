"""
Penyimpanan bobot hasil retraining di Hugging Face Hub.

Masalah yang diselesaikan: `models/` masuk `.gitignore`, jadi bobot hasil
retraining **tidak ikut image**. Pada Railway sebuah redeploy menghapusnya
kecuali ada volume terpasang; pada HF Spaces penyimpanannya memang sementara.
Kalau ini dilupakan, hasil enam iterasi active learning hilang pada deploy
berikutnya dan layanan diam-diam kembali ke bobot publik - tanpa galat, hanya
angka yang memburuk.

Hub dipilih bukan sekadar sebagai tempat menaruh berkas. Ia berbasis git,
sehingga tiap putaran retraining menjadi commit tersendiri: jejak audit yang
lebih kuat daripada `provenance.json` di disk lokal, dan pemulihan ke iterasi
mana pun menjadi satu perintah. Ia juga menjadi "penyimpanan bobot bersama"
yang dibutuhkan bila nanti replika inferensi ditambah (Tahap 2 pada
`docs/RENCANA_DEPLOY.md`).

**Seluruh operasi di sini tidak boleh fatal.** Layanan punya rantai fallback di
setiap modul; gagal menyinkronkan bobot harus menurunkan mutu, bukan
mematikan layanan. Kegagalan unggah khususnya tidak boleh membatalkan
pelatihan yang sudah berjalan dua puluh menit.
"""
import os
import shutil
from typing import Optional

from app.config import settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

# Nama artefak di dalam repo Hub. Sengaja mencerminkan tata letak `models/`
# supaya unduhan bisa langsung ditempatkan tanpa pemetaan nama.
BERKAS_ASPEK = 'aspect_retrained.pt'
DIR_SENTIMEN = 'sentiment_retrained'


def aktif() -> bool:
    """Sinkronisasi menyala hanya bila repo tujuan disetel."""
    return bool(settings.hf_weights_repo)


def _api():
    """
    Impor `huggingface_hub` selambat mungkin.

    Paket ini hanya dibutuhkan ketika sinkronisasi dipakai, dan menjadikannya
    impor tingkat modul berarti deployment yang memakai volume biasa ikut
    menanggung kegagalan impor kalau paketnya belum terpasang.
    """
    from huggingface_hub import HfApi

    return HfApi(token=settings.hf_token or None)


def unduh_bobot() -> bool:
    """
    Tarik bobot dari Hub ke `settings.model_path` saat startup.

    Dipanggil SEBELUM service dibangun, karena setiap service menyelesaikan
    path checkpoint-nya di konstruktor dan jatuh ke bobot publik bila berkasnya
    tidak ada.

    Bobot lokal yang sudah ada TIDAK ditimpa. Pada mesin pengembangan, disk
    lokal adalah sumber kebenaran - menariknya dari Hub akan menimpa hasil
    pelatihan yang belum diunggah.
    """
    if not aktif():
        return False

    tujuan = settings.model_path
    sudah_ada = (
        os.path.exists(os.path.join(tujuan, BERKAS_ASPEK))
        or os.path.isdir(os.path.join(tujuan, DIR_SENTIMEN))
    )
    if sudah_ada:
        logger.info(
            f"Bobot lokal sudah ada di {tujuan}; unduhan dari Hub dilewati "
            "(disk lokal adalah sumber kebenaran)."
        )
        return False

    try:
        from huggingface_hub import snapshot_download

        logger.info(f"Menarik bobot dari Hub: {settings.hf_weights_repo}")
        os.makedirs(tujuan, exist_ok=True)
        snapshot_download(
            repo_id=settings.hf_weights_repo,
            repo_type='model',
            local_dir=tujuan,
            token=settings.hf_token or None,
        )
        logger.info(f"✅ Bobot dari Hub ditempatkan di {tujuan}")
        return True
    except Exception as e:  # noqa: BLE001 - sinkronisasi tidak boleh fatal
        logger.warning(
            f"Gagal menarik bobot dari Hub ({e}); layanan lanjut dengan bobot "
            "yang tersedia secara lokal atau bobot publik."
        )
        return False


def unggah_bobot(modul: str, catatan: Optional[str] = None) -> bool:
    """
    Dorong checkpoint satu modul ke Hub setelah retraining diterima.

    `modul` adalah 'sentiment' atau 'aspect'. Berkas asal-usul
    (`provenance.json`) ikut terunggah, sehingga tiap commit di Hub membawa
    sidik jari kolam dan metrik sebelum/sesudahnya.
    """
    if not aktif():
        return False

    try:
        api = _api()
        api.create_repo(
            repo_id=settings.hf_weights_repo,
            repo_type='model',
            private=True,
            exist_ok=True,
        )

        pesan = catatan or f'Checkpoint {modul} hasil retraining'

        if modul == 'sentiment':
            folder = os.path.join(settings.model_path, DIR_SENTIMEN)
            if not os.path.isdir(folder):
                logger.warning(f"Tidak ada folder {folder}; unggahan dilewati")
                return False
            api.upload_folder(
                folder_path=folder,
                path_in_repo=DIR_SENTIMEN,
                repo_id=settings.hf_weights_repo,
                repo_type='model',
                commit_message=pesan,
            )
        else:
            berkas = os.path.join(settings.model_path, BERKAS_ASPEK)
            if not os.path.exists(berkas):
                logger.warning(f"Tidak ada berkas {berkas}; unggahan dilewati")
                return False
            api.upload_file(
                path_or_fileobj=berkas,
                path_in_repo=BERKAS_ASPEK,
                repo_id=settings.hf_weights_repo,
                repo_type='model',
                commit_message=pesan,
            )
            # Asal-usul disimpan di sebelah berkas .pt, bukan di dalamnya.
            prov = f'{berkas}.provenance.json'
            if os.path.exists(prov):
                api.upload_file(
                    path_or_fileobj=prov,
                    path_in_repo=f'{BERKAS_ASPEK}.provenance.json',
                    repo_id=settings.hf_weights_repo,
                    repo_type='model',
                    commit_message=pesan,
                )

        logger.info(f"✅ Checkpoint {modul} diunggah ke {settings.hf_weights_repo}")
        return True
    except Exception as e:  # noqa: BLE001 - jangan batalkan pelatihan yang sudah jadi
        logger.warning(
            f"Gagal mengunggah checkpoint {modul} ke Hub ({e}). Bobotnya AMAN "
            f"di {settings.model_path}, tetapi akan hilang bila kontainer "
            "diganti tanpa volume. Unggah manual bila perlu."
        )
        return False


def ruang_disk_cukup(butuh_mb: int = 1500) -> bool:
    """
    Periksa sisa ruang sebelum menulis checkpoint.

    Satu checkpoint ~473 MB dan retraining menulis dua (kandidat terbaik plus
    hasil akhir). Disk penuh di tengah penyimpanan meninggalkan checkpoint
    separuh tertulis - persis keadaan yang pernah terjadi di proyek ini ketika
    `config.json` tersimpan sementara `model.safetensors` tidak.
    """
    try:
        _, _, bebas = shutil.disk_usage(settings.model_path or '.')
        bebas_mb = bebas // (1024 * 1024)
        if bebas_mb < butuh_mb:
            logger.warning(
                f"Sisa disk {bebas_mb} MB, di bawah {butuh_mb} MB yang "
                "dibutuhkan untuk menulis checkpoint dengan aman."
            )
            return False
        return True
    except Exception:  # noqa: BLE001
        return True
