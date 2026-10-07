import logging
import sys
from pathlib import Path
from datetime import datetime

def setup_logger(name: str, log_file: str = None, level=logging.INFO):
    """
    Siapkan logger dengan handler konsol dan berkas.

    Args:
        name: Nama logger, lazimnya `__name__`.
        log_file: Path berkas log. Bila None, dipakai `settings.log_file` -
            sebelumnya TIDAK ADA pemanggil yang mengisi argumen ini, sehingga
            handler berkas tidak pernah terpasang dan setting `LOG_FILE`
            hanyalah knob mati. Di Railway stdout memang ditangkap platform,
            tetapi saat menelusuri kegagalan secara lokal berkas log jauh lebih
            berguna daripada menggulung terminal.
        level: Ambang level logging.

    Returns:
        Logger yang sudah terkonfigurasi.
    """
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    # Format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    # Pesan log memakai emoji (✅ 🔁 ⚠️). Konsol Windows default cp1252 tidak bisa
    # meng-encode-nya sehingga setiap baris berubah menjadi "--- Logging error ---"
    # dan isi log sebenarnya hilang. Paksa UTF-8 bila stream mendukungnya.
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):  # stream tanpa reconfigure (mis. saat di-capture)
        pass

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file is None:
        try:
            from app.config import settings
            log_file = settings.log_file
        except Exception:  # noqa: BLE001 - logging tidak boleh menjatuhkan impor
            log_file = None

    if log_file:
        try:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except OSError as exc:
            # Sistem berkas hanya-baca (lazim di kontainer) tidak boleh
            # menghentikan service; konsol tetap menerima seluruh log.
            logger.warning(f"Log berkas dinonaktifkan ({log_file}): {exc}")

    return logger