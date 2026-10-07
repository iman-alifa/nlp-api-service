# Image untuk Hugging Face Spaces (SDK: docker), dan untuk host Docker mana pun.
#
# Railway memakai Nixpacks dan MENGABAIKAN berkas ini - konfigurasinya ada di
# `nixpacks.toml` dan `railway.json`. Keduanya dipertahankan supaya pilihan
# hosting tidak mengunci proyek pada satu penyedia.

FROM python:3.11-slim

# gcc/g++ dibutuhkan saat membangun roda untuk hdbscan dan umap-learn
# (dependensi BERTopic); keduanya tidak menyediakan wheel siap pakai untuk
# semua platform. libgomp1 adalah runtime OpenMP yang dipakai scikit-learn.
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces menjalankan kontainer sebagai UID 1000, bukan root. Menulis sebagai
# root membuat cache HuggingFace dan direktori bobot tidak bisa ditulis saat
# runtime, dan kegagalannya muncul sebagai "model gagal dimuat" - bukan sebagai
# galat izin yang jelas.
RUN useradd -m -u 1000 pengguna
USER pengguna
ENV PATH="/home/pengguna/.local/bin:$PATH" \
    HOME=/home/pengguna

WORKDIR /app

# Dependensi disalin lebih dulu supaya lapisan ini di-cache dan tidak dibangun
# ulang setiap kali kode berubah. Membangun torch + BERTopic memakan menit.
COPY --chown=pengguna requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=pengguna app/ ./app/
COPY --chown=pengguna run.py README.md ./

# Cache HuggingFace diarahkan ke HOME yang bisa ditulis pengguna 1000.
# app/config.py menyetel HF_HOME saat impor, tetapi variabel ini menjadi
# nilai bawaannya bila .env tidak menyebutkannya.
ENV HF_HOME=/home/pengguna/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    APP_ENV=production

# HF Spaces mengarahkan lalu lintas ke 7860 kecuali `app_port` disetel di
# frontmatter README. Host lain menyuntikkan PORT sendiri.
ENV PORT=7860
EXPOSE 7860

# `--workers 1` BUKAN pilihan penyetelan: keadaan model, termasuk bobot hasil
# retraining, hidup di global tingkat proses. Dua worker berarti dua salinan
# bobot dan keadaan yang menyimpang setelah retraining.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1"]
