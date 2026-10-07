# Panduan deploy — hanya langkah yang harus Anda jalankan

> **DIGANTIKAN (23 September 2026).** Rencana yang disepakati sisi layanan dan
> sisi web adalah **satu VPS** untuk web + layanan NLP. Langkah yang berlaku ada
> di `../text-analysis-web/DEPLOYMENT.md`; alasan teknisnya di
> `docs/JAWABAN_HOSTING_SATU_VPS.md`. Dokumen ini disimpan sebagai riwayat
> pertimbangan (HF Spaces ternyata butuh PRO untuk SDK Docker/Gradio).

Ditulis 23 September 2026.

Nama pengguna Hugging Face yang dipakai di dokumen ini: **`imanalifan`**.
Ganti bila berbeda.

---

## Yang sudah saya kerjakan — jangan diulang

| | hasil |
|---|---|
| Kunci API dibangkitkan dan **ditulis ke kedua `.env`** | nilainya identik, sudah diverifikasi |
| Suite tes Python | **447 lolos**, tiga kali berturut-turut |
| Suite tes Laravel | **125 lolos** |
| Uji mode produksi lokal | `/health` 200 · `/docs` 404 · `/openapi.json` 404 · retrain tanpa kunci **401** · analyze tanpa kunci **401** · dengan kunci **200** |
| Penjaga gagal-tertutup | produksi tanpa `API_KEY` → `RuntimeError`, layanan **menolak start** |
| `Dockerfile` untuk HF Spaces | dibuat, berjalan sebagai UID 1000 |
| `.railwayignore` / `.dockerignore` | kode yang ikut image tinggal **0,39 MB** |

Kunci API Anda — sudah terpasang di kedua `.env`, salin untuk panel hosting:

```
<nilai API_KEY dari .env lokal>
```

**Tidak ada tes lokal yang perlu Anda jalankan.** Langsung ke Langkah 1.

---

## Di mana menjalankannya

| lambang | artinya |
|---|---|
| **[PS-NLP]** | PowerShell di `C:\Skripsi\nlp-api-service` |
| **[PS-SKRIPSI]** | PowerShell di `C:\Skripsi` |
| **[PS-SPACE]** | PowerShell di `C:\Skripsi\hf-space` (dibuat di Langkah 3) |
| **[WEB-HF]** | Peramban → huggingface.co |
| **[WEB-RW]** | Peramban → railway.app |

Membuka PowerShell di folder tertentu: buka folder itu di File Explorer, ketik
`powershell` di bilah alamat, tekan Enter.

---

# Langkah 1 — Token Hugging Face  **[WEB-HF]**

1. Buka huggingface.co, daftar atau masuk.
2. Foto profil → **Settings** → **Access Tokens** → **Create new token**.
3. Tipe **Write**. Nama `skripsi-deploy`.
4. **Salin nilainya sekarang** — hanya ditampilkan sekali. Simpan di catatan.

Token ini dipakai dua kali: mengunggah bobot (Langkah 2) dan sebagai secret di
Space (Langkah 4).

---

# Langkah 2 — Repo bobot

## 2a. Buat repo  **[WEB-HF]**

**+ New** (kanan atas) → **Model**.

- Owner: `imanalifan`
- Model name: `skripsi-nlp-weights`
- **Private** ← wajib
- **Create model**

## 2b. Masuk CLI  **[PS-NLP]**

```powershell
.\venv\Scripts\Activate.ps1
pip install -U "huggingface_hub[cli]"
hf auth login
```

Tempel token Write saat diminta. Pertanyaan *"Add token as git credential?"* →
jawab **n**.

> Bila perintah `hf` tidak dikenali, pakai `huggingface-cli` — nama lamanya,
> argumennya sama persis.

## 2c. Unggah bobot  **[PS-NLP]**

Tempel keempat baris ini. Total ~1,4 GB, biarkan berjalan.

```powershell
hf upload imanalifan/skripsi-nlp-weights models/sentiment_retrained sentiment_retrained --repo-type=model
hf upload imanalifan/skripsi-nlp-weights models/aspect_retrained.pt aspect_retrained.pt --repo-type=model
hf upload imanalifan/skripsi-nlp-weights models/aspect_retrained.pt.provenance.json aspect_retrained.pt.provenance.json --repo-type=model
hf upload imanalifan/skripsi-nlp-weights models/aspect_baseline.pt aspect_baseline.pt --repo-type=model
```

**`aspect_baseline.pt` wajib ikut.** Protokol active learning melatih dari
bobot dasar itu; tanpanya retraining diam-diam kembali ke "melanjutkan
checkpoint" dengan hanya satu baris peringatan di log. Itu persis kegagalan
yang membatalkan satu iterasi pada studi kasus Anda.

## 2d. Pastikan  **[WEB-HF]**

Buka `https://huggingface.co/imanalifan/skripsi-nlp-weights` → tab **Files**.
Harus ada empat: folder `sentiment_retrained/`, `aspect_retrained.pt`,
`aspect_retrained.pt.provenance.json`, `aspect_baseline.pt`.

---

# Langkah 3 — Kirim kode ke Space

## 3a. Buat Space  **[WEB-HF]**

**+ New** → **Space**.

- Owner: `imanalifan`
- Space name: `skripsi-nlp-api`
- License: kosongkan
- SDK: **Docker** → template **Blank**
- **Private** ← wajib
- Hardware: **CPU basic** (gratis)
- **Create Space**

## 3b. Salin kode  **[PS-SKRIPSI]**

```powershell
git clone https://huggingface.co/spaces/imanalifan/skripsi-nlp-api hf-space
cd hf-space
Copy-Item -Recurse -Force ..\nlp-api-service\app .\app
Copy-Item -Force ..\nlp-api-service\requirements.txt, ..\nlp-api-service\run.py, ..\nlp-api-service\Dockerfile .
Get-ChildItem -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force
```

Bila `git clone` meminta kredensial: nama pengguna `imanalifan`, kata sandi =
**token Write** dari Langkah 1 (bukan kata sandi akun Anda).

## 3c. Buat README Space  **[PS-SPACE]**

Space butuh frontmatter YAML di baris pertama. Tempel blok ini apa adanya:

```powershell
@"
---
title: Skripsi NLP API
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Skripsi NLP API

Layanan analisis teks berbahasa Indonesia: sentimen, ekstraksi aspek, dan
pemodelan topik. Backend untuk aplikasi Laravel text-analysis-web.

Seluruh endpoint selain /health memerlukan header X-API-Key.
"@ | Out-File -FilePath README.md -Encoding utf8
```

## 3d. Kirim  **[PS-SPACE]**

```powershell
git add -A
git commit -m "Layanan NLP: inti service"
git push
```

Setelah `push`, Space langsung mencoba membangun dan **akan gagal** — secret
belum ada. Itu wajar. Lanjut ke Langkah 4.

---

# Langkah 4 — Secret Space  **[WEB-HF]**

Space → **Settings** → gulir ke **Variables and secrets**.

Klik **New secret** untuk dua ini (nilainya disembunyikan):

| Name | Value |
|---|---|
| `API_KEY` | `<nilai API_KEY dari .env lokal>` |
| `HF_TOKEN` | token Write dari Langkah 1 |

Klik **New variable** untuk lima ini (nilainya terlihat, tidak masalah):

| Name | Value |
|---|---|
| `HF_WEIGHTS_REPO` | `imanalifan/skripsi-nlp-weights` |
| `APP_ENV` | `production` |
| `TRUST_PROXY` | `true` |
| `MODEL_PATH` | `/home/pengguna/models` |
| `ASPECT_BASELINE_CHECKPOINT` | `/home/pengguna/models/aspect_baseline.pt` |

`LARAVEL_PUBLIC_URL` menyusul di Langkah 6 — domainnya belum ada.

Setelah tersimpan, Space membangun ulang sendiri. **Butuh 10–20 menit** karena
torch besar.

## Pastikan  **[WEB-HF]** → Space → tab **Logs**

```
Menarik bobot dari Hub: imanalifan/skripsi-nlp-weights
✅ Bobot dari Hub ditempatkan di /home/pengguna/models
🚀 Starting NLP API Service...
✅ All models loaded successfully!
```

Lalu **[PS-NLP]**:

```powershell
curl https://imanalifan-skripsi-nlp-api.hf.space/health
```

Cari dua baris ini di jawabannya:

```
"sentiment_retrained": true
"aspect_retrained": true
```

> **Kalau keduanya `false`, berhenti dan beri tahu saya.** Artinya bobot tidak
> tertarik: layanan tetap jalan memakai bobot publik dan **tidak ada galat yang
> muncul** — hasilnya hanya lebih buruk, tanpa penjelasan. Ini pemeriksaan
> terpenting di seluruh panduan.

---

# Langkah 5 — Laravel di Railway

## 5a. Kunci aplikasi  **[PS-SKRIPSI]**

```powershell
cd C:\Skripsi\text-analysis-web
php artisan key:generate --show
```

Salin hasilnya (diawali `base64:`).

## 5b. Proyek dan basis data  **[WEB-RW]**

1. railway.app → **New Project** → **Deploy from GitHub repo** → pilih repo Anda.
2. Buka service yang terbentuk → **Settings** → **Root Directory** → isi
   `text-analysis-web` → **Save**.
3. **Settings** → **Networking** → **Generate Domain**. Salin domainnya.
4. Di kanvas proyek: **+ Create** → **Database** → **Add MySQL**.

## 5c. Variabel service web  **[WEB-RW]**

Service web → tab **Variables** → **Raw Editor** → tempel seluruh blok, lalu
ganti dua baris bertanda panah:

```
APP_NAME=Analisis Teks
APP_ENV=production
APP_DEBUG=false
APP_KEY=base64:GANTI            ← hasil 5a
APP_URL=https://GANTI.up.railway.app   ← domain dari 5b

LOG_CHANNEL=stack
LOG_LEVEL=error

DB_CONNECTION=mysql
DB_HOST=${{MySQL.MYSQLHOST}}
DB_PORT=${{MySQL.MYSQLPORT}}
DB_DATABASE=${{MySQL.MYSQLDATABASE}}
DB_USERNAME=${{MySQL.MYSQLUSER}}
DB_PASSWORD=${{MySQL.MYSQLPASSWORD}}

QUEUE_CONNECTION=database
DB_QUEUE_RETRY_AFTER=4200

NLP_API_URL=https://imanalifan-skripsi-nlp-api.hf.space
NLP_API_KEY=<nilai API_KEY dari .env lokal>
NLP_API_TIMEOUT=600
NLP_API_BATCH_SIZE=50
```

> Tiga angka `600 < 1800 < 4200` adalah rantai timeout yang sudah dikalibrasi
> dan diuji. Jangan ubah salah satunya sendirian — ketidakcocokannya pernah
> membuat **setiap** analisis berjalan dua kali secara bersamaan.

## 5d. Queue worker sebagai service terpisah  **[WEB-RW]**

**+ Create** → **GitHub Repo** → repo yang **sama**.

- **Settings** → **Root Directory**: `text-analysis-web`
- **Settings** → **Custom Start Command**:
  ```
  php artisan queue:work --timeout=1800 --tries=3 --sleep=3
  ```
- **Variables** → **Raw Editor** → tempel **blok yang sama persis** dari 5c.

Worker wajib proses sendiri. Menjalankannya di service web berarti ia mati
setiap kali web di-restart, dan analisis menggantung tanpa penjelasan apa pun.

## 5e. Migrasi  **[WEB-RW]** → service web → tab **Shell**

Tunggu deploy pertama selesai (indikator hijau), lalu:

```bash
php artisan migrate --force
php artisan config:cache
php artisan route:cache
```

---

# Langkah 6 — Tutup lingkarannya  **[WEB-HF]**

Space → **Settings** → **New variable**:

| Name | Value |
|---|---|
| `LARAVEL_PUBLIC_URL` | domain Railway Anda, mis. `https://xxx.up.railway.app` |

Space restart sendiri. Ini yang membuat CORS mengizinkan peramban pengguna.

---

# Langkah 7 — Pemeriksaan akhir

**[PS-NLP]** — ganti URL bila berbeda:

```powershell
curl https://imanalifan-skripsi-nlp-api.hf.space/health
curl https://imanalifan-skripsi-nlp-api.hf.space/docs
curl -X POST https://imanalifan-skripsi-nlp-api.hf.space/api/retrain/sentiment -H "Content-Type: application/json" -d "[]"
```

| # | perintah | harapan |
|---|---|---|
| 1 | `/health` | 200, `sentiment_retrained: true`, `aspect_retrained: true` |
| 2 | `/docs` | **404** |
| 3 | retrain tanpa kunci | **401** |

Lalu di peramban:

| # | lakukan | harapan |
|---|---|---|
| 4 | Buka aplikasi Railway, daftar/masuk | halaman termuat |
| 5 | Unggah CSV ~20 baris, jalankan analisis | selesai, hasil tampil |
| 6 | **[WEB-RW]** → service **worker** → tab Logs | ada panggilan ke NLP, **tanpa 401** |
| 7 | Buka halaman koreksi | baris muncul, urut dari keyakinan terendah |

Nomor **3** dan **6** yang paling menentukan. **3** membuktikan orang luar
tidak bisa menimpa bobot Anda. **6** membuktikan Laravel bisa — artinya
kuncinya cocok. Kalau nomor 6 menunjukkan 401, nilai `API_KEY` di Space dan
`NLP_API_KEY` di Railway tidak identik; paling sering karena spasi ikut
tersalin di ujung.

---

# Setelah hidup

- **Pemanasan sudah otomatis.** Membuka halaman analisis memicu pemuatan bobot,
  sehingga biayanya bertumpang tindih dengan waktu Anda memilih berkas. Tidak
  perlu tindakan apa pun.
- **Sebelum sidang**, buka `/health` beberapa menit lebih awal — Space gratis
  tidur bila lama tidak dipakai.
- **Jangan retraining saat demo.** Memakan ~20 menit dan mengunci inferensi
  selama itu (perlindungan yang disengaja, bukan bug). Tunjukkan riwayat di
  `/admin/training`: di sana ada `val_size`, delta F1, dan suhu kalibrasi tiap
  putaran — jauh lebih informatif bagi penguji daripada menonton bilah
  kemajuan.
- **Retraining di produksi otomatis mendorong bobot ke HF Hub** sebagai commit
  baru. Periksa halaman repo bobot setelah putaran pertama; itu bukti
  sinkronisasinya bekerja.
- **Jangan pernah commit `.env`.** Bila kunci pernah masuk git, bangkitkan yang
  baru dan ganti di empat tempat: dua `.env` lokal, Space, dan Railway.
