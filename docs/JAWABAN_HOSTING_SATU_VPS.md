# Jawaban: hosting satu VPS untuk web + layanan NLP

Dari sisi `nlp-api-service`, 23 September 2026. Menjawab
`docs/TITIPAN_HOSTING_SATU_VPS.md`.

**Usulan satu VPS disetujui.** Masalah 1.3 memang hilang dengan disk permanen,
dan 8 GB cukup. Tapi menjawab T1 memunculkan satu masalah yang **harus**
dibereskan di sisi web sebelum pelatihan pertama di VPS (§T1), dan T2 terjawab
dengan bukti bahwa opsi (a) lebih buruk dari yang terlihat (§T2).

Perubahan di sisi layanan hari ini, semuanya teruji (456 tes lolos):

| perubahan | berkas |
|---|---|
| 503 + `Retry-After` untuk analisis saat model sedang dilatih (saklar, mati secara bawaan) | `app/main.py`, `app/config.py` |
| Penangan `HTTPException` kini meneruskan header — sebelumnya `Retry-After` **terbuang diam-diam** | `app/main.py` |
| Bawaan `settings.host` → `127.0.0.1` | `app/config.py` |
| Pemanggil berkunci sah dibebaskan dari batas laju (lihat §Catatan tambahan) | `app/main.py`, `app/security.py` |

Tidak ada perubahan kontrak API selain 503 baru yang hanya muncul bila saklarnya
dinyalakan.

---

## T1. Lama pelatihan di CPU 4 core

**Diukur di CPU saja, tanpa GPU**, pada laptop pemilik proyek:

| | nilai |
|---|---|
| CPU | AMD Ryzen 7 5700U — 8 core fisik / 16 thread, clock dasar 1,8 GHz |
| torch | `2.9.1+cpu` (build CPU; `torch.version.cuda = None`), 8 thread |
| sentimen, kolam 481 | 13 mnt 30 dtk → **1,68 dtk/sampel** |
| aspek, kolam 398 | 7 mnt 2 dtk → **1,06 dtk/sampel** |

Angkanya satu putaran penuh: 3 epoch + evaluasi sebelum/sesudah + kalibrasi,
diambil dari `model_trainings` #14 dan #15.

**Perkiraan di Contabo 4 vCPU: 2–3x lebih lambat.** vCPU di VPS bersama
biasanya satu thread, bukan satu core fisik, dan dibagi dengan tetangga. Jadi
pada kolam sekarang: sentimen **27–40 menit**, aspek **14–21 menit**, total
**41–61 menit**. Ini perkiraan; ukur pada pelatihan pertama di VPS dan setel
`RETRAIN_SEC_PER_SAMPLE_*` dengan angka nyatanya.

Waktu tumbuh **linear terhadap kolam** — `retrain_from_baseline` melatih
seluruh kolam setiap putaran.

### Masalah yang harus dibereskan sisi web lebih dulu

`NLPApiService::executeRetrain` memakai `$this->http($this->timeout * 2)` =
**1.200 detik**. Pelatihan sentimen di VPS kemungkinan melewati itu.

Yang terjadi kemudian **bukan** sekadar gagal. Menyerahnya klien tidak
menghentikan pekerjaan di server — terukur dengan server sungguhan (§T2).
Pelatihan jalan terus dan **menyimpan checkpoint**, sementara Laravel
mencatat putaran itu `failed`:

- `model_trainings` bilang gagal, padahal bobot yang hidup sudah berganti.
- Penjaga "pelatihan belum diukur" tidak melihatnya karena statusnya `failed`.
- Evaluasi berikutnya dikaitkan ke putaran yang salah.

Itu persis jenis catatan-vs-bobot yang tidak cocok yang dulu membatalkan
iterasi 2.

**Usul:** batas HTTP retrain dipisah dari batas analisis, dengan rantai yang
sama dengan rantai analisis (HTTP < job < `retry_after`):

| lapis | usulan |
|---|---|
| HTTP retrain | 6.600 dtk |
| `RetrainModel::$timeout` | 7.200 dtk (tetap) |
| `retry_after` antrean `training` | 7.500 dtk |

Kalau Laravel mencoba lagi saat layanan masih melatih, ia dapat **409** —
penjaga itu sudah ada dan tidak berubah.

Jangka panjang, bentuk yang benar adalah retrain asinkron: `202` + id tugas,
lalu Laravel menanyakan statusnya. Itu perubahan kontrak, jadi tidak saya
kerjakan sekarang.

---

## T2. Analisis yang datang saat pelatihan: pilih (b)

### Kenapa (a) lebih buruk dari "tertunda atau gagal sekali"

Diuji dengan uvicorn sungguhan: satu endpoint memegang lock 4 detik, endpoint
lain menunggunya, klien menyerah setelah 1 detik.

```
klien : TIMEOUT, menyerah
server: analisis TETAP DIJALANKAN setelah menunggu 3.7 dtk
```

Starlette tidak membatalkan handler ketika klien putus. Jadi setiap analisis
yang menyerah di detik ke-600 tetap berjalan begitu pelatihan selesai, tanpa ada
yang menunggu hasilnya, sementara Laravel sudah mengirim salinan baru. Di
server 4 vCPU, tumpukan pekerjaan hantu itu jatuh tepat sesudah pelatihan
paling berat.

(c) punya masalah yang sama, hanya lebih lama.

### Yang sudah dipasang

Saklar `RETRAIN_REJECT_INFERENCE` (**mati secara bawaan**). Bila menyala:

- Analisis yang butuh model **yang sedang dilatih** langsung dibalas **503**
  dengan `Retry-After`.
- `/analyze/aspect` dan `/analyze/combined` ikut ditolak saat **sentimen**
  dilatih, karena `AspectService` memakai `SentimentService` bersama untuk
  polaritas per aspek.
- `/analyze/topic` **tidak pernah** ditolak — model topik tidak dilatih lewat API.
- Analisis yang menunggu **analisis lain** tetap mengantre seperti biasa, karena
  itu memang singkat.
- `Retry-After` = perkiraan sisa waktu (sampel × dtk/sampel − yang sudah
  berjalan), dibatasi **30–600 detik**. Meleset tidak masalah: mencoba terlalu
  cepat hanya menghasilkan 503 lagi, yang murah.
- Status pelatihan dibersihkan walau pelatihan melempar galat — kalau
  tertinggal, semua analisis sesudahnya akan ditolak selamanya. Ada tesnya.

Bug yang ditemukan sekalian: penangan `HTTPException` membangun `JSONResponse`
baru dan **membuang `exc.headers`**. `Retry-After` tidak akan pernah sampai ke
Laravel. Sudah diperbaiki, dan tesnya dipastikan gagal kalau perbaikan itu
dicabut (`KeyError: 'Retry-After'`).

### Yang perlu dari sisi web — dan urutannya penting

1. `NLPApiService`: bila **503 membawa `Retry-After`**, jangan ulang HTTP tiga
   kali; lepas job kembali ke antrean dengan jeda sebesar itu **tanpa
   menghabiskan jatah percobaan**. 503 **tanpa** header itu tetap berarti
   "layanan belum siap" dan ditangani seperti sekarang.
2. **Baru setelah itu** setel `RETRAIN_REJECT_INFERENCE=true` di layanan.

Kalau saklar dinyalakan lebih dulu, analisis yang sekarang berhasil menunggu
pelatihan singkat justru gagal setelah tiga percobaan.

Dengan (b), usulan §5.1 (antrean `training` terpisah + worker kedua) menjadi
benar-benar berguna, bukan sekadar memindahkan antrean: worker analisis tidak
lagi tertahan menunggu lock, ia menerima 503 dalam milidetik dan melepas job.

---

## T3. systemd + venv

Setuju systemd + venv. Di VPS tunggal ia lebih ringan, log menyatu di
`journalctl`, dan `models/` otomatis berada di disk host. `Dockerfile` tetap
disimpan untuk HF Spaces dan host kontainer lain.

```ini
# /etc/systemd/system/nlp-api.service
[Unit]
Description=NLP Analysis API
After=network-online.target
Wants=network-online.target

[Service]
User=nlp
WorkingDirectory=/opt/nlp-api-service
EnvironmentFile=/opt/nlp-api-service/.env
# Mutlak, bukan relatif - lihat T5.
Environment=HF_HOME=/opt/nlp-api-service/.cache/huggingface
# Sisakan satu vCPU untuk PHP-FPM dan MySQL. Tanpa ini torch memakai keempat
# vCPU saat pelatihan dan web terasa beku selama 40-60 menit. Harganya:
# pelatihan kira-kira 25-33% lebih lama.
Environment=OMP_NUM_THREADS=3
Environment=MKL_NUM_THREADS=3
ExecStart=/opt/nlp-api-service/venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8001 --workers 1
Restart=on-failure
RestartSec=5
TimeoutStopSec=30

[Install]
WantedBy=multi-user.target
```

- `--host 127.0.0.1` ditulis eksplisit walau bawaannya kini sudah
  `127.0.0.1` — lapis kedua bila `.env` suatu saat menyetel `HOST`.
- `--workers 1` wajib (RENCANA_DEPLOY §1.1).
- Layanan juga menulis `logs/api.log` (`settings.log_file`). Pastikan direktori
  itu bisa ditulis user `nlp` dan pasang logrotate, atau kosongkan `LOG_FILE`
  dan andalkan journald saja.
- Me-restart layanan **saat pelatihan berjalan** membuang pelatihan itu.
  Checkpoint lama tetap utuh karena bobot baru hanya ditulis setelah
  diterima — aman, hanya sia-sia.

---

## T4. torch versi CPU: aman, dan justru lebih setia

Mesin yang menghasilkan semua checkpoint sendiri memakai `torch 2.9.1+cpu`. Jadi
build CPU di VPS adalah **build yang sama dengan yang melatih modelnya**,
bukan kompromi.

Checkpoint dimuat dengan `map_location=self.device`, dan tanpa CUDA
`self.device` = `cpu`.

```bash
venv/bin/pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
venv/bin/pip install -r requirements.txt
```

Urutannya penting. `torch==2.9.1` di `requirements.txt` sudah terpenuhi oleh
`2.9.1+cpu` (label lokal PEP 440), jadi pip tidak menarik build CUDA di
langkah kedua.

---

## T5. Isi `models/` dan cache HF

### Wajib disalin (~1,42 GB)

| berkas | kenapa |
|---|---|
| `sentiment_retrained/` (seluruh isinya) | bobot, tokenizer, `calibration.json` (T = 0,9336), `provenance.json` |
| `aspect_retrained.pt` + `.provenance.json` | bobot aspek iterasi 6 |
| `aspect_baseline.pt` | **titik berangkat protokol kolam** untuk aspek (K6, md5 `0ce4032e42e0b6966b8b3c68d5895bae`). Tanpanya retraining diam-diam melanjutkan checkpoint dengan hanya satu `WARNING` di log — kegagalan yang dulu membatalkan iterasi 2 |

`aspect_retrained.pt.bak_k1` **tidak ikut** — itu K1, eksperimen lama. Isi yang
sama persis sudah ada di repo HF `imanalifan/skripsi-nlp-weights`, jadi bisa
juga ditarik dari sana (T6) alih-alih disalin dari laptop.

### Yang diunduh dari HF Hub saat jalan

| model | kapan | revisi di mesin pengembang |
|---|---|---|
| `indobenchmark/indobert-base-p1` | pertama kali aspek/topik dimuat (tokenizer + arsitektur) | `c2cd0b51ddce6580eb35263b39b0a1e5fb0a39e2` |
| `crypter70/IndoBERT-Sentiment-Analysis` | **hanya saat retraining sentimen** (protokol kolam menukar ke bobot dasar). Inferensi biasa memakai `sentiment_retrained/` | `e4f806186f3e4adebe369bef85b328bc3c819eb2` |

Tujuannya `HF_HOME`, yang bawaannya **relatif**: `./.cache/huggingface`.
Temuan yang relevan: di mesin pengembang, `crypter70` ternyata **tidak** ada di
cache proyek, melainkan di `~/.cache/huggingface` milik pengguna. Artinya
lokasi cache saat ini ditentukan lingkungan, bukan konfigurasi. Karena itu unit
systemd di T3 menyetel `HF_HOME` **mutlak**.

Unduh keduanya saat pemasangan, supaya kegagalan jaringan muncul saat instal,
bukan di tengah pelatihan pertama:

```bash
sudo -u nlp env HF_HOME=/opt/nlp-api-service/.cache/huggingface \
  /opt/nlp-api-service/venv/bin/hf download crypter70/IndoBERT-Sentiment-Analysis \
  --revision e4f806186f3e4adebe369bef85b328bc3c819eb2
sudo -u nlp env HF_HOME=/opt/nlp-api-service/.cache/huggingface \
  /opt/nlp-api-service/venv/bin/hf download indobenchmark/indobert-base-p1 \
  --revision c2cd0b51ddce6580eb35263b39b0a1e5fb0a39e2
```

### Risiko yang belum ditutup

Kode memanggil `from_pretrained(nama)` **tanpa** `revision`. Di server yang baru
dipasang, ia mengambil revisi terbaru. Kalau pemilik `crypter70` memperbarui
modelnya, bobot dasar protokol kolam ikut berganti **diam-diam**, dan putaran
berikutnya tidak lagi sebanding dengan iterasi 0–6 di skripsi. Mengunduh
revisi tertentu seperti di atas tidak cukup, karena `from_pretrained` tetap
memeriksa revisi `main` ke Hub.

Perbaikannya memakukan `revision` di kode. Belum saya kerjakan: pemuatan model
tersebar di 8 titik pada dua service, dan suite tes sengaja tidak memuat bobot,
jadi perubahan di sana tidak bisa diverifikasi dengan baik tepat sebelum deploy.
Risikonya kecil tapi nyata. Saya sarankan dikerjakan sebagai tindak lanjut.

---

## T6. Cadangan: pertahankan repo HF — tapi yang paling penting bukan bobot

**Pertahankan `weights_store.py` + repo HF.** Ia sudah terpasang, gratis (repo
privat itu dibuat saat akun belum PRO, jadi akun gratis memang boleh
menyimpannya), berversi (satu commit per retraining yang diterima, beserta
`provenance.json`), dan berjalan otomatis. Salinan manual ke laptop cenderung
terlupa.

Di VPS: setel `HF_WEIGHTS_REPO=imanalifan/skripsi-nlp-weights` dan `HF_TOKEN`
(bertipe **write**, supaya unggahan setelah retraining jalan).
`unduh_bobot()` tidak pernah menimpa bobot lokal yang sudah ada — disk VPS tetap
sumber kebenaran. Pemulihan setelah pasang ulang: kosongkan `models/`, start
layanan, bobot tertarik sendiri.

**Yang jauh lebih penting untuk dicadangkan adalah MySQL.** Bobot bisa
dibangkitkan ulang dari kolam koreksi: protokol kolam deterministik (seed 42),
dan itu sudah terbukti — pelatihan #5 dan #6 menghasilkan delta, `val_size`, dan
suhu yang identik. **Koreksinya sendiri tidak bisa dibangkitkan ulang.**
`training_items`, golden dataset, dan `evaluation_runs` adalah satu-satunya
data yang tak tergantikan di seluruh sistem. `mysqldump` harian ke luar server
lebih mendesak daripada cadangan bobot mana pun.

---

## Catatan tambahan untuk DEPLOYMENT.md

- **Kunci produksi baru**, seperti usulan §3 — jangan pakai kunci yang kini ada
  di `.env` lokal pengembangan.
- **`TRUST_PROXY=false`**. Laravel memanggil lewat loopback, tanpa proxy di
  depan layanan NLP. Ubah hanya bila layanan suatu saat dipasang di belakang
  nginx.
- **Pembatas laju kini membebaskan pemanggil berkunci sah.** Versi sebelumnya
  membatasi 30 permintaan/menit per IP, padahal Laravel memanggil dari satu IP
  dan memecah analisis per 50 teks. Analisis TNI (3.366 baris) berarti ~136
  permintaan. Terukur dengan pembebasan dimatikan: permintaan ke-4 sudah 429.
  Di satu VPS masalahnya sama persis, karena semua datang dari `127.0.0.1`.
- **Memori.** Layanan terukur 1,79 GB setelah `/api/warmup` (sentimen, aspek,
  dan encoder topik termuat). Puncak pelatihan ~3,2 GB masih perkiraan. Satu
  hal yang mudah terlewat: analisis topik **tidak** mengambil lock, jadi ia bisa
  berjalan bersamaan dengan pelatihan, dan UMAP/HDBSCAN pada korpus besar
  menambah memori di atas puncak itu. Dengan sisi web ±1,5 GB, 8 GB + swap 4 GB
  tetap cukup, tapi pantau `free -h` pada pelatihan pertama.
- **`deploy:check --produksi`** sebagai gerbang akhir: setuju. `/health` hijau
  memang tidak membuktikan kuncinya diterima.
