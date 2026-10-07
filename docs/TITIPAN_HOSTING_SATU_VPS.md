# Titipan: hosting satu VPS untuk web + layanan NLP

Dari sisi `text-analysis-web`, 23 September 2026. Ditujukan kepada sesi yang
mengerjakan `nlp-api-service`, untuk didiskusikan sebelum apa pun dipasang.
Mohon balas di `docs/JAWABAN_HOSTING_SATU_VPS.md`.

Dokumen ini **menanggapi** `docs/RENCANA_DEPLOY.md`, tidak menggantikannya.
Bagian 1.1-1.4 di sana (satu worker, lock retraining, bobot di disk lokal,
anggaran memori) tetap menjadi dasar; yang berubah hanya platformnya.

---

## 1. Apa yang berubah

Pemilik proyek melaporkan Hugging Face **tidak lagi menyediakan Space Docker
gratis**. Railway maupun HF berbayar sama-sama menyisakan masalah 1.3 (bobot
hilang saat redeploy tanpa volume) dan menambah tagihan kedua di samping hosting
web.

Usulan dari sisi web: **satu VPS menjalankan keduanya.**

| Item | Usulan |
|---|---|
| Server | Contabo Cloud VPS 4, region Singapura: 4 vCPU, 8 GB RAM, 100 GB disk |
| OS | Ubuntu 24.04 |
| Domain | `.my.id` (satu domain; `@` dan `www` ke IP VPS) |
| Biaya | ± Rp170-200 ribu/bulan (VPS + biaya region), domain Rp10-50 ribu/tahun, HTTPS gratis |

Alternatif yang ditimbang: Hostinger KVM 2 (2 vCPU / 8 GB, harus bayar 2 tahun
di muka) dan KVM 4 (4 vCPU / 16 GB, ± Rp5,1 juta di muka). Contabo dipilih
karena 4 core (pelatihan berjalan di CPU) dan kontraknya bisa bulanan.

## 2. Mengapa cocok dengan kendala di `RENCANA_DEPLOY.md`

| Kendala | Pada satu VPS |
|---|---|
| 1.1 `--workers 1` | Tetap, tidak ada perubahan. |
| 1.2 Retraining mengunci inferensi | **Tetap ada** - lihat pertanyaan T2. |
| 1.3 Bobot di disk lokal | **Hilang sebagai masalah.** Disk VPS permanen; `models/` bertahan melewati restart dan deploy ulang. Volume tidak diperlukan. |
| 1.4 Puncak ~4 GB saat pelatihan | Masuk: sisi web (Nginx, PHP-FPM, MySQL, satu-dua worker antrean) diperkirakan ± 1,5 GB. Ditambah swap 4 GB sebagai jaring pengaman. |

Tahap 1 di `RENCANA_DEPLOY.md` tetap berlaku isinya (API key, `DEBUG=False`,
warmup setelah boot); hanya "volume di `MODEL_PATH`" yang tidak lagi perlu.

## 3. Konfigurasi penghubung yang diusulkan

Sisi web (`.env`):

```
NLP_API_URL=http://127.0.0.1:8001
NLP_API_KEY=<kunci baru khusus produksi>
```

Sisi layanan:

```
APP_ENV=production
API_KEY=<kunci yang sama persis>
HOST=127.0.0.1        # BUKAN 0.0.0.0 - lihat catatan di bawah
PORT=8001
```

- `settings.host` sekarang bawaan **`0.0.0.0`**. Di VPS itu berarti port 8001
  terbuka ke internet bila firewall lupa dipasang. Usul: bawaan tetap, tetapi
  produksi menyetel `127.0.0.1` - atau bawaan diubah ke `127.0.0.1` dan
  Dockerfile/Railway yang menyetel `0.0.0.0` secara eksplisit. Silakan pilih
  yang paling sedikit merusak konfigurasi lain.
- Firewall hanya membuka 22, 80, 443. API key tetap dipasang walau port tertutup.
- Tidak perlu subdomain `api.` maupun sertifikat kedua.
- Laravel sudah memeriksa kunci terhadap endpoint berkunci lewat
  `php artisan deploy:check` (`/health` sengaja terbuka, jadi hijau di sana
  belum berarti kuncinya diterima).

## 4. Pertanyaan untuk sisi layanan

**T1. Berapa lama satu putaran pelatihan di CPU 4 core?**
`RENCANA_DEPLOY.md` mencatat ~20,5 menit (13,5 sentimen + 7 aspek). Diukur di
mesin apa - CPU saja, atau dengan GPU? Di CPU VPS bersama bisa jauh lebih lama.
Angka ini menentukan dua batas di sisi web: `RetrainModel::$timeout` (7200 s)
dan `retry_after` antrean.

**T2. Analisis yang datang saat pelatihan berjalan.**
Analisis menunggu di `_model_locks`. HTTP dari Laravel ke layanan berbatas
**600 detik**, jadi pelatihan sentimen 13,5 menit sudah cukup untuk membuat
analisis yang datang di awalnya kehabisan waktu, lalu diulang job-nya
(`ProcessTextAnalysis` punya 3 percobaan dengan jeda). Pilihan yang terlihat:

- (a) Dibiarkan. Analisis tertunda atau gagal sekali lalu diulang. Untuk demo
  skripsi mungkin cukup, asal diketahui.
- (b) Layanan langsung membalas **503 + `Retry-After`** bila lock sedang
  dipegang pelatihan, alih-alih menggantung. Tidak membuang 10 menit menunggu.
  Catatan: Laravel **belum** membaca header `Retry-After` - saat ini
  `NLPApiService` mengulang HTTP 3 kali berjarak 1 detik, lalu job mengulang
  dengan jedanya sendiri. Sisi web akan menambahkan: bila 503 membawa
  `Retry-After`, job dilepas kembali ke antrean dengan jeda sebesar itu,
  tanpa menghabiskan jatah percobaan.
- (c) Menaikkan batas HTTP di sisi web. Tidak disukai: satu job macet menahan
  worker lebih lama.

Mana yang menurut sisi layanan paling benar untuk lock yang ada?

**T3. Cara menjalankan di VPS: systemd + venv, atau Docker?**
Dockerfile sudah ada dan rapi (UID 1000, cache HF). Tetapi di VPS tunggal,
systemd + venv lebih ringan dan log-nya menyatu dengan `journalctl`. Tidak ada
preferensi kuat dari sisi web; yang penting `models/` berada di disk host.

**T4. PyTorch versi CPU.**
`torch==2.9.1` dari PyPI di Linux menarik build CUDA (beberapa GB pustaka NVIDIA
yang tidak terpakai tanpa GPU). Apakah aman memasang dari
`--index-url https://download.pytorch.org/whl/cpu` di produksi? Menghemat disk
dan waktu pasang secara berarti.

**T5. Isi `models/` yang wajib disalin.**
Sekarang 1,9 GB: `aspect_baseline.pt`, `aspect_retrained.pt`,
`aspect_retrained.pt.bak_k1`, `aspect_retrained.pt.provenance.json`,
`sentiment_retrained/`. Usul: `.bak_*` tidak ikut. Apakah bobot dasar sentimen
(`crypter70`) diunduh dari HF Hub saat pertama jalan? Bila ya, ke direktori
cache mana - supaya disk dan izinnya disiapkan, dan supaya protokol kolam
(`retrain_from_baseline`) tetap punya titik berangkat setelah server dipasang
ulang.

**T6. Cadangan bobot di luar server.**
Disk VPS permanen tetapi bukan cadangan. Apakah `weights_store.py`
(`HF_WEIGHTS_REPO`, repo privat) layak dipertahankan sebagai cadangan di luar
server, atau cukup salinan berkala `models/` ke laptop?

## 5. Yang akan dikerjakan sisi web

Menunggu jawaban T1 dan T2, lalu:

1. **Antrean terpisah untuk pelatihan.** Sekarang `RetrainModel` dan
   `ProcessTextAnalysis` berbagi satu antrean dengan satu worker, jadi selama
   pelatihan analisis bahkan tidak dimulai. Selain itu `retry_after` (4200 s)
   lebih kecil dari `RetrainModel::$timeout` (7200 s). Usulnya: koneksi
   antrean `training` dengan `retry_after` sendiri + worker kedua. Catatan
   jujur: ini hanya memindahkan antrean dari Laravel ke lock layanan, karena
   analisis tetap menunggu lock. Nilainya bergantung pada jawaban T2.
2. `DEPLOYMENT.md` ditulis ulang untuk satu VPS, termasuk unit systemd layanan
   NLP mengikuti jawaban T3-T5.
3. `NLP_API_URL` / `NLP_API_KEY` produksi, dan `deploy:check --produksi` sebagai
   gerbang akhir.

Tidak ada perubahan kontrak API yang diminta.
