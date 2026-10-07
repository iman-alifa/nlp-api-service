# Rencana deploy layanan NLP

> **DIGANTIKAN (23 September 2026).** Rencana yang disepakati sisi layanan dan
> sisi web adalah **satu VPS** untuk web + layanan NLP. Langkah yang berlaku ada
> di `../text-analysis-web/DEPLOYMENT.md`; alasan teknisnya di
> `docs/JAWABAN_HOSTING_SATU_VPS.md`. Dokumen ini disimpan sebagai riwayat
> pertimbangan (HF Spaces ternyata butuh PRO untuk SDK Docker/Gradio).

Ditulis 13 September 2026. Ditujukan untuk menjawab dua hal: bagaimana layanan
ini bisa diakses banyak orang, dan apakah layak berkontribusi ke Hugging Face.

---

## 0. Satu hal yang harus dikerjakan lebih dulu

**Tidak ada autentikasi pada satu endpoint pun.** `app/main.py` memasang
`CORSMiddleware` dan tidak ada yang lain - tidak ada `Depends`, tidak ada API
key, tidak ada bearer token.

CORS hanya membatasi **peramban**. `curl` mengabaikannya sepenuhnya. Jadi
begitu layanan ini punya URL publik, siapa pun yang menemukannya bisa:

```bash
curl -X POST https://url-anda/api/retrain/sentiment \
  -d '[{"text":"...","label":"positive"}, ...]'
```

dan **menimpa bobot model**. Ini model poisoning, dan tidak butuh keahlian
khusus - endpoint-nya terdokumentasi sendiri di `/docs`.

Risiko kedua: `/api/analyze/combined` menerima hingga `max_batch_size` teks
berisi 10.000 karakter, dijalankan pada CPU. Beberapa permintaan sekaligus
cukup untuk membuat layanan tidak responsif, dan karena inferensi berjalan
serial di event loop, antreannya menumpuk.

**Minimum sebelum publik:**

1. **API key pada semua endpoint yang mengubah keadaan** (`/api/retrain/*`,
   `/api/warmup`) - `Depends` dengan header `X-API-Key` dibandingkan terhadap
   `settings.api_key`. Laravel sudah satu-satunya pemanggil, jadi ini hanya
   menambah satu header di `NLPApiService`.
2. **Nonaktifkan `/docs` dan `/redoc` di produksi** (`docs_url=None` ketika
   `app_env == 'production'`). Tidak perlu mengumumkan permukaan serangnya.
3. **Batas laju** pada endpoint analisis - per IP, sederhana saja.
4. `DEBUG=False` (sudah), dan pastikan `allowed_origins` tidak berisi `*`.

Tanpa (1), tidak ada gunanya melanjutkan ke bagian mana pun di bawah.

---

## 1. Kendala yang melekat pada rancangan sekarang

Empat hal ini bukan kekurangan yang bisa "di-tuning", melainkan konsekuensi
rancangan yang memang disengaja. Semuanya menentukan bentuk deploy-nya.

### 1.1 `--workers 1` bersifat arsitektural

Keadaan model - termasuk bobot hasil retraining - hidup di **global tingkat
proses**. Dua worker berarti dua salinan bobot (RAM dua kali lipat) **dan**
keadaan yang menyimpang setelah retrain: satu worker memuat ulang bobot baru,
satunya tidak. Karena itu `Procfile`, `nixpacks.toml`, dan `railway.json`
semuanya memakukannya.

Artinya: **satu kontainer tidak bisa diskalakan ke dalam.** Menambah kapasitas
berarti menambah kontainer, dan itu memunculkan masalah 1.3.

### 1.2 Retraining mengunci inferensi

`_use_models()` memegang satu `asyncio.Lock` per model, dipakai bersama oleh
jalur inferensi dan jalur pelatihan. Itu disengaja - tanpa itu, analisis yang
datang di tengah pelatihan membaca bobot setengah terlatih dengan dropout
aktif, dan mengembalikan angka yang salah tanpa galat apa pun.

Konsekuensinya untuk banyak pengguna: satu putaran pelatihan penuh pada kolam
sekarang memakan **~20,5 menit** (terukur: 13,5 menit sentimen + 7 menit
aspek). Selama itu setiap analisis mengantre.

### 1.3 Bobot hasil retraining ada di disk lokal kontainer

`models/` masuk `.gitignore`, jadi bobotnya tidak ikut image. Pada Railway,
redeploy menghapusnya kecuali ada volume yang dipasang di `MODEL_PATH`. Pada
Hugging Face Spaces, penyimpanannya bersifat sementara kecuali membayar
persistent storage.

**Kalau ini dilupakan, hasil enam iterasi active learning hilang pada deploy
berikutnya** dan layanan diam-diam kembali ke bobot publik.

### 1.4 Anggaran memori

Empat model seukuran BERT bisa hidup bersamaan: sentimen, aspek, encoder
pemeta semantik aspek, dan encoder topik. Masing-masing ~473 MB.

| keadaan | perkiraan RAM |
|---|---|
| semua bobot termuat | ~1,9 GB + runtime torch (~0,5 GB) |
| ditambah pelatihan (gradien + state Adam) | **~4 GB puncak** |

Inilah angka yang menentukan pilihan platform, dan ini pula yang dulu membuat
kontainer terbunuh ketika dua model sentimen tidak sengaja termuat bersamaan.

---

## 2. Rencana bertahap

### Tahap 1 - demo skripsi (yang sebenarnya Anda butuhkan sekarang)

Satu kontainer, persis konfigurasi Railway yang sudah ada, ditambah:

| item | nilai |
|---|---|
| volume | dipasang di `MODEL_PATH` (mis. `/data/models`), minimal 3 GB |
| RAM | minimal 4 GB (pelatihan), 2,5 GB kalau retraining dimatikan |
| `DEBUG` | `False` |
| `API_KEY` | disetel, dan Laravel mengirimnya |
| healthcheck | `/health`, timeout 300 s (sudah) |
| warmup | `POST /api/warmup` setelah boot (Laravel sudah memanggilnya) |

Komponen penuh yang perlu dihosting:

```
[pengguna] → Laravel (web)  → MySQL
                ↓              ↑
           queue worker  ──────┘
                ↓
        layanan NLP (FastAPI, 1 worker, volume)
```

Empat komponen: web, worker antrean, basis data, layanan NLP. Queue worker
**wajib** terpisah - `ProcessTextAnalysis` punya `$timeout = 1800`, dan rantai
timeout-nya sudah dikalibrasi (lihat `QueueTimeoutConfigTest`).

Ini cukup untuk sidang, demo ke penguji, dan puluhan pengguna yang tidak
bersamaan.

### Tahap 2 - kalau benar-benar banyak pengguna bersamaan

Pisahkan peran melayani dari peran melatih. Ini pola baku *training/serving
split*, dan pada kode ini perubahannya kecil:

```
                  ┌──────────────────────┐
   analisis  →    │ replika inferensi ×N │  (read-only, retrain DIMATIKAN)
                  └──────────┬───────────┘
                             │ memuat bobot
                  ┌──────────▼───────────┐
                  │  penyimpanan bobot   │  (HF Hub / object storage)
                  └──────────▲───────────┘
                             │ menulis bobot
                  ┌──────────┴───────────┐
   retrain   →    │  satu kontainer      │  (1 worker, RAM besar)
                  │  pelatih             │
                  └──────────────────────┘
```

Yang perlu ditambahkan di kode:

- Variabel `ROLE` (`serve` / `train`). Pada `serve`, endpoint `/api/retrain/*`
  mengembalikan 405 - replika tidak boleh menulis bobot.
- Berkas versi bobot di penyimpanan bersama; replika memeriksanya berkala dan
  memuat ulang ketika berubah. `provenance.json` yang sudah ada sudah memuat
  `sidik_jari_kolam` dan `dicatat_pada`, jadi tinggal dipakai sebagai penanda
  versi.

Dengan ini `--workers 1` tetap berlaku per kontainer, tetapi skalanya keluar
lewat jumlah replika, dan pelatihan tidak lagi mengunci pengguna.

**Jujur: Tahap 2 kemungkinan besar tidak Anda perlukan untuk skripsi.** Ia
layak ditulis di bab "saran pengembangan", bukan dikerjakan sekarang.

---

## 3. Hugging Face - dua hal yang berbeda

### 3.1 HF Hub: mengunggah bobot hasil penelitian

**Ini kontribusi yang nyata dan paling layak dikerjakan.**

Model ekstraksi aspek berbahasa Indonesia berbasis penandaan BIO tingkat token
**jarang tersedia publik**. Itu celah yang riil, dan checkpoint Anda mengisinya.
Model sentimennya kurang baru (adaptasi domain atas `crypter70`), tetapi tetap
berguna sebagai contoh domain komentar YouTube lembaga publik.

Nilai tambahnya untuk naskah: artefak yang **dapat disitasi** dan dapat
diverifikasi orang lain - penguji menyukai ini, dan ia menjawab pertanyaan
"apakah hasilnya bisa direproduksi" tanpa perlu mengirim berkas 473 MB.

**Yang harus diperiksa lebih dulu, jangan dilewati:**

1. **Lisensi model dasar.** Bobot Anda adalah turunan dari
   `indobenchmark/indobert-base-p1` dan `crypter70/IndoBERT-Sentiment-Analysis`.
   Buka halaman masing-masing, baca lisensinya, dan ikuti kewajibannya
   (atribusi, lisensi yang sama, atau larangan komersial). Turunan mewarisi
   kewajiban itu.
2. **JANGAN mengunggah korpus komentar YouTube maupun golden dataset begitu
   saja.** Bobot model adalah turunan; teks komentar adalah karya pengguna lain
   dan tunduk pada ketentuan YouTube serta soal data pribadi. Kalau ingin
   melepas anotasinya, pertimbangkan melepas **label tanpa teks mentah** (id
   video + offset), dan putuskan itu secara sadar, bukan sebagai efek samping.

**Isi model card - yang wajib ada:**

- Model dasar dan lisensinya.
- Data pelatihan: 481 koreksi manusia (sentimen) / 398 (aspek) dari komentar
  YouTube enam lembaga pemerintah Indonesia, dikumpulkan lewat active learning
  dengan uncertainty sampling.
- Metrik pada golden dataset 300 baris: aspek micro-F1 **74,38**, sentimen
  akurasi **85,67**.
- **Keterbatasan, ditulis menonjol.** Terutama: checkpoint aspek yang dilatih
  dari data proyeksi leksikon memberi F1 0,97 in-domain tetapi **0,029
  span-level pada TermA** - ia menghafal daftar istilah. Orang yang mengunduh
  model ini berhak tahu itu sebelum memakainya di domain lain. Sertakan juga
  bahwa perbaikan sentimen **tidak signifikan secara statistik**
  (McNemar p=0,3222).
- Cara memakainya, dengan kode contoh.

Model card yang menyembunyikan angka 0,029 itu akan menyesatkan orang. Yang
mencantumkannya justru membuat kontribusinya kredibel.

### 3.2 HF Spaces: menghosting layanannya

Spaces bisa menjalankan FastAPI lewat Docker SDK. Tingkat CPU gratisnya lapang
soal RAM - **periksa batas yang berlaku sekarang**, karena ketentuannya
berubah.

Dua hal yang perlu diperhatikan:

- **Space gratis tidur setelah tidak dipakai**, dan bangunnya memakan waktu
  memuat model (menit, karena bobotnya diunduh). Untuk demo sidang, panggil
  `/api/warmup` sebelum mulai.
- **Penyimpanan bersifat sementara.** Ini menabrak kendala 1.3 secara
  langsung.

**Tetapi ada jalan keluar yang justru elegan:** jadikan HF Hub sebagai
penyimpanan bobot. Setelah retraining berhasil, dorong checkpoint ke repo Hub
privat; saat boot, tarik dari sana. Ini sekaligus:

- menyelesaikan masalah penyimpanan sementara di Spaces,
- menyelesaikan masalah volume di Railway,
- memberi **versioning bobot gratis** (Hub berbasis git), sehingga setiap
  iterasi active learning punya commit sendiri - jejak audit yang jauh lebih
  baik daripada `provenance.json` di disk lokal,
- dan menjadi "penyimpanan bobot bersama" yang dibutuhkan Tahap 2.

Paketnya `huggingface_hub`, belum ada di `requirements.txt`. Perlu
`HF_TOKEN` sebagai secret.

---

## 4. Rekomendasi urutan

| prioritas | pekerjaan | alasan |
|---|---|---|
| **1** | API key + matikan `/docs` di produksi | tanpa ini, deploy publik berarti membiarkan siapa pun menimpa model |
| **2** | Volume di `MODEL_PATH` (atau HF Hub sebagai penyimpanan bobot) | tanpa ini, hasil enam iterasi hilang pada redeploy pertama |
| **3** | Deploy Tahap 1 (4 komponen) | cukup untuk sidang dan demo |
| **4** | Unggah dua model ke HF Hub + model card | kontribusi nyata, artefak yang dapat disitasi |
| 5 | Batas laju | mencegah penyalahgunaan biaya CPU |
| 6 | Tahap 2 (pisah serve/train) | tulis di bab saran, kerjakan hanya bila benar-benar perlu |

Butir 1 dan 2 sama-sama **wajib** sebelum URL publik dibagikan. Butir 4 bisa
dikerjakan kapan saja dan tidak bergantung pada deploy.
