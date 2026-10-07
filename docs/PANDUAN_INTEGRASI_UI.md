# Panduan Integrasi UI Laravel ↔ nlp-api-service

Dokumen titipan untuk pengerjaan antarmuka. Isinya **hanya** hal yang berubah
atau perlu diperhatikan agar UI sesuai dengan perilaku service yang sekarang.

Diverifikasi terhadap service pada 3 September 2026: 325 tes Python + 76 tes
Laravel lulus, analisis gabungan berjalan lewat HTTP nyata.

---

## 0. WAJIB DIKERJAKAN LEBIH DULU

```powershell
cd c:\Skripsi\text-analysis-web
php artisan config:clear
php artisan queue:restart
```

Tanpa ini `retry_after` yang baru tidak terbaca, dan **setiap analisis akan
tetap berjalan ganda** — worker menganggap job yang masih berjalan sudah hilang
lalu melepasnya kembali ke antrean. Ini perbaikan terpenting dari seluruh sesi;
jangan sampai terlewat.

Pastikan pula `.env` memuat:

```
NLP_API_TIMEOUT=600
DB_QUEUE_RETRY_AFTER=2100
```

---

## 1. Data baru yang sudah dikirim API tetapi belum ditampilkan

Semuanya sudah tersimpan di database lewat `predictions` dan `metrics`.
Tinggal ditampilkan.

### 1.1 `review_queue` — prioritas koreksi (nilai tertinggi)

Blok ini ada di `results.sentiment.review_queue`:

```json
{ "threshold": 0.94, "count": 18, "share": 0.225, "indices": [7, 22, 41, ...] }
```

`indices` **sejajar dengan urutan teks masukan** dan sudah diurutkan dari yang
paling tidak yakin. Terukur pada SmSA test: meninjau 17% baris menangkap **64%
kesalahan**, 3,7 kali lebih efisien daripada meninjau acak.

**Saran tampilan:** tombol/tab "Perlu Ditinjau (18)" pada halaman hasil yang
menyaring tabel prediksi ke baris-baris itu, urut sesuai `indices`. Inilah
jalan termurah mengumpulkan data koreksi untuk retraining.

> **JANGAN pakai `review_queue` untuk mengukur akurasi pada studi kasus.**
> Antrean ini sengaja bias ke baris yang model tidak yakin, jadi akurasi yang
> dihitung darinya akan terlalu rendah. Untuk **mengukur**, ambil sampel acak.
> Untuk **mengumpulkan koreksi**, pakai antrean ini.

### 1.2 `method` per prediksi — apakah hasilnya benar-benar dari model

Setiap prediksi sentimen kini membawa `method`:

| nilai | arti | saran UI |
|---|---|---|
| `indobert` | prediksi model sungguhan | normal |
| `rule-based` | model gagal dimuat, jatuh ke daftar kata | **beri peringatan**; mutunya turun ~25 poin |
| `empty` | baris tanpa isi setelah pembersihan | tampilkan abu-abu / "tidak dinilai" |
| `error` | teks gagal dinilai | tandai |

Kalau seluruh baris berbunyi `rule-based`, service sedang berjalan tanpa bobot
model — itu perlu terlihat pengguna, bukan tersembunyi.

### 1.3 Penghitung mutu di `metrics`

```json
{ "total_texts": 82, "total_analyzed": 80, "total_empty": 2,
  "total_truncated": 0, "total_failed": 0, "avg_confidence": 0.9369 }
```

- `total_empty` — baris kosong, **sudah dikeluarkan dari persentase**. Bila > 0,
  sebutkan: "2 baris kosong tidak ikut dihitung".
- `total_truncated` — teks melebihi 512 token, ekornya tidak ikut dinilai.
- `total_failed` — teks yang gagal dinilai model.

Ketiganya nol pada korpus normal; kalau tidak nol, pengguna berhak tahu.

### 1.4 `text` vs `processed_text`

`text` sekarang berisi **teks asli**, `processed_text` berisi hasil pembersihan.
Dulu `text` berisi hasil pembersihan dan Laravel menambalnya sendiri; tambalan
itu sudah disesuaikan. Panel "teks setelah preprocessing" di halaman hasil tetap
berfungsi — jangan menimpa `processed_text` lagi di sisi Laravel.

---

## 2. Yang HARUS diperbaiki pada formulir analisis

### 2.1 Petunjuk jumlah topik keliru secara empiris

Formulir menulis *"Rekomendasi: 3-7 topik untuk hasil optimal"*. Terukur pada
885 komentar, itu **salah** — koherensi dan keberagaman justru lebih baik pada
jumlah topik lebih besar (k=14–20 memberi c_v 0,51 sementara k=4 hanya 0,43).

**Ganti** dengan sesuatu seperti: *"Kosongkan atau pilih Otomatis agar jumlah
topik dicari sendiri berdasarkan koherensi. Untuk laporan yang perlu dibandingkan
antar waktu, tetapkan angka tertentu."*

### 2.2 Mode otomatis (`num_topics = 0`) belum terbuka di UI

API menerima `num_topics: 0` yang berarti "cari sendiri jumlah topik terbaik
dengan memaksimalkan c_v, dibatasi agar tidak ada topik yang mendominasi".
Tambahkan pilihan **"Otomatis"** pada dropdown jumlah topik yang mengirim `0`.

Catatan untuk ekspektasi pengguna: jumlah topik pada mode otomatis **tidak
stabil antar-jalan** (terukur 6–19 topik lintas seed) meski mutunya stabil
(c_v 0,5023 ± 0,0221). Jadi untuk perbandingan antar-waktu, tetapkan angkanya.

### 2.3 Nilai `num_topics` yang ditolak

`num_topics = 1` **ditolak** API (HTTP 422). Valid: `0` (otomatis) atau `2–20`.
Pastikan dropdown tidak pernah mengirim 1.

### 2.4 Pratinjau preprocessing sekarang bisa jujur

`POST /api/preprocess` menerima parameter baru `task`:

| `task` | untuk modul | efek |
|---|---|---|
| `transformer` | Sentimen | stemming & stopword **dipaksa mati** |
| `bag_of_words` | Topik | stemming & stopword **dipaksa aktif** |
| `span` | Aspek | teks dikembalikan **utuh** |
| *(dikosongkan)* | — | perilaku lama: konfigurasi apa adanya |

**Kenapa penting:** tanpa `task`, pratinjau berbohong. Pengguna menyalakan
stemming, melihat teks ter-stem, lalu menjalankan analisis sentimen yang
diam-diam mematikannya. Respons juga membawa `applied_policy` berisi opsi mana
yang ditimpa, sehingga UI bisa menjelaskan: *"Stemming dinonaktifkan untuk
analisis sentimen karena merusak deteksi negasi."*

**Saran:** kirim `task` sesuai jenis analisis yang sedang dipilih pengguna di
formulir.

---

## 3. Kesiapan model dan cold start

Bobot model dimuat **malas**. Permintaan pertama tiap jenis membayar biaya muat
model — di kontainer baru bisa memakan menit, dan dari sisi UI itu tidak bisa
dibedakan dari analisis yang menggantung.

- `GET /health` kini membawa `weights_loaded`:
  ```json
  { "sentiment": true, "aspect": true, "topic": false }
  ```
- `POST /api/warmup` memuat semuanya (terukur hangat: **6,7 detik**).
- `ProcessTextAnalysis` sudah memanggil `warmUp()` otomatis sebelum analisis.

**Saran UI:** pada halaman status/koneksi, tampilkan kesiapan per model dan
tombol "Panaskan Model". Bila ada yang `false` saat pengguna memulai analisis
besar, tampilkan "Menyiapkan model..." alih-alih membiarkan progres diam.

`GET /health` juga membawa provenance model sentimen — berguna untuk halaman
"Tentang" atau lampiran skripsi:

```json
{ "sentiment_source": "crypter70/IndoBERT-Sentiment-Analysis",
  "sentiment_base": "indobenchmark/indobert-base-p1",
  "sentiment_temperature": 2.7748,
  "sentiment_review_threshold": 0.94 }
```

---

## 4. Batas yang ditegakkan API

| batas | nilai | perilaku bila dilanggar |
|---|---|---|
| Jumlah teks per permintaan | 10 000 | HTTP 422 |
| Panjang satu teks | 10 000 karakter | HTTP 422, menyebut indeks pertama |
| `num_topics` | 0 atau 2–20 | HTTP 422 |
| `mode` | `automatic` / `rule-based` | HTTP 422 |

Sebaiknya UI memvalidasi lebih dulu agar pesan galatnya ramah, bukan menunggu
422 dari API.

---

## 5. Angka yang boleh ditampilkan sebagai mutu analisis

Modul topik mengirim blok `quality` yang belum ditampilkan:

```json
{ "c_v": 0.5032, "c_npmi": -0.394, "diversity": 0.9222, "outlier_rate": 0.1638 }
```

**c_v adalah angka utama** (koherensi topik). Rambu penafsirannya:

| c_v | tafsiran |
|---|---|
| < 0,30 | tidak koheren |
| 0,30–0,40 | lemah |
| **0,40–0,55** | **baik untuk teks pendek/tidak baku** |
| 0,55–0,70 | sangat baik (lazim pada dokumen panjang) |
| > 0,70 | patut dicurigai |

Bila ditampilkan, sertakan konteksnya — c_v **tidak sebanding antar korpus**,
jadi jangan disajikan seolah nilai absolut yang bisa dibandingkan dengan
penelitian lain.

`outlier_rate` = porsi dokumen yang tidak masuk topik mana pun; wajar di
kisaran 5–20%.

---

## 6. Hal yang JANGAN diubah dari sisi Laravel

1. **Jangan memfilter atau mengurutkan ulang daftar `texts`** sebelum dikirim.
   Seluruh sistem bergantung pada kontrak: N teks masuk → N hasil keluar, dan
   `predictions[i]`, `document_aspects[i]`, `document_topics[i]` semuanya
   menunjuk teks ke-i. Membuang baris kosong akan menggeser seluruh indeks
   setelahnya dan menyimpan hasil pada teks yang salah.

2. **Jangan menimpa `processed_text`** — lihat §1.4.

3. **Jangan menghitung distribusi sentimen sendiri tanpa menyaring
   `method: 'empty'`.** `NLPApiService::scorablePredictions()` sudah
   melakukannya; ikuti pola itu bila menambah perhitungan baru.

4. **Jangan menaikkan `NLP_API_TIMEOUT` melebihi timeout job (1800 detik).**
   Rantainya sengaja disusun agar job — yang bisa mencatat sebab dan mencoba
   ulang — yang menghentikan pekerjaan macet, bukan permintaan HTTP yang
   menggantung.

---

## 7. Ringkasan endpoint

| endpoint | keterangan |
|---|---|
| `GET /health` | status, `weights_loaded`, provenance model |
| `POST /api/warmup` | muat bobot lebih dulu |
| `POST /api/preprocess` | pratinjau; terima `task` |
| `POST /api/analyze/sentiment` | + `review_queue`, `method`, `processed_text` |
| `POST /api/analyze/aspect` | mendukung `predefined_aspects` + `mode` |
| `POST /api/analyze/topic` | `num_topics: 0` = otomatis; + `quality` |
| `POST /api/analyze/combined` | keempat blok sekaligus, termasuk `association` |
| `POST /api/analyze/association` | PMI aspek–topik untuk jalur batch |
| `POST /api/retrain/{sentiment,aspect}` | active learning |
| `POST /api/retrain/preview` | laporan data latih **tanpa** melatih |

`/api/retrain/preview` belum dipakai Laravel. Ia melaporkan distribusi label,
rasio ketimpangan, komposisi split, bobot kelas, dan peringatan — layak
dipanggil sebelum tombol "Latih Ulang" ditekan, agar pengguna tahu datanya
memadai atau belum.

---

## 8. Urutan pengerjaan yang saya sarankan

1. `config:clear` + `queue:restart` (§0) — tanpa ini sisanya percuma.
2. Tab "Perlu Ditinjau" dari `review_queue` (§1.1) — dampak terbesar, dan
   langsung menyiapkan data untuk studi kasus.
3. Perbaiki petunjuk jumlah topik + tambah opsi "Otomatis" (§2.1, §2.2).
4. Tampilkan `total_empty` / `total_truncated` / `total_failed` (§1.3).
5. Kirim `task` pada pratinjau preprocessing (§2.4).
6. Tampilkan `quality` topik dan kesiapan model (§5, §3).
