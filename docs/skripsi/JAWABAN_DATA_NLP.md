# Jawaban Permintaan Data Bab IV (nlp-api-service)

Disusun 27 September 2026 oleh sesi Claude Code di `nlp-api-service`.
Semua angka diambil langsung dari kode, log, basis data, dan berkas hasil,
bukan disalin dari dokumen skripsi. Model aktif dan konfigurasi produksi tidak
diubah selama pengumpulan data ini.

Penomoran iterasi mengikuti permintaan:

| Nama di naskah | Nama di aplikasi | Golden Dataset | Run evaluasi |
|---|---|---|---|
| Iterasi 0 | Iterasi 0 | GD1 | 2 |
| Iterasi 1–6 | Iterasi 1–6 | GD1 | 3, 5, 6, 7, 8, 9 |
| Iterasi 6 diukur pada GD2 | Iterasi 0 | GD2 | 10 |
| Iterasi 7 | Iterasi 1 | GD2 | 11 |
| Iterasi 8 | Iterasi 2 | GD2 | 14 |

Model akhir = sentimen iterasi 8 (pelatihan #18) + aspek iterasi 7
(pelatihan #17).

Metode uji yang dipakai di seluruh jawaban ini:

- **Sentimen:** McNemar binomial eksak dua sisi.
- **Aspek:** bootstrap berpasangan atas baris, statistik micro-F1 ketat
  (bentuk dasar Sastrawi kedua sisi), **10.000 resampel, `numpy
  default_rng(42)`, persentil 2,5–97,5**, ditambah McNemar untuk exact match.

---

## 1. Hasil analisis lanjutan

Label **[BAB IV]** berarti layak dimasukkan sebagai temuan; label **[TEKNIS]**
berarti cukup sebagai keterangan atau lampiran.

### 1a. Kesepakatan antaranotator Golden Dataset kedua [BAB IV]

- **Pertanyaan:** seberapa andal label emas GD2?
- **Metode:**
  - Anotator 1 (label emas) dan anotator 2 bekerja terpisah. Anotator 2 tidak
    melihat label anotator 1.
  - Keduanya memakai berkas yang menampilkan tebakan model setelah anotator
    memutuskan.
  - Sentimen diukur dengan Cohen's kappa. Aspek diukur dengan F1 antaranotator
    pada bentuk dasar.
- **Angka (n = 300):**
  - Sentimen: κ = **0,698** (kuat, Landis & Koch), kesepakatan mentah 81,7%.
  - Pada 233 baris yang tidak diubah anotator mana pun setelah tebakan
    terlihat: κ = 0,789, kesepakatan mentah 88,0%.
  - Aspek: F1 antaranotator **62,97**, exact match 42,0%. Anotator 1 menandai
    569 aspek, anotator 2 menandai 476.
- **Temuan:** kesepakatan sentimen setara GD1 (κ 0,669). Kesepakatan aspek
  jauh lebih rendah, yang menunjukkan definisi aspek masih longgar antarmanusia.

### 1b. Bias jangkar pada anotasi GD2 [BAB IV, sebagai keterbatasan]

- **Pertanyaan:** apakah menampilkan tebakan model memengaruhi label emas?
- **Metode:** mencatat label yang diubah setelah tebakan terlihat, dan mengukur
  kedekatan masing-masing anotator dengan tebakan tersebut.
- **Angka (n = 300):**

  | Ukuran | Anotator 1 | Anotator 2 |
  |---|---|---|
  | Label sentimen diubah setelah tebakan terlihat | **44 baris (14,7%)**, **42 di antaranya ke arah tebakan model** | 29 baris |
  | Daftar aspek diubah setelah tebakan terlihat | 66 baris (22%) | – |
  | Sentimen sama dengan tebakan model | **83,0%** | 70,3% |
  | F1 aspek terhadap tebakan model | **64,6** | 40,6 |

- **Temuan:** label emas anotator 1 lebih dekat ke tebakan model. Model
  mencapai F1 aspek sekitar 76 terhadap anotator 1, lebih tinggi daripada
  kesepakatan antarmanusia (63). Bias ini **menaikkan angka absolut** evaluasi
  pada GD2, tetapi **tidak menjelaskan** penurunan iterasi 7 → 8 (lihat butir 8).

### 1c. Uji aspek fase studi kasus [BAB IV — temuan utama tujuan 3]

- **Pertanyaan:** apakah enam putaran koreksi meningkatkan ekstraksi aspek?
- **Metode:** iterasi 0 lawan 6 pada GD1 (n = 300). Uji ini ditetapkan di muka
  sebagai uji utama.
- **Angka:**
  - Micro-F1 ketat **51,84 → 74,38 (+22,54 poin)**, CI 95% **[+18,33; +27,03]**.
  - Exact match 27,67% → 40,67%; 62 baris membaik, 23 memburuk, McNemar
    **p = 2,8×10⁻⁵**.
  - Recall (ketat) 36,49% → 74,86%; presisi 89,44% → 73,90%.
- **Temuan:** peningkatan signifikan, terutama lewat recall. Model berubah dari
  sangat konservatif menjadi seimbang.
- **Pembanding sentimen, uji yang sama:** 29 membaik, 21 memburuk,
  **p = 0,322** (tidak signifikan). Akurasi +2,67 poin, macro-F1 +2,71 poin.

### 1d. Pemilihan berbasis ketidakpastian pada pengguna asli [BAB IV]

- **Pertanyaan:** apakah antrean tinjauan juga memusatkan kesalahan pada
  pengguna nyata?
- **Metode:** membandingkan baris yang dikoreksi pengguna asli dengan seluruh
  prediksi di analisis mereka.
- **Angka:**
  - 13 akun pengguna asli; 12 menjalankan 16 analisis (1.047 teks).
  - **115 koreksi dari 7 pengguna** pada 24 September 2026.
  - Median keyakinan: baris yang dikoreksi **0,589**, seluruh 889 prediksi
    0,910.
  - **46,1%** label sentimen berubah (fase studi kasus: 49,7%).
  - Menurut pita keyakinan:
    - 0,00–0,50: 8 dari 23 (34,8%)
    - 0,50–0,70: 43 dari 89 (48,3%)
    - 0,70–0,90: 2 dari 3
  - Daftar aspek hanya diubah pada **13,9%** baris (fase studi kasus: 58,5%).
    Rata-rata 0,97 aspek per teks (fase studi kasus: 2,07).
- **Temuan:** mekanisme pemilihan bekerja sama baiknya pada pengguna nyata,
  tetapi pengguna hampir hanya mengoreksi sentimen. Koreksi aspek dari mereka
  sangat sedikit.

### 1e. Keputusan model akhir aspek [BAB IV, ringkas]

- Aspek iterasi 8 turun signifikan dibanding iterasi 7 (butir 8), sehingga
  bobot aspek dikembalikan ke iterasi 7.
- Bobot diambil dari commit penyimpanan bobot `9116ca2fd9a0` (sha256
  `a5366e0140ed…`, sidik jari kolam `e3011f86d97550a5`, 450 sampel).
- Model sentimen iterasi 8 tetap dipakai.
- Pemilihan ini memakai GD2, sehingga angka GD2 untuk model akhir sedikit
  optimistis. **[BAB IV, keterbatasan]**
- **[TEKNIS]** Verifikasi: prediksi aspek setelah pemulihan identik 300/300
  dengan run iterasi 7 dan berbeda di 76 baris dari run iterasi 8.

### 1f. Hal teknis yang tidak perlu masuk Bab IV [TEKNIS]

- Evaluasi iterasi 8 sempat gagal karena layanan kehabisan memori setelah
  pelatihan. Masalahnya sudah diperbaiki dan run diulang. Tidak memengaruhi
  angka mana pun.
- Semua CI aspek dihitung ulang dengan satu metode seragam (butir 2).

---

## 2. Seed dan jumlah resampel bootstrap aspek iterasi 0 lawan 6

Ada tiga perhitungan, dan ketiganya mengukur hal yang sama: bootstrap
berpasangan atas 300 baris GD1, selisih micro-F1 ketat iterasi 6 − iterasi 0.

| CI 95% | Resampel | Seed / generator | Asal | Bisa direproduksi? |
|---|---|---|---|---|
| +18,16 s.d. +26,94 | 5.000 | **tidak tercatat** | `docs/RINGKASAN_TEMUAN_UNTUK_SKRIPSI.md` (sesi sebelumnya) | Tidak persis |
| +18,25 s.d. +26,79 | 2.000 | Python `random.seed(42)`, kuantil indeks 50/1949 | perhitungan 27 September (sesi ini) | Ya |
| **+18,33 s.d. +27,03** | **10.000** | **`numpy.random.default_rng(42)`, persentil 2,5/97,5** | dihitung ulang untuk jawaban ini | **Ya** |

Pembanding dengan generator numpy yang sama pada seed 42:

- 2.000 resampel: [+18,08; +26,87]
- 5.000 resampel: [+18,21; +27,10]

Perbedaan antarangka hanya ±0,2 poin, yaitu variasi Monte Carlo. Batas
bawahnya konsisten sekitar +18, jadi kesimpulan tidak berubah.

**Rekomendasi:** pakai **CI [+18,33; +27,03]** dan tuliskan metodenya:
bootstrap berpasangan atas baris, 10.000 resampel, seed 42, persentil. Angka
+18,16/+26,94 jangan dipakai lagi, karena seed-nya tidak tercatat.

Semua CI aspek dengan metode yang sama:

| Perbandingan | GD | Micro-F1 ketat | Selisih | CI 95% |
|---|---|---|---|---|
| Iterasi 0 → 6 | GD1 | 51,84 → 74,38 | +22,54 | [+18,33; +27,03] |
| Iterasi 6 → 7 | GD2 | 76,62 → 76,45 | −0,17 | [−1,46; +1,12] |
| Iterasi 6 → 8 | GD2 | 76,62 → 73,21 | −3,40 | [−5,15; −1,74] |
| Iterasi 7 → 8 | GD2 | 76,45 → 73,21 | −3,23 | [−4,96; −1,57] |

---

## 3. Pelabelan korpus K2 oleh LLM

**Model LLM.** `gemini-1.5-flash`, lewat skrip `label_aspect.py`. Sumber
informasi ini adalah docstring `scripts/annotate_aspects.py` baris 6. Versi
atau tanggal rilis API yang lebih rinci **TIDAK TERSEDIA**: `label_aspect.py`
tidak ada di repo, dan keluaran mentahnya tidak mencatat versi model.

**Instruksi pelabelan.** Teks *prompt* **TIDAK TERSEDIA** karena alasan yang
sama. Yang tercatat di docstring hanya:

- satu *prompt* memuat **200 kalimat**;
- LLM mengembalikan frasa aspek per kalimat;
- span dicari dengan `text.find()`, sehingga hanya kemunculan pertama yang
  ditandai.

**Masalah pada keluaran LLM mentah** (docstring):

- 56% kemunculan istilahnya sendiri tidak ditandai;
- 12,4% span berkepala verba;
- 6% span terlalu panjang (sampai 17 kata) atau memuat kurung penjelas.

**Jumlah data** (`data/aspect_news_*.json`):

| Tahap | Jumlah |
|---|---|
| Teks korpus berita | 2.126 |
| Kandidat istilah dari keluaran LLM | 2.487 |
| Istilah ditolak penyaringan | 446 |
| **Istilah diterima** | **2.041** |
| Teks beraspek setelah proyeksi konsisten | **1.939 (91,2%)** |
| Span | **4.203** |
| Pembagian K2 | latih 1.551, validasi 388 (= 1.939) |

**Kriteria penyaringan** (`annotate_aspects.py: reject_reason`, jumlah dari
`rejected_counts`):

| Alasan | Ditolak | Contoh |
|---|---|---|
| Berkepala verba (awalan me-/di-/ter-/ber-, diverifikasi dengan stem Sastrawi; ada daftar pengecualian nomina) | 288 | "merelokasi StarOne dan Flexi", "dipilih kembali" |
| Nama diri | 85 | "Mentari", "SMS Update" |
| Mengandung angka | 33 | "layanan 3G", "satelit Telkom-2" |
| Berkepala kata opini | 25 | "lolos ke Liga Champions", "menang empat kali" |
| Berkepala kata fungsi | 14 | "tak terkalahkan", "paling panas" |
| Berkepala satuan waktu | 1 | – |

**Normalisasi sebelum penyaringan:**

- membuang kurung penjelas dan tanda baca di ujung;
- membuang angka dan satuan waktu di ekor;
- maksimal 4 kata (`MAX_TERM_WORDS`) dan minimal 3 karakter (`MIN_TERM_CHARS`).

Setelah itu setiap istilah yang diterima **diproyeksikan ke seluruh
kemunculannya di korpus**, sehingga pelabelannya konsisten.

---

## 4. Asal angka c_v 0,5032 / outlier 5,9% dan c_v 0,4998 / outlier 12,34%

**Kedua angka pada capaian *Modelling* berasal dari dua run yang berbeda.**
Keduanya memakai korpus penyetelan yang sama,
`data/external/youtube/test_gold.json` (885 komentar YouTube, satu korpus
gabungan), dan tercatat di `data/experiments/topic_results.json`.

| Angka di draf | Run | Waktu | Konfigurasi | c_v | Outlier | Topik |
|---|---|---|---|---|---|---|
| c_v **0,5032** | `final-preprocessing` | 2 Sep 2026 23:34 | jumlah topik **otomatis** (`requested_topics = 0`), seed 42 | **0,5032** | **16,38%** | 9 |
| outlier **5,9%** | `v10-slang` | 2 Sep 2026 11:26 | jumlah topik **tetap 15**, seed 42 | 0,5061 | **5,88%** | 9 |

Jadi pasangan "c_v 0,5032 dan outlier 5,9%" **tidak berasal dari satu
pengukuran**. Pasangan yang sah: `final-preprocessing` = c_v 0,5032 dengan
outlier 16,38%, atau `v10-slang` = c_v 0,5061 dengan outlier 5,88%.

**Korpus studi kasus (c_v 0,4998, outlier 12,34%):**

- Rerata dari **enam model topik terpisah**, satu per lembaga, atas 8.385
  komentar dengan total 65 topik.
- Dijalankan lewat aplikasi pada **5 September 2026**: tipe *combined*,
  `num_topics = 0` (otomatis), seed topik 42 (`settings.topic_seed`).
- Nilai per analisis (dari hasil analisis tersimpan):

  | Lembaga | c_v | Outlier | Topik |
  |---|---|---|---|
  | TNI | 0,4710 | 25,28% | 12 |
  | Presiden | 0,5281 | 3,98% | 5 |
  | Kejaksaan Agung | 0,5184 | 1,21% | 17 |
  | KPK | 0,4677 | 10,60% | 9 |
  | Polri | 0,5392 | 12,94% | 5 |
  | DPR | 0,4742 | 20,05% | 17 |
  | **Rerata** | **0,4998** | **12,34%** | 65 total |

**Mengapa berbeda:**

- Korpus berbeda: 885 komentar gabungan untuk penyetelan, enam korpus
  lembaga untuk studi kasus.
- Ukuran korpus berbeda. c_v sangat bergantung ukuran korpus: pada korpus dan
  konfigurasi yang sama, n = 300 memberi 0,3597 dan n = 1.139 memberi 0,4498.
- Rerata enam model tidak setara dengan satu model.

**Saran:** capaian *Modelling* memakai run `final-preprocessing` (c_v 0,5032,
outlier 16,38%) sebagai hasil penyetelan, sedangkan capaian *Evaluation*
memakai 0,4998 / 12,34% dari korpus studi kasus.

---

## 5. Parameter pelatihan ulang yang berjalan

Sumbernya kode (`app/config.py`, `app/schemas/request_schemas.py`,
`app/main.py`, kedua `_retrain_sync`) dan muatan dari Laravel
(`RetrainingService`). Nilainya dikonfirmasi dengan catatan pelatihan #16–#19.

| Parameter | Sentimen | Aspek | Bukti di catatan pelatihan |
|---|---|---|---|
| Titik awal | bobot dasar `crypter70` (`retrain_from_baseline = true`) | bobot dasar `aspect_baseline.pt` (K6) | log: "Protokol kolam: melatih dari bobot dasar …" |
| Laju pembelajaran | 2×10⁻⁵ (Laravel mengirim `0.00002`) | 2×10⁻⁵ | – |
| Optimizer | AdamW, *weight decay* 0,01, *gradient clipping* 1,0 | AdamW, *weight decay* 0,01, *warmup* linear 10% | – |
| Epoch | 3 (bawaan Laravel `auto_retrain.epochs` dan formulir manual) | 3 | `epoch_history` 3 entri |
| Pemilihan epoch | epoch terbaik menurut *weighted*-F1 validasi | epoch terbaik menurut F1 token aspek validasi | #17 epoch 2, #19 epoch 3, #18 epoch 2, #16 epoch 3 |
| Batch size | 8 (tetap di `DataLoader`) | 16 (bawaan `retrain()`, tidak dikirim endpoint) | – |
| Panjang token latih | dinamis mengikuti data, kelipatan 8, ≤ 512 | 128 (label BIO) | log #16: "Panjang padding retraining: 184 token" |
| Bobot kelas | inversi frekuensi (selalu) | tidak dipakai (`use_class_weights = False`) | `class_weights` ada di hasil #16 |
| Rasio validasi | 0,2, split berstrata (label) | 0,2, split berstrata (ada/tidak aspek) | val 111/555, 119/591, 90/450, 91/456 |
| Seed | 42 (`retrain_seed`) | 42 | `seed` di hasil |
| Metrik keputusan penerimaan | *weighted*-F1 validasi, dibanding **bobot awal pelatihan (bobot dasar)** | F1 token aspek validasi, dibanding bobot dasar | `weighted_f1_delta`, `f1_delta` |
| `retrain_min_delta` | 0,0 (bawaan kode) | 0,0 | – |
| `calibration_min_samples` | 50 (kalibrasi suhu hanya diukur ulang bila validasi ≥ 50) | – (aspek tidak dikalibrasi) | #16 dan #18 memakai suhu hasil fit (val 111 dan 119) |

**Keterbatasan data:** `.env` produksi bisa menimpa nilai bawaan, dan sesi ini
tidak membaca `.env` produksi. Tetapi seed 42, tiga epoch, dan rasio
validasi 0,2 terkonfirmasi dari catatan pelatihan. Suhu hasil fit juga
konsisten dengan ambang 50.

---

## 6. Panjang masukan

| Jalur | Batas token | Sumber di kode |
|---|---|---|
| Sentimen, inferensi | **512** | `SentimentService.MAX_MODEL_TOKENS`. Teks yang mencapai batas ditandai `truncated: true`. |
| Aspek, inferensi | **256** | `settings.aspect_inference_max_length`. Komentar di kode: 0,3% komentar produksi melebihi 128 token. |
| Aspek, pelatihan dan label BIO | **128** | `_prepare_bio_dataset` dan `_generate_bio_tags` (`max_length = 128`) |
| Aspek, skor ketidakpastian | 128 | `get_uncertainty_scores` |
| Sentimen, pelatihan | dinamis, ≤ 512 | `_encode` |

**Satu teks terpotong pada korpus studi kasus:**

- Analisis **Presiden**, ditandai oleh **modul sentimen** (`predictions[].truncated`).
- Panjangnya 4.874 karakter, atau **1.043 token** dengan tokenizer `crypter70`.
  Teks ini melebihi 512, sehingga juga melebihi batas aspek.
- Hanya sentimen yang mencatat pemotongan. Modul aspek memotong tanpa menandai.

**Catatan untuk draf:** tabel distribusi panjang di Bab IV menulis maksimum
**708 subkata**, padahal teks ini 1.043 token pada teks mentah. Kemungkinan
angka 708 dihitung pada teks yang sudah dibersihkan, atau dengan alat hitung
lain. Periksa dasar hitungannya.

**Keterbatasan data:** nilai `aspect_inference_max_length` pada 5 September
2026 (saat studi kasus berjalan) **TIDAK TERSEDIA**. Repo tidak punya riwayat
commit sejak commit awal, jadi tanggal perubahan dari 128 ke 256 tidak bisa
dipastikan. Nilai saat ini 256.

---

## 7. Pengujian otomatis layanan analisis

**Waktu jalan:** 27 September 2026, 19:48 WIB. Perintah:
`python -m pytest tests/ -q`.

**Hasil:** **483 lulus, 0 gagal, 0 error, 0 dilewati** (49,1 detik).

| No. | Sasaran (mengikuti tabel pengujian Bab IV) | Berkas uji | Kasus |
|---|---|---|---|
| 1 | Kontrak antarmuka | test_api_contract 41, test_health_identitas_bobot 6 | 47 |
| 2 | Penjajaran indeks dan urutan label | test_alignment_contract 11, test_label_order_safety 8 | 19 |
| 3 | Implementasi *preprocessing* | test_preprocessing 34, test_preprocessing_policy 11 | 45 |
| 4 | Kalkulasi metrik | test_topic_metrics 19, test_training_utils 37 | 56 |
| 5 | Skema BIO dan span | test_bio_tagging 17, test_gold_bio 6, test_aspect_merging 39, test_aspect_normalization 15 | 77 |
| 6 | Konkurensi pelatihan ulang | test_retrain_concurrency 26 | 26 |
| 7 | Jalur cadangan | test_rule_based_mode 15, test_model_loading_smoke 4 | 19 |
| 8 | Perilaku komponen analisis | test_sentiment_service 45, test_clause_sentiment 14, test_topic_service 36, test_topic_label_vocab 18, test_association_service 9, test_annotator 22, test_augmentation 6, test_provenance 7 | 157 |
| 9 | Keamanan akses layanan (baru) | test_security 16 | 16 |
| 10 | Penyimpanan dan pemulihan bobot (baru) | test_hub_revisions 11, test_baseline_path 2, test_lepas_memori 8 | 21 |
| | **Total (27 berkas)** | | **483** |

Dibanding tabel di draf (434 kasus layanan analisis):

- Kelompok 1 bertambah 6 kasus (`test_health_identitas_bobot`).
- Kelompok 6 bertambah dari 20 menjadi 26 kasus.
- Kelompok 9 dan 10 baru.

Untuk aplikasi web, sesi text-analysis-web mencatat **544 lulus** pada tanggal
yang sama. Rinciannya ada di sesi tersebut.

---

## 8. Penurunan modul aspek pada iterasi 8

Sudah diteliti. Penyebab yang didukung data:

**(a) Penurunannya nyata, bukan artefak label emas.**

| Acuan | Iterasi 6 (GD2) | Iterasi 7 | Iterasi 8 |
|---|---|---|---|
| Anotator 1 (label emas) | 76,62 | 76,45 | **73,21** |
| Anotator 1, tanpa baris yang diubah setelah tebakan (n = 234) | 77,99 | 77,48 | **74,51** |
| Irisan aspek yang disepakati kedua anotator | 59,63 | 60,02 | **56,73** (CI 7 → 8: −4,63 s.d. −1,94) |
| Anotator 2 | 57,14 | 57,09 | 56,44 (CI 7 → 8: −2,25 s.d. +1,06, tidak signifikan) |

Nilai adalah micro-F1 ketat. CI pada baris irisan dan anotator 2 dihitung
dengan 2.000 resampel dan `random.seed(42)`.

**(b) Bentuk penurunannya: iterasi 8 terlalu royal menandai aspek.**

- Dibanding iterasi 7 terhadap label emas, muncul **105 aspek baru (86 salah)**
  dan 25 aspek hilang (9 di antaranya benar).
- Presisi turun.
- Contoh aspek keliru baru: *hukum mati* (3), *kerja*, *orang tua*, *belanja*,
  *sekolah*, *sidak*, *ibu* (masing-masing 2).

**(c) Jumlah sampel aspek baru sangat kecil.**

- Iterasi 8 hanya menambah **6 sampel latih aspek** (kolam aspek 450 → 456,
  aspek 1.083 → 1.095), dari 36 koreksi pengguna setelah pemicu.
- Dua sampel meragukan secara definisi aspek:
  - `cucu` pada "Cucu krakatau tuh";
  - `adik` pada "…dinamain adik KRAKATAU".
- Kata kekerabatan seperti itu cocok dengan sebagian aspek keliru baru (*ibu*,
  *orang tua*, *istri*).

**(d) Penyebab utama yang didukung data: ketidakstabilan pelatihan ulang pada
kolam kecil.**

- 6 sampel tidak mungkin sendirian mengubah prediksi pada **76 dari 300
  baris**.
- Pelatihan #17 (iterasi 7) dan #19 (iterasi 8) **sama baiknya di validasi**:
  F1 token 0,8056 (epoch 2) lawan 0,8052 (epoch 3), dengan validasi 90 dan 91
  baris.
- Keduanya berbeda sekitar 3 poin di GD2.
- Setiap putaran dilatih ulang dari bobot dasar. Karena kolam sedikit berubah,
  pembagian latih/validasi berubah dan epoch terbaik berbeda.
- Set validasi yang kecil tidak dapat membedakan keduanya.
- Penjaga penerimaan hanya membandingkan dengan bobot dasar (delta +0,3356),
  bukan dengan iterasi sebelumnya.

**(e) Konsistensi koreksi aspek pengguna.**

- Secara umum wajar: 0,9% aspek pengguna tidak ditemukan persis di teksnya
  (fase studi kasus: 2,6%).
- Kelemahannya pada **cakupan**, bukan kesalahan: pengguna hanya mengubah
  13,9% daftar aspek dan menandai rata-rata 0,97 aspek per teks.
- Konsistensi antarpengguna **TIDAK TERSEDIA**, karena tidak ada teks yang
  dikoreksi lebih dari satu pengguna.

**Kesimpulan yang boleh ditulis:** penurunan aspek iterasi 8 nyata dan
signifikan. Penyebab utamanya adalah variasi pelatihan ulang pada data kecil
yang tidak tertangkap set validasi, dengan kontribusi kecil dari segelintir
koreksi aspek pengguna yang meragukan. Bias jangkar pada label emas
menaikkan angka absolut, tetapi bukan penyebab penurunan.

---

## Verifikasi tambahan (2 Oktober 2026)

Atas tiga temuan dari sesi penulisan. Sumbernya data mentah keenam analisis
lembaga di basis data aplikasi (`text_analyses.raw_data`) dan koreksi di
`training_items`.

### V1. Jumlah korpus: 7.968 atau 8.385?

**Yang benar-benar terjadi: 8.385.** Duplikat dan komentar berbahasa asing
**dihitung, tetapi tidak dihapus**. Yang dihapus sebelum unggah hanya dua teks
kosong (8.387 → 8.385).

| Lembaga | Baris yang dianalisis | Duplikat (trim + huruf kecil) | Duplikat persis |
|---|---|---|---|
| TNI | 3.548 | **297** | 274 |
| Presiden | 1.858 | **77** | 72 |
| Kejaksaan Agung | 1.493 | **10** | 6 |
| DPR | 778 | **6** | 4 |
| KPK | 368 | **15** | 14 |
| Polri | 340 | **0** | 0 |
| **Total** | **8.385** | **405** | 370 |

- Kolom "Duplikat" di tabel verifikasi (297/77/10/6/15/0 = 405) **cocok persis**
  dengan duplikat setelah *trim* dan huruf kecil. Pemeriksaannya memang
  dilakukan.
- Baris 8.385 pada analisis berarti ke-405 duplikat tetap ikut dianalisis.
- TNI 3.549 → 3.548 dan Presiden 1.859 → 1.858 sesuai dengan kolom "Teks
  Kosong" (1 dan 1).
- **Kolom "Bersih" (total 7.968) adalah hitungan seandainya duplikat dan
  bahasa asing dibuang. Itu tidak pernah diterapkan.**
- Deduplikasi baru terjadi saat pengambilan sampel Golden Dataset: 143
  duplikat ditolak di antara kandidat yang lolos panjang minimal.
- Jumlah komentar berbahasa asing (12) **tidak dapat diverifikasi ulang**,
  karena tidak ada detektor bahasa yang tersimpan. Angka itu tetap seperti
  catatan pemeriksaan awal.

**Saran penulisan:** ubah kolom "Bersih" menjadi "Tersisa bila dideduplikasi",
atau hapus kolomnya. Lalu tegaskan bahwa yang dibuang hanya 2 teks kosong,
sedangkan duplikat dipertahankan karena komentar berulang adalah bagian dari
percakapan nyata. Keterwakilannya dijaga pada evaluasi, karena populasi
Golden Dataset adalah komentar unik.

### V2. Batas panjang masukan dan maksimum 708

Diukur ulang dengan tokenizer `crypter70` pada 8.385 komentar. Hitungan
subkata tanpa token khusus `[CLS]`/`[SEP]`.

| Ukuran | Teks setelah pembersihan (dilihat model **sentimen**) | Teks mentah (dilihat model **aspek**) |
|---|---|---|
| Minimum | 0 | 1 |
| Median | **10** | 12 |
| Rerata | **15,67** | 18,52 |
| Persentil 95 | **46** | 52 |
| Persentil 99 | 94 | 113 |
| **Maksimum** | **708** | **1.041** (1.043 dengan token khusus) |
| Muat dalam 128 token | **99,30%** (59 melebihi) | 99,19% (68 melebihi) |
| Muat dalam 256 token | 99,95% (4 melebihi) | 99,90% (8 melebihi) |
| Muat dalam 512 token | 99,99% (1 melebihi) | 99,99% (1 melebihi) |

- **Tabel distribusi panjang di draf (median 10, rerata 15,7, persentil 95
  sebesar 46, maksimum 708, cakupan 99,30% pada 128) diukur pada teks yang
  sudah dibersihkan.** Angka-angkanya benar untuk teks bersih.
- Komentar terpanjang adalah komentar yang sama: 1.041 subkata saat mentah,
  708 setelah dibersihkan. Draf hanya perlu menyebut bahwa ukuran itu diambil
  setelah pembersihan.
- **Batas "128 subkata" di draf tidak sesuai dengan model yang berjalan:**

  | Modul | Batas | Teks | Melebihi |
  |---|---|---|---|
  | Sentimen, inferensi | 512 | teks bersih | 1 komentar (708 subkata) — inilah satu teks yang tercatat terpotong |
  | Aspek, inferensi | 256 | teks mentah (modul aspek tidak membersihkan teks karena berbasis posisi karakter) | 8 komentar |
  | Aspek, pelatihan dan label BIO | 128 | teks mentah | – |

- Angka 99,30% pada 128 karena itu tidak menggambarkan satu pun modul secara
  tepat.

**Saran penulisan:** sajikan batas per modul seperti tabel di atas. Pertahankan
statistik distribusi teks bersih untuk sentimen, dan tambahkan baris teks
mentah untuk aspek.

### V3. Jumlah koreksi pada analisis pita keyakinan: 330 atau 481?

**Benar: 330 = koreksi dua putaran pertama saja**, TNI 218 + Presiden 112.
Koreksi lengkap per putaran: TNI 218, Presiden 112, Kejaksaan Agung 62,
KPK 21, Polri 22, DPR 46 = **481**.

| Pita keyakinan | 330 koreksi (TNI + Presiden) | **481 koreksi (seluruh fase studi kasus)** |
|---|---|---|
| 0,00–0,50 | 56 data, 32 berubah (57,1%) | **77 data, 45 berubah (58,4%)** |
| 0,50–0,70 | 274 data, 132 berubah (48,2%) | **404 data, 192 berubah (47,5%)** |
| ≥ 0,70 | 0 | 0 |
| **Label sentimen berubah** | 164/330 = 49,7% | **237/481 = 49,3%** |
| **Daftar aspek berubah** | 193/330 = 58,5% | **288/481 = 59,9%** |
| Median keyakinan | 0,5785 | **0,5876** |

- Konsentrasi terhadap laju kesalahan dasar 17% (akurasi iterasi 0 pada GD1
  83,00%) tetap **2,9×**: 49,3% / 17%.
- Seluruh koreksi berada di bawah keyakinan 0,70, sesuai ambang ajakan
  koreksi.

**Saran penulisan:** pakai angka 481, karena mencakup seluruh fase studi
kasus. Kesimpulannya tidak berubah.

STATUS: SELESAI
