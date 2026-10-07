# Metodologi Modul Analisis Sentimen

Dokumen ini mencatat rancangan, pengukuran, dan keputusan pada modul analisis
sentimen. Semua angka dapat direproduksi dengan `scripts/evaluate_sentiment.py`;
seed tetap (`settings.retrain_seed = 42`).

---

## 1. Rumusan Masalah

Modul mengklasifikasikan teks berbahasa Indonesia ke dalam tiga kelas
(`positive` / `neutral` / `negative`) dan menjadi sumber angka utama yang
ditampilkan antarmuka: distribusi sentimen korpus. Ia juga dipakai ulang di
dalam `AspectService` untuk menilai polaritas per-aspek pada tingkat klausa.

Sebelum perbaikan ini, **modul sama sekali belum pernah diukur.** Modul aspek
punya `evaluate_aspect.py` dengan tolok ukur TermA/KEPS, modul topik punya
`eval_topics.py` beserta uji ketahanan dan varians seed. Sentimen tidak punya
padanannya — tidak ada dataset acuan, tidak ada angka, dan hampir tidak ada tes
otomatis. Akibat langsungnya dibahas di §5.

---

## 2. Pemilihan Model

### 2.1 Model yang dipakai

| | |
|---|---|
| Checkpoint | `crypter70/IndoBERT-Sentiment-Analysis` |
| **Model dasar** | **`indobenchmark/indobert-base-p1`** (IndoBERT, Wilie dkk. 2020) |
| Data *fine-tuning* | **IndoNLU SmSA / Prosa**, 11 000 dokumen |
| Peta label | `0 = positive`, `1 = neutral`, `2 = negative` |
| Suhu kalibrasi | 2,7748 (§7) |

Sitasi yang benar untuk skripsi adalah **IndoBERT (Wilie dkk., 2020)** dan
**IndoNLU SmSA**; checkpoint-nya hanya hasil *fine-tuning* keduanya.

### 2.2 Empat kandidat diukur, bukan dipilih dari nama

Model awal proyek (`mdhugol`) dipakai tanpa pernah dibandingkan. Empat kandidat
publik karena itu diukur lewat jalur produksi yang sama
(`scripts/bench_sentiment_models.py`), masing-masing dengan suhu kalibrasinya
sendiri:

| checkpoint | model dasar | IndoBERT? | test acc / macro F1 | ECE | perilaku |
|---|---|---|---|---|---|
| **`crypter70`** | **`indobenchmark/indobert-base-p1`** | ✅ | **0,9120 / 0,8855** | **0,0240** | **0,952** |
| `mdhugol` (awal) | `indobenchmark/indobert-base-p1` | ✅ | 0,9080 / 0,8762 | 0,0367 | 0,924 |
| `w11wo` | `flax-community/indonesian-roberta-base` | ❌ RoBERTa | 0,9160 / 0,8853 | 0,0209 | 0,972 |
| `ayameRushia` | `cahya/bert-base-indonesian-1.5G` | ❌ | 0,9180 / 0,9010 | — | 0,956 |

**Arsitektur diverifikasi dari `_name_or_path` pada config tiap checkpoint,
bukan dari namanya.** `w11wo` bernama "indonesian-roberta" dan memang RoBERTa;
tetapi `ayameRushia` yang bernama "bert-base-indonesian" ternyata memakai
`cahya/bert-base-indonesian-1.5G`, bukan IndoBERT. Hanya `crypter70` dan
`mdhugol` yang benar-benar berbasis IndoBERT.

**Dua aturan pemilihan, keduanya metodologis:**

1. **Kandidat harus IndoBERT**, karena itu yang diajukan pada seminar proposal.
   `w11wo` unggul pada uji perilaku (0,972) tetapi berpindah arsitektur, jadi
   tidak dipakai — ia tetap disimpan di registry sebagai cadangan.
2. **Pemilihan TIDAK boleh berdasarkan set uji.** Peringkatnya memang berbeda:
   di test `ayameRushia` menang, di valid `w11wo` menang, dan selisih antar
   kandidat (0,9020–0,9180) berada di bawah galat baku ±1,3 poin pada n=500.
   Yang sah dipakai memilih hanya **set validasi** dan **uji perilaku**.

Di antara dua kandidat IndoBERT, `crypter70` mengungguli `mdhugol` pada
**setiap** metrik — termasuk kemampuan yang paling menentukan pada data nyata:

| | `mdhugol` | `crypter70` |
|---|---|---|
| kontras "X buruk **tapi** Y bagus" | 0,417 | **0,917** |
| recall netral (valid) | 0,824 | **0,855** |
| ECE terkalibrasi (test) | 0,0367 | **0,0240** |

Komentar nyata padat konstruksi kontrastif, sehingga kegagalan 58% pada pola itu
bukan masalah teoretis — ia akan muncul pada evaluasi studi kasus.

### 2.3 Registry kandidat: model bisa ditukar tanpa mengubah kode

`SENTIMENT_MODELS` di `app/config.py` menyimpan keempat kandidat beserta model
dasar, **suhu kalibrasi**, dan **ambang peninjauan** masing-masing. Mengubah
`sentiment_base_model` otomatis menarik kalibrasi yang benar untuk model itu.

Ini perlu karena keduanya **milik bobot tertentu, bukan konstanta global**:
`mdhugol` memerlukan T = 1,3057 sementara `crypter70` memerlukan T = 2,7748.
Memakai suhu model lain membuat confidence salah kalibrasi, dan antrean koreksi
ikut salah sasaran — tanpa memunculkan galat apa pun. Ambangnya juga berbeda:
`crypter70` lebih percaya diri sehingga butuh 0,94 untuk menangkap porsi
kesalahan yang sama dengan 0,90 pada `mdhugol` (§8).

### 2.4 Urutan label dibaca dari model, tidak lagi diasumsikan

Urutan label **berbeda-beda antar checkpoint dan tidak mengikuti konvensi apa
pun**: `crypter70` dan `mdhugol` memakai 0=positive, tetapi
`taufiqdp/indonesian-sentiment` memakai **0=negatif** — persis kebalikannya.
Memakai peta yang salah membalik seluruh hasil **tanpa satu pun galat muncul**.

`_resolve_label_map()` karena itu membaca `id2label` dari config model dan hanya
jatuh ke peta bawaan bila config tidak informatif (`LABEL_0/1/2`, seperti pada
`mdhugol`). Jalur retraining memakai `label_to_id` yang diturunkan dari peta
yang sama — melatih dengan peta yang salah akan menukar label, dan lossnya tetap
turun tanpa tanda apa pun bahwa hasilnya kacau.

### 2.5 Konsekuensi metodologis yang harus disebut di skripsi

Seluruh kandidat di-*fine-tune* pada SmSA train, sehingga evaluasi pada SmSA
test adalah uji **in-domain** bagi semuanya. Angkanya batas atas, bukan bukti
generalisasi — persis pola yang sudah terdokumentasi pada modul aspek (F1 0,97
in-domain vs 0,029 lintas domain). Konsekuensi lain: **mengganti model tidak
menyelesaikan keterbatasan domain**, karena semuanya mewarisi keterbatasan data
latih yang sama.

---

## 3. Hasil Pengukuran

`python scripts/evaluate_sentiment.py` — SmSA test, 500 dokumen, jalur produksi
penuh (`SentimentService.analyze`, termasuk preprocessing).

### 3.1 Ringkasan

| Metrik | Nilai |
|---|---|
| Accuracy | **0,9120** |
| Macro F1 | **0,8855** |
| Weighted F1 | 0,9091 |
| ECE (terkalibrasi) | 0,0240 |

Sebagai pembanding, IndoNLU melaporkan F1 ~0,88 untuk IndoBERT-base pada SmSA,
jadi jalur produksi ini bekerja setara publikasi rujukannya — pembersihan teks
yang dilakukan modul tidak merusak model.

### 3.2 Per kelas

| Kelas | Precision | Recall | F1 | n |
|---|---|---|---|---|
| positive | 0,9147 | 0,9279 | 0,9212 | 208 |
| neutral | 0,8857 | **0,7045** | 0,7848 | 88 |
| negative | 0,9178 | 0,9853 | 0,9504 | 204 |

### 3.3 Confusion matrix

| gold \ prediksi | positive | neutral | negative |
|---|---|---|---|
| **positive** | 193 | 8 | 7 |
| **neutral** | **15** | 62 | 11 |
| **negative** | 3 | 0 | 201 |

---

## 4. Keterbatasan Utama: Recall Kelas Netral

Kelas `neutral` adalah satu-satunya kelemahan nyata: **recall 0,7045**, dengan
15 dari 88 dokumen netral dinilai positif dan 11 dinilai negatif. Penyebabnya
struktural — pada SmSA train kelas netral hanya 1 148 dari 11 000 dokumen
(10,4%).

**Mengganti model tidak menyelesaikannya.** Keempat kandidat mewarisi data latih
yang sama, dan recall netralnya bergerak dalam pita sempit: 0,5795–0,7727 pada
test. `crypter70` termasuk yang terbaik, tetapi tetap jauh di bawah dua kelas
lainnya. Ini menegaskan penyebabnya ada di **data**, bukan arsitektur.

**Ini juga bukan masalah ambang, dan tidak bisa ditambal dengan kalibrasi.**
Diukur langsung pada model awal (`mdhugol`): bobot per-kelas dicari pada SmSA
**valid** (1 260 dokumen) lalu diterapkan pada test. Kesimpulannya berlaku untuk
semua kandidat karena sebabnya sama.

| bobot kelas netral | macro F1 (valid) |
|---|---|
| **1,00 (tanpa penyesuaian)** | **0,9119** |
| 1,25 | 0,9093 |
| 1,50 | 0,9079 |
| 2,00 | 0,9049 |
| 3,00 | 0,9031 |
| 5,00 | 0,8999 |

Optimum jatuh tepat di "tanpa penyesuaian", dan **setiap** penguatan kelas
netral justru menurunkan macro F1. Alasannya terlihat pada analisis kesalahan:
model salah dengan **sangat yakin** — "perempuan yang malang itu sekarang sudah
tutup usia" dinilai negatif pada 0,9952, "di indonesia lagi banyak bencana"
negatif pada 0,9897. Tidak ada aturan keputusan yang memisahkannya, karena
kesalahan itu tidak berada di dekat batas keputusan.

Sebagian di antaranya bahkan **debatable secara anotasi**: kalimat bermuatan
peristiwa buruk yang diberi label netral oleh anotator SmSA.

Kesimpulan yang sama dengan keterbatasan pemetaan aspek: **memperbaikinya butuh
data berlabel, bukan penyetelan parameter.** Jalurnya adalah loop active
learning yang sudah ada (§8).

---

## 5. Cacat yang Ditemukan dan Diperbaiki

Semua ditemukan justru karena harness pengukuran §3 akhirnya ada.

### 5.1 Teks kosong dinilai `positive` dengan keyakinan 0,9429

Kontrak penjajaran mengharuskan N teks masuk = N hasil keluar, sehingga baris
kosong **tidak boleh dibuang**. Tetapi baris itu dulu ikut diteruskan ke
IndoBERT, yang mengklasifikasi urutan `[CLS] [SEP]` dan mengembalikan
`positive` pada 0,9429 — konsisten, dan sepenuhnya tanpa makna.

Dampaknya bukan kosmetik: setiap baris kosong pada CSV yang diunggah pengguna
**menaikkan persentase positif**, yaitu angka utama yang ditampilkan Laravel.
Terukur pada contoh 8 baris dengan 3 baris tanpa isi: distribusi berubah dari
`62,5% positif` menjadi `40% positif / 20% netral / 40% negatif`.

Pemeriksaan dilakukan **setelah** preprocessing, karena pembersihan bisa
*menghasilkan* teks kosong dari masukan yang tidak kosong (`"a"`, `"😀"`).
Baris seperti itu kini kembali sebagai netral berkeyakinan 0 bertanda
`method: 'empty'`, dan dikeluarkan dari penyebut distribusi — sejajar dengan
`[]` pada modul aspek dan `-1` pada modul topik.

### 5.2 Tidak ada penanda jalur yang dipakai

Jalur cadangan menandai hasilnya `method: 'rule-based'`, tetapi jalur model
tidak menandai apa pun. Pemanggil karena itu tidak bisa membedakan prediksi
IndoBERT dari tebakan daftar kata. Kelas kegagalan diam-diam yang sama pernah
terjadi di modul topik, di mana hanya kolom `method` yang mengungkap bahwa
BERTopic tidak pernah berjalan. Setiap prediksi kini menyebut jalurnya:
`indobert` / `rule-based` / `empty` / `error`.

### 5.3 `text` berisi hasil pembersihan, bukan teks asli

`"wkwkwk anjay bener bang 😂😂"` kembali sebagai `"tertawa benar bang"`. Laravel
menambalnya sendiri, tetapi API yang berdiri sendiri menjadi menyesatkan. Kini
`text` berisi teks asli dan `processed_text` berisi hasil pembersihan;
`ProcessTextAnalysis::attachOriginalTexts` disesuaikan agar tidak menimpa nilai
yang sudah dikirim API.

### 5.4 Distribusi jalur batch berbeda dari jalur tunggal

Setelah §5.1, jalur batch Laravel (dataset > `batch_size`) masih menghitung
baris kosong, sehingga dataset besar dan kecil melaporkan persentase berbeda
pada data yang sama. `NLPApiService::scorablePredictions()` kini menyaringnya,
sama seperti sisi Python.

### 5.5 Pemotongan teks panjang tidak terlihat sama sekali

Teks melebihi 512 token dipotong diam-diam, dan bagian setelahnya tidak pernah
ikut dinilai. Tidak satu pun korpus proyek ini terdampak — panjang maksimum
terukur 253 token (YouTube), 132 (SmSA train), 106 (ulasan) — tetapi API
menerima teks sampai 10 000 karakter, sehingga dokumen panjang yang diunggah
pengguna bisa kehilangan ekornya. Pada ulasan, justru bagian akhir yang lazim
memuat kesimpulan penilaian.

Prediksi yang mengisi seluruh jendela model kini ditandai `truncated: true`,
dihitung pada `metrics.total_truncated`, dan dicatat ke log. Yang diperbaiki
adalah **kesenyapannya**, bukan pemotongannya: memecah dokumen menjadi
potongan lalu menggabungkan hasilnya akan menambah kerumitan yang tidak bisa
diuji terhadap korpus mana pun yang tersedia di proyek ini.

### 5.6 Satu chunk gagal menjatuhkan 256 teks sekaligus

Inferensi dijalankan per chunk 256 teks. Bila satu chunk gagal — lazimnya
kehabisan memori pada kontainer Railway — **seluruh** 256 teks di dalamnya
dulu dikembalikan sebagai netral berkeyakinan 0, padahal yang bermasalah
umumnya hanya satu teks.

`_score_chunk()` kini membelah chunk yang gagal menjadi dua dan mencoba ulang,
sampai ukuran satu. Hanya teks yang benar-benar gagal yang ditandai
`method: 'error'`; sisanya tetap memperoleh prediksi sungguhan. Diverifikasi
dengan menyuntikkan kegagalan: dari 3 teks pada chunk yang selalu gagal saat
ukuran > 1, ketiganya tetap memperoleh prediksi IndoBERT
(`total_failed = 0`). Jumlah kegagalan dilaporkan pada `metrics.total_failed`.

---

## 6. Kebijakan Preprocessing, Diukur

`_sanitize_config()` **memaksa** `stemming`, `remove_stopwords`, dan
`lemmatization` mati untuk jalur BERT, apa pun yang dikirim pemanggil.

Ablasi pada SmSA test (`python scripts/evaluate_sentiment.py --ablation`):

| konfigurasi | `crypter70` acc / macro F1 | `mdhugol` acc / macro F1 |
|---|---|---|
| tanpa preprocessing | 0,9060 / 0,8716 | **0,9180** / 0,8889 |
| huruf kecil + buang tanda baca | **0,9140** / 0,8872 | 0,9040 / 0,8701 |
| + normalisasi slang/typo | 0,9120 / 0,8855 | 0,9080 / 0,8762 |
| seperti Laravel (minta stemming + stopword) | 0,9120 / 0,8855 | 0,9080 / 0,8762 |

Tiga hal terbaca, dan yang ketiga adalah temuan yang jujur harus dilaporkan:

1. **Pengaman `_sanitize_config` benar-benar bekerja.** Baris terakhir meminta
   stemming dan stopword removal, dan hasilnya **identik sampai empat desimal**
   dengan baris yang tidak memintanya — pada kedua model. Itu hanya mungkin bila
   keduanya memang dimatikan.
2. **Alasan mematikannya adalah kebenaran, bukan penyetelan.** 5 dari 6 kalimat
   bernegasi berbalik `negative → positive` pada ~0,99 ketika stemming aktif,
   karena "pelayanannya tidak bagus sama sekali" menjadi "layan bagus". Ini
   kerusakan sistematis, bukan selisih beberapa dokumen.
3. **Arah efek pembersihan lain BERBEDA antar model, dan besarnya di bawah
   derau.** Pada `mdhugol` pembersihan merugikan (−1,0 poin); pada `crypter70`
   justru menguntungkan (+0,8 poin). Normalisasi slang menaikkan `mdhugol`
   (+0,4) tetapi sedikit menurunkan `crypter70` (−0,2). Seluruh selisih ini
   setara 1–5 dokumen dari 500, yaitu **di dalam galat baku ±1,3 poin**.
   Kesimpulan yang bisa dipertahankan: **di luar perlindungan negasi, pilihan
   pembersihan bersifat sekunder** dan tidak layak disetel pada satu korpus.

**Mengapa kebijakan tidak disetel pada SmSA.** Selain selisihnya berada di
dalam derau, **SmSA bukan domain produksi**: normalisasi menyentuh **5,0%**
token SmSA tetapi **11,2%** token korpus YouTube — 2,2 kali lebih banyak.
Menyetel kebijakan pada korpus yang efeknya separuh domain sasaran adalah
kesalahan yang persis sudah terdokumentasi di modul topik, di mana normalisasi
*menurunkan* c_v tetapi menghapus *register cluster*. Normalisasi karena itu
dipertahankan atas dasar kesesuaian domain, bukan atas dasar angka SmSA.

---

## 7. Kalibrasi Confidence (Temperature Scaling)

### 7.1 Model ini memang terlalu yakin

Klasifikasi berbasis transformer terkenal *overconfident* (Guo, Pleiss, Sun &
Weinberger, 2017). Diukur pada modul ini sebelum kalibrasi:

| set | ECE | akurasi | rata-rata confidence | selisih |
|---|---|---|---|---|
| SmSA valid | 0,0624 | 0,9357 | 0,9958 | **+0,0601** |
| SmSA test | 0,0827 | 0,9120 | 0,9926 | **+0,0806** |

Model ini bahkan **lebih** *overconfident* daripada model awal: rata-rata
keyakinannya 0,9926 padahal benar 0,9120. Itu bukan alasan menolaknya —
justru contoh mengapa kalibrasi diperlukan, dan setelah dikalibrasi ia menjadi
**lebih** jujur daripada model awal (0,0240 vs 0,0367).

### 7.2 Suhu dicari di validasi, dilaporkan di uji

`fit_temperature()` mencari T yang meminimalkan NLL dengan LBFGS pada SmSA
**valid** (1 260 dokumen), lalu hasilnya dilaporkan pada SmSA **test** yang
tidak ikut mencarinya. **T = 2,7748.**

| set | T | ECE | accuracy |
|---|---|---|---|
| valid | 1,0000 | 0,0624 | 0,9357 |
| valid | **2,7748** | **0,0142** | 0,9357 |
| test | 1,0000 | 0,0827 | 0,9120 |
| test | **2,7748** | **0,0240** | 0,9120 |

**ECE turun 77% (valid) dan 71% (test), dan akurasi sama persis.**

Suhu tiap kandidat berbeda jauh — 1,3057 untuk `mdhugol`, 1,6132 untuk `w11wo`,
2,7748 untuk `crypter70` — yang menegaskan suhu adalah **milik bobot tertentu**
dan harus ikut berpindah saat model diganti (§2.3). Itu bukan
kebetulan melainkan sifat metodenya: membagi logit dengan skalar positif tidak
mengubah urutannya, sehingga argmax — dan karenanya accuracy, F1, dan seluruh
confusion matrix — dijamin identik. Hanya keyakinannya yang menjadi jujur.
Inilah yang membuat temperature scaling aman dipakai sebagai lapisan akhir.

Reproduksi: `python scripts/evaluate_sentiment.py --fit-temperature`.

### 7.3 Kalibrasi tidak selalu terdefinisi — dan itu ditangani

Ini ditemukan saat menguji jalur retraining, bukan diperkirakan sebelumnya.
Dengan set validasi 19 sampel yang **semuanya benar**, optimasi menghasilkan
**T = 0,308** — bukan kalibrasi melainkan kebalikannya: model dibuat maksimal
*overconfident*. Penyebabnya struktural, bukan bug numerik: tanpa satu pun
kesalahan, NLL terus turun selama keyakinan dinaikkan, sehingga T meluncur ke
nol. Suhu tidak teridentifikasi di sana.

Kalau lolos, akibatnya justru pada bagian yang paling bergantung padanya:
`review_queue` memilih baris berdasarkan confidence, dan confidence yang
sengaja dinaikkan membuat antrean koreksi kosong.

`fit_temperature()` karena itu menolak mengalibrasi dan mengembalikan 1,0
("tanpa kalibrasi") bila:

- sampel validasi < 50 (Guo dkk. memakai ribuan),
- tidak ada kesalahan sama sekali pada set validasi,
- tidak ada prediksi benar sama sekali, atau
- T di luar rentang wajar [0,5 ; 5,0].

Verifikasi: fit sesungguhnya pada 1 260 sampel tetap menghasilkan suhu yang
wajar untuk setiap kandidat (2,7748 / 1,6132 / 1,3057) — pengaman ini hanya
memblokir kasus patologis, bukan kasus nyata.

---

## 8. Confidence sebagai Penggerak Active Learning

Setelah kalibrasi, akurasi per pita confidence pada SmSA test:

| confidence | porsi dokumen | akurasi |
|---|---|---|
| 0,90 – 1,00 | 91,0% | **0,9495** |
| 0,70 – 0,90 | 5,8% | 0,5172 |
| 0,50 – 0,70 | 2,8% | 0,5714 |
| < 0,50 | 0,4% | 0,5000 |

Pemisahannya tajam: di atas 0,90 model benar 95%, di bawahnya hanya sekitar
separuh — persis dasar yang dibutuhkan untuk memilih baris yang perlu ditinjau
manusia.

**Ambangnya dipilih dari pengukuran, bukan angka bulat, dan berbeda tiap
model.** Tabel berikut menjawab pertanyaan yang sebenarnya: kalau meninjau
semua baris di bawah ambang X, berapa persen kesalahan yang tertangkap?

| ambang | ditinjau | porsi korpus | kesalahan tertangkap | recall kesalahan | lift |
|---|---|---|---|---|---|
| 0,70 | 16 | 3,2% | 7 | 15,9% | 4,97 |
| 0,80 | 26 | 5,2% | 13 | 29,5% | 5,68 |
| 0,90 | 45 | 9,0% | 21 | 47,7% | 5,30 |
| 0,92 | 55 | 11,0% | 23 | 52,3% | 4,75 |
| **0,94** | **86** | **17,2%** | **28** | **63,6%** | **3,70** |
| 0,95 | 137 | 27,4% | 35 | 79,5% | 2,90 |
| 0,96 | 377 | 75,4% | 43 | 97,7% | 1,30 |

`lift` = recall kesalahan dibagi porsi korpus yang ditinjau; 1,0 berarti sama
saja dengan meninjau acak.

**Ambang 0,94 dipakai: meninjau 17% baris menangkap 64% kesalahan — 3,7 kali
lebih efisien daripada meninjau acak.** Menaikkannya ke 0,95 memang menambah
16 persen poin recall, tetapi beban tinjaunya melonjak dari 17% ke 27%; 0,96
praktis berarti meninjau tiga perempat korpus.

**Ambang ini TIDAK boleh disalin antar model.** Pada `mdhugol`, titik kerja yang
setara justru ada di 0,90 (18,0% ditinjau → 67,4% kesalahan). Memakai 0,90 pada
`crypter70` hanya menangkap 47,7% kesalahan, karena model ini lebih percaya
diri sehingga lebih sedikit baris yang jatuh di bawahnya. Itulah sebabnya ambang
disimpan per model di `SENTIMENT_MODELS` (§2.3).

`SentimentService._build_review_queue()` mengembalikan blok `review_queue`
(`threshold`, `count`, `share`, `indices` terurut dari yang paling tidak yakin)
pada setiap respons. Indeksnya sejajar dengan daftar `texts` pemanggil, sehingga
Laravel dapat langsung mengurutkan antrean koreksi. Baris `empty` sengaja
dikecualikan — tidak ada yang bisa dikoreksi darinya.

Inilah penghubung eksplisit antara evaluasi dan retraining: koreksi yang masuk
lewat antarmuka mengisi `/api/retrain/sentiment`.

---

## 9. Retraining

Aturan bersama dengan modul aspek (seed tetap, split stratifikasi, bobot kelas,
ukur-sebelum-dan-sesudah, tolak checkpoint yang menurunkan metrik) ada di
`app/utils/training.py`. Dua perbaikan khusus jalur sentimen:

### 9.1 Panjang padding diturunkan dari data

Sebelumnya setiap sampel di-pad ke **512 token** tanpa memandang isinya. Data
koreksi berisi komentar pendek, dan self-attention berskala kuadratik terhadap
panjang urutan, sehingga sebagian besar komputasi terbuang pada token semu.

Diukur pada 92 sampel koreksi, CPU, 1 epoch penuh:

| panjang padding | waktu 1 epoch |
|---|---|
| 512 (lama) | **296,7 detik** |
| 88 (diturunkan dari data) | **43,6 detik** |

**6,8x lebih cepat.** Pada Railway yang tanpa GPU, ini selisih antara retraining
yang selesai dan yang kehabisan waktu. Batas atas tetap 512 dan dibulatkan ke
kelipatan 8.

### 9.2 Epoch terbaik, bukan epoch terakhir

Sebelumnya bobot **epoch terakhir** yang dipakai. Pada data koreksi yang kecil
dan timpang, weighted F1 validasi lazim memuncak di epoch 1–2 lalu turun karena
overfitting — menjalankan 3 epoch berarti membuang hasil terbaik yang sudah
dicapai.

Kini setiap epoch dievaluasi, yang terbaik disimpan, dan bobot dikembalikan ke
sana sebelum dibandingkan dengan model lama. Perbandingan penerimaan checkpoint
karena itu selalu memakai kandidat **terbaik**, bukan kandidat kebetulan-terakhir.
Respons memuat `best_epoch` dan `epoch_history` agar overfitting terlihat.

Penyimpanan dilakukan ke **disk**, bukan salinan `state_dict` di RAM: menyalin
bobot IndoBERT menambah ~500 MB, dan seluruh service dirancang muat di kuota RAM
Railway (alasan yang sama membuat rollback dilakukan dengan memuat ulang dari
disk). Direktori sementara dihapus setelah selesai.

### 9.3 Kalibrasi ikut dicari ulang

Suhu adalah milik **bobot tertentu**. Setelah fine-tuning distribusi logit
berubah, sehingga suhu model dasar tidak lagi berlaku — memakainya membuat
`review_queue` memilih baris yang salah. Suhu karena itu dicari ulang pada set
validasi retraining dan disimpan bersama checkpoint sebagai `calibration.json`,
lalu dimuat otomatis saat service dimulai ulang.

Bila set validasi terlalu kecil atau modelnya benar semua, `fit_temperature`
mengembalikan 1,0 alih-alih angka patologis — lihat §7.3, kasus itu ditemukan
justru pada jalur ini.

Verifikasi jalan penuh (92 sampel, rasio ketimpangan 20:1): padding 88 token,
bobot kelas `{positive: 4,06, neutral: 8,11, negative: 0,38}`, epoch terbaik 1
dari 3, checkpoint diterima, `calibration.json` tersimpan dengan T = 1,0
(19 sampel validasi < 50, kalibrasi dilewati sebagaimana mestinya), direktori
sementara bersih.

---

## 10. Jalur Cadangan (Daftar Kata)

Dipakai **hanya** bila bobot model gagal dimuat sama sekali. Mutunya kini ikut
terukur (`python scripts/evaluate_sentiment.py --rule-based`), bukan diasumsikan:

| | accuracy | macro F1 |
|---|---|---|
| IndoBERT | 0,9180 | 0,8889 |
| daftar kata — **sebelum** | 0,6120 | 0,5974 |
| daftar kata — **sesudah** | **0,6720** | **0,6538** |

Perbaikannya (+6,0 poin akurasi) berasal dari satu koreksi konseptual:
**negasi dikeluarkan dari daftar kata negatif.** Versi lama memasukkan `tidak`
dan `kurang` sebagai kata negatif, sehingga "tidak jelek" menghasilkan dua
hitungan negatif dan "tidak bagus" berakhir seri lalu dinilai netral. Kini
negator membalik polaritas maksimal dua kata sesudahnya — jangkauan yang cukup
untuk pola bahasa Indonesia ("tidak begitu bagus") tanpa menyeret klausa
berikutnya.

Selisihnya dengan IndoBERT (~25 poin) adalah **biaya degradasi yang kini
diketahui angkanya**, dan itulah gunanya mencatatnya.

---

## 11. Evaluasi Lintas Domain: Tolok Ukur yang Terkontaminasi

Untuk menguji generalisasi, kandidat paling jelas adalah **NusaX-senti**
(Indonesia, 3 kelas, berlabel manusia). Sebelum dipakai, tumpang tindihnya
dengan data latih model diperiksa — dan hasilnya menggugurkan pemakaiannya:

| | jumlah | terhadap SmSA (train+valid+test) |
|---|---|---|
| NusaX test | 400 | **111 sama persis**, **328 (82%) berbagi 8-gram** |
| NusaX valid | 100 | 28 sama persis, 79 berbagi 8-gram |

NusaX Indonesia berbagi bahan sumber dengan SmSA. Memakainya sebagai tolok ukur
"lintas domain" akan **melaporkan hafalan sebagai generalisasi**.

Setelah penyaringan ketat (buang yang sama persis maupun yang berbagi 6-gram),
hanya **57 dokumen** yang tersisa bersih. Pada subset itu:

| Metrik | Nilai |
|---|---|
| Accuracy | 0,9474 |
| Macro F1 | 0,9361 |
| Recall netral | 0,8824 |

Mutu **tidak runtuh** di luar SmSA — indikasi baik. Tetapi n = 57 terlalu kecil
untuk klaim kuat (selang kepercayaan ±6 poin), dan teksnya masih berupa ulasan,
bukan komentar media sosial seperti domain produksi. **Laporkan sebagai
pemeriksaan kewajaran, bukan sebagai bukti generalisasi.**

Temuan metodologisnya sendiri layak ditulis: *tolok ukur sentimen bahasa
Indonesia yang tersedia umum berbagi bahan sumber, sehingga evaluasi lintas
domain yang sahih menuntut pemeriksaan kontaminasi lebih dulu.*

---

## 12. Uji Perilaku (CheckList)

Akurasi agregat menyembunyikan kegagalan sistematis: model bisa mencetak 0,91
sambil selalu salah pada kalimat bernegasi. Ribeiro, Wu, Guestrin & Singh (2020)
menunjukkan kelemahan seperti itu baru terlihat bila kemampuannya diuji satu per
satu.

**Nilainya khusus untuk skripsi ini: uji ini tidak memerlukan data berlabel.**
Labelnya diketahui dari cara kalimatnya dibangun, sehingga generalitas modul bisa
diukur sekarang — sementara korpus produksi belum punya anotasi manusia.

`scripts/behavioral_sentiment.py` membangun 249 kasus dari templat:

| jenis | arti |
|---|---|
| **MFT** | kalimat dari templat, labelnya pasti dari konstruksinya |
| **INV** | perubahan permukaan yang TIDAK boleh mengubah label |
| **DIR** | perubahan yang harus menggeser prediksi ke arah tertentu |

### 12.1 Hasil per kemampuan

| kemampuan | `crypter70` | `mdhugol` | `w11wo` |
|---|---|---|---|
| MFT dasar positif / negatif | 1,000 | 1,000 | 1,000 |
| MFT faktual netral | 1,000 | 1,000 | 1,000 |
| MFT negasi atas kata positif | 1,000 | 1,000 | 1,000 |
| **MFT negasi atas kata negatif** | **0,542** | 0,500 | **0,917** |
| MFT kontras → negatif | 1,000 | 1,000 | 0,917 |
| **MFT kontras → positif** | **0,917** | **0,417** | 0,750 |
| INV ejaan slang / emoji / huruf berulang / kapital / tanda baca | 1,000 | 1,000 | 1,000 |
| DIR tambah klausa negatif | 1,000 | 1,000 | 1,000 |
| **TOTAL** | **0,952** | 0,924 | 0,972 |

### 12.2 Yang terbaca dari tabel ini

**Ketahanan terhadap ragam bahasa media sosial sempurna (INV 1,000 pada semua
perturbasi).** Ejaan slang, emoji, huruf berulang, huruf kapital, dan tanda baca
berlebih tidak mengubah satu pun label. Ini penting karena justru itu ciri
domain produksi, dan sebagian besar berkat normalisasi di `TextCleaner`.

**Kemampuan menangani kontras adalah alasan utama model diganti.** Model awal
gagal pada 58% kalimat berpola "X buruk **tapi** Y bagus" — dan komentar nyata
padat konstruksi itu. `crypter70` memperbaikinya ke 0,917.

**Negasi atas kata negatif tetap lemah (0,542), dan sebagian ini keterbatasan
ujinya sendiri.** Kegagalannya seluruhnya pada pola "kurang buruk" / "kurang
mengecewakan", sementara pola "tidak buruk" lulus 100%. Frasa "kurang + kata
negatif" memang janggal dalam bahasa Indonesia sehari-hari, jadi sebagian
kegagalan ini menghukum konstruksi yang jarang dipakai penutur asli. Yang
penting: **pola negasi yang lazim ("tidak bagus", "tidak buruk") ditangani
dengan benar.**

### 12.3 Batas dari metode ini

Kalimat templat **lebih mudah** daripada komentar nyata: pendek, satu klausa,
tanpa sarkasme atau konteks. Skor 0,952 karena itu **bukan** perkiraan akurasi
di lapangan — ia menguji apakah kemampuan dasarnya utuh, bukan seberapa baik
modul bekerja pada data sungguhan. Untuk yang terakhir tetap dibutuhkan
evaluasi studi kasus dengan anotasi manusia.

---

## 13. Keterbatasan yang Harus Ditulis di Skripsi

1. **Evaluasi in-domain.** Seluruh kandidat di-*fine-tune* pada SmSA train,
   sehingga SmSA test adalah data uji milik mereka sendiri. Angka 0,9120 adalah
   batas atas, bukan bukti generalisasi.
2. **Tidak ada data berlabel pada domain produksi.** Korpus YouTube
   pajak/korupsi tidak punya label sentimen manusia. Inilah alasan loop active
   learning ada, dan alasan `review_queue` dibuat. Evaluasi studi kasus akan
   menutup celah ini.
3. **Recall netral 0,7045**, dan terbukti tidak bisa diperbaiki dengan kalibrasi
   (§4) maupun dengan mengganti model (§2.2) — sebabnya ada di data latih SmSA
   yang hanya 10,4% netral.
4. **Tidak ada tolok ukur lintas domain yang bersih** (§11); NusaX terkontaminasi
   82% terhadap SmSA, dan sisa bersihnya hanya 57 dokumen.
5. **Sentimen tingkat dokumen**, bukan tingkat entitas. Kalimat yang memuji satu
   hal dan mengecam hal lain memperoleh satu label. Modul aspek menutup sebagian
   celah ini dengan menilai polaritas per klausa.
6. **Bias kontras berkurang tetapi tidak hilang** (0,917, sebelumnya 0,417), dan
   negasi atas kata negatif masih 0,542 (§12).
7. **Pemilihan model dibatasi checkpoint publik.** Empat kandidat yang diukur
   semuanya dilatih pada SmSA, sehingga perbandingannya adil tetapi tidak
   mencakup kemungkinan model yang dilatih pada data lain.

**Catatan penting untuk evaluasi studi kasus.** Untuk MENGUKUR akurasi, ambil
sampel **acak** dari korpus — jangan memakai `review_queue`. Antrean itu sengaja
bias ke baris yang model tidak yakin, sehingga akurasi yang dihitung darinya akan
terlalu rendah. `review_queue` untuk **mengumpulkan data koreksi**, sampel acak
untuk **mengukur**. Keduanya mudah tertukar dan kesalahannya sulit terlihat.

Bila hasil studi kasus menunjukkan kandidat lain lebih cocok, penggantiannya
cukup mengubah `sentiment_base_model` — suhu kalibrasi dan ambang peninjauannya
ikut otomatis (§2.3), dan seluruh kandidat sudah diukur.

## 14. Reproduksi

```powershell
# Evaluasi utama (SmSA test, 500 dokumen) - accuracy, macro F1, ECE,
# tabel hasil antrean tinjau, dan kalibrasi per pita
python scripts/evaluate_sentiment.py

# Cari suhu kalibrasi di set validasi, laporkan efeknya di set uji
python scripts/evaluate_sentiment.py --fit-temperature

# Ablasi kebijakan preprocessing
python scripts/evaluate_sentiment.py --ablation

# Mutu jalur cadangan
python scripts/evaluate_sentiment.py --rule-based --out data/experiments/sentiment_eval_rulebased.json

# Subset lintas domain yang sudah dibersihkan dari kontaminasi
python scripts/evaluate_sentiment.py --gold data/external/nusax/clean_gold.json

# Bandingkan kandidat model (in-domain + kesepakatan di domain produksi)
python scripts/bench_sentiment_models.py

# Uji perilaku 249 kasus - tidak butuh data berlabel
python scripts/behavioral_sentiment.py --show-failures

# Ukur kandidat lain tanpa mengubah setting produksi
python scripts/evaluate_sentiment.py --model w11wo/indonesian-roberta-base-sentiment-classifier --fit-temperature
python scripts/behavioral_sentiment.py --model w11wo/indonesian-roberta-base-sentiment-classifier

# Tes unit (tanpa memuat bobot model)
python -m pytest tests/test_sentiment_service.py -q
```

Keluaran tersimpan di `data/experiments/sentiment_*.json`; kesalahan
berkeyakinan tertinggi di `data/experiments/sentiment_errors.json` sebagai bahan
analisis kualitatif.
