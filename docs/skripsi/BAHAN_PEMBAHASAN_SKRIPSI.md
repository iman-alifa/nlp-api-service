# Bahan Penyusunan Bab IV–V: Temuan, Pemenuhan Tujuan, dan Perbaikan Draf

Disiapkan 27 September 2026 dari pemeriksaan draf
`Templat Skripsi Prodi KS MS Word Iman (1).md` dan dari basis data aplikasi
(lokal untuk fase skripsi, server produksi untuk fase pengguna asli).
Semua angka di bawah diambil langsung dari sistem, bukan dari draf.

Dokumen ini dipakai sebagai bahan diskusi penyusunan naskah. Isinya:

1. Peta pemenuhan tujuan penelitian (ringkas, untuk kerangka Bab V)
2. Capaian terhadap *quality gate* Bab III (jujur, per kriteria)
3. Temuan per tujuan beserta datanya
4. Yang perlu ditambahkan ke draf
5. Yang perlu diperbaiki di draf (kesalahan faktual dan inkonsistensi)
6. Yang sebaiknya dipangkas
7. Keterbatasan yang harus dinyatakan
8. Data yang masih harus dilengkapi
9. Usulan kerangka kesimpulan

Prinsip yang dipegang: pembahasan berfokus pada **temuan** dan **pemenuhan
tujuan**. Cerita perbaikan teknis selama pengembangan tidak dibahas sebagai
subbab tersendiri.

---

## 1. Peta pemenuhan tujuan

| Tujuan (Bab I §1.3) | Status | Bukti utama |
|---|---|---|
| T1. Model adaptif IndoBERT (sentimen, aspek) + BERTopic untuk komentar media sosial | **Tercapai sebagian**. Model berjalan dan beradaptasi; *quality gate* macro-F1 ≥ 0,80 belum tercapai pada titik akhir. | Sentimen 85,67% / macro-F1 78,39 (fase 1, iterasi 6); aspek micro-F1 parsial 80,06; c_v 0,4998 |
| T2. Aplikasi web *end-to-end* | **Tercapai** | Alur lengkap berjalan; 8.385 komentar terproses tanpa kegagalan; 10/10 skenario UAT; aplikasi di-*hosting* dan dipakai pengguna asli |
| T3. *Active learning* dengan *uncertainty sampling* | **Tercapai**, dengan bukti terkuat pada modul aspek | Aspek F1 ketat +22,54 poin (95% CI +18,25 s.d. +26,79), exact match p = 2,8×10⁻⁵; baris yang dikoreksi 2,9× lebih sering keliru |
| T4. Evaluasi model dan aplikasi | **Tercapai**, kecuali data CSUQ yang belum masuk draf | Golden dataset beku 300 baris, κ antar-anotator 0,669; uji statistik antariterasi; seluruh pengujian otomatis lulus |

---

## 2. Capaian terhadap *quality gate* Bab III (Tabel 3.5)

Bab IV draf saat ini **menilai dengan kriteria yang berbeda** dari Bab III.
Tabel capaian *Evaluation* memakai "akurasi melampaui *majority baseline*",
sedangkan Bab III menetapkan **macro-F1 ≥ 0,80**. Penguji akan membandingkan
keduanya. Tabel berikut memakai kriteria Bab III apa adanya.

| Komponen | Kriteria Bab III | Iterasi 0 | Titik akhir fase 1 (it. 6) | Fase 2 (model akhir) | Status jujur |
|---|---|---|---|---|---|
| Sentimen | Macro-F1 ≥ 0,80 | 75,68 | 78,39 (puncak 82,51 di it. 3) | 78,38 | **Belum terpenuhi** pada titik akhir; sempat terpenuhi di it. 1 dan it. 3; CI 95% it. 6 (71,09–84,27) mencakup 80 |
| Aspek | Macro-F1 ≥ 0,80 | macro 24,46 (parsial) / 29,12 (ketat, support ≥ 2) | macro 43,68 (parsial) / 72,76 (ketat, support ≥ 2); **micro parsial 80,06** | micro parsial 80,33 | **Belum terpenuhi** untuk macro; **terpenuhi** untuk micro parsial. Lihat catatan metrik di bawah. |
| Topik | c_v ≥ 0,50 | – | 0,4998 | – | **Mendekati** (selisih 0,0002, jauh di bawah simpangan antar-*seed* ±0,0221) |
| Adaptasi (AL) | Tren positif macro-F1 | – | sentimen +2,71 poin (tidak signifikan); aspek naik signifikan | sentimen +2,98 poin; aspek turun di it. 2 lalu dipulihkan | **Terpenuhi** untuk tren; signifikansi hanya pada aspek fase 1 |
| Fungsionalitas | Lulus *black box* tanpa kegagalan kritis | – | seluruh pengujian lulus | seluruh pengujian lulus | **Terpenuhi** |
| Kegunaan | CSUQ ≥ 3,50 | – | – | **data belum ada di draf** | **Perlu data kuesioner** |

**Catatan metrik aspek.** Macro-F1 aspek dirata-ratakan atas 336 jenis aspek,
dan 274 di antaranya hanya muncul satu kali di golden dataset. Skor tiap jenis
yang muncul sekali hanya bisa bernilai 0 atau 1, sehingga macro penuh tidak
bermakna secara statistik. Pilihan penyajian yang dapat dipertanggungjawabkan:

- tetap laporkan macro-F1 sesuai Bab III, dan nyatakan kriterianya belum terpenuhi;
- sandingkan dengan macro-F1 untuk aspek ber-*support* ≥ 2 (62 jenis) dan
  micro-F1 sebagai ukuran yang stabil;
- jadikan temuan ini keterbatasan desain kriteria di Bab V, bukan alasan
  mengganti kriteria diam-diam.

---

## 3. Temuan per tujuan

### T1 — Kinerja model

**Sentimen (IndoBERT, `crypter70`).**

- Pada SmSA test (n=500, jalur produksi): akurasi 0,9160, macro-F1 0,8888,
  ECE 0,0209 setelah kalibrasi suhu.
- Pada golden dataset studi kasus (komentar YouTube, n=300, label manusia):
  akurasi 83,00% (CI 78,34–86,83), macro-F1 75,68%, κ model 0,6897, jauh di
  atas *majority baseline* 62,33%.
- **Temuan: kesenjangan domain.** Model yang sama turun ±8 poin akurasi dan
  ±13 poin macro-F1 dari data ulasan (SmSA) ke komentar YouTube. Ini dasar
  empiris kebutuhan adaptasi (T3) dan sejalan dengan batasan masalah nomor 3
  di Bab I.
- Kelas netral paling lemah. Recall netral 0,6932 di SmSA dan 60,87% di golden
  it. 6 (n = 23 saja). Penyebabnya data: netral hanya 10,4% data latih SmSA.

**Aspek (IndoBERT token classification, K6).**

- Eksperimen enam kondisi menunjukkan skor validasi tinggi tidak menjamin
  generalisasi: K1 validasi 0,9607 tetapi TermA 0,0390.
- K6 dipilih karena seimbang: TermA 0,8835, YouTube *held-out* 0,9558.
  **Catatan:** 0,9558 diukur pada label hasil proyeksi leksikon, bukan label
  manusia, jadi bukan bukti generalisasi.
- Ukuran yang sah untuk komentar nyata adalah golden dataset (label manusia).
  Pada iterasi 0: micro-F1 parsial 56,49%, presisi 92,48%, **recall 40,66%**.
  Model sangat konservatif, dan inilah alasan status *Lulus Bersyarat* yang
  dijawab oleh T3.

**Topik (BERTopic).** c_v agregat 0,4998, diversity 0,9599, outlier 12,34%,
65 topik pada enam korpus lembaga. Jumlah topik bergantung *seed*
(13,6 ± 5,0, rentang 6–19), sementara c_v stabil (0,5023 ± 0,0221). Sebutkan
*seed* di naskah.

### T2 — Aplikasi *end-to-end*

- Arsitektur terpisah: layanan analisis FastAPI dan aplikasi web Laravel,
  berkomunikasi lewat REST/JSON.
- 8.385 komentar diproses dalam 87 menit 5 detik (0,62 detik/komentar), tanpa
  kegagalan, dengan kesejajaran masukan–keluaran 8.385/8.385 pada ketiga modul.
- UAT 10/10 skenario berhasil.
- **Belum ada di draf: penerapan pada pengguna asli.** Aplikasi di-*hosting*
  di VPS dengan domain `text-analyze.ing.biz.id` (HTTPS) dan dibuka untuk
  pengguna nyata mulai 24 September 2026. Angka agregat (jumlah pengguna,
  analisis, teks, koreksi) diambil dari server; lihat §8. Yang sudah pasti:
  pengguna asli memberi **115 koreksi**, 79 sebelum pemicu dan 36 sesudahnya.
- Jumlah pengujian otomatis per 27 September 2026: layanan analisis **483**
  lulus, aplikasi web **544** lulus (total 1.027). Draf masih memakai 880 dan
  814 (lihat §5).

### T3 — *Active learning*: bukti utama penelitian

#### 3a. Pemilihan data berbasis ketidakpastian bekerja

Sudah ada di draf dan sudah benar:

- Median keyakinan baris yang dikoreksi 0,5754, sedangkan median seluruh
  kolam 0,9486.
- 164 dari 330 baris yang ditinjau berubah label (49,7%), dibanding laju
  kesalahan dasar 17%. Artinya konsentrasi kesalahan **2,9×**.
- Draf sudah membatasi klaimnya dengan tepat: tidak ada kelompok pembanding
  acak, jadi yang dibuktikan adalah konsentrasi kesalahan, bukan keunggulan
  atas *random sampling*. Pertahankan.

#### 3b. Fase 1: enam putaran pada korpus studi kasus (golden dataset 1)

Tabel ini **belum ada di draf** dan paling penting untuk T3. Sumbernya
`php artisan evaluation:report` dan basis data lokal.

| Iterasi | Kolam | Akurasi | CI 95% | Macro-F1 | κ model | Aspek F1 ketat | Aspek F1 parsial | Presisi / Recall aspek (ketat) | Exact match |
|---|---|---|---|---|---|---|---|---|---|
| 0 (sebelum AL) | 0 | 83,00 | 78,34–86,83 | 75,68 | 0,690 | 51,84 | 56,49 | 89,44 / 36,49 | 27,67 |
| 1 (+218 TNI) | 218 | 86,33 | 81,98–89,76 | 80,25 | 0,739 | 67,86 | 74,52 | 67,91 / 67,82 | 32,33 |
| 2 (+112 Presiden) | 330 | 85,67 | 81,25–89,18 | 78,80 | 0,712 | 72,86 | 79,03 | 71,35 / 74,43 | 36,67 |
| 3 (+62 Kejagung) | 392 | 87,33 | 83,09–90,63 | **82,51** | 0,753 | 73,14 | 78,99 | 74,33 / 71,98 | 40,00 |
| 4 (+21 KPK) | 413 | 85,67 | 81,25–89,18 | 79,93 | 0,714 | 73,15 | 79,55 | 71,78 / 74,57 | 37,67 |
| 5 (+22 Polri) | 435 | 84,33 | 79,79–88,01 | 76,43 | 0,680 | 72,78 | 78,96 | 72,01 / 73,56 | 37,67 |
| 6 (+46 DPR) | 481 | 85,67 | 81,25–89,18 | 78,39 | 0,703 | **74,38** | **80,06** | 73,90 / 74,86 | **40,67** |

Uji utama yang ditetapkan di muka adalah iterasi 0 lawan iterasi 6:

| Modul | Hasil | Uji | Kesimpulan |
|---|---|---|---|
| Sentimen | akurasi +2,67 poin; macro-F1 +2,71 poin | McNemar: 29 membaik, 21 memburuk, **p = 0,322** | Tren naik, **belum signifikan**. Daya uji: perlu perbaikan bersih ≥ 16 baris (≈ 5,33 poin) untuk terdeteksi pada n = 300. |
| Aspek | F1 ketat 51,84 → 74,38 (**+22,54**) | Bootstrap berpasangan 2.000 putaran: **CI 95% [+18,25; +26,79]** | **Signifikan** |
| Aspek | exact match 27,67 → 40,67 | McNemar: 62 membaik, 23 memburuk, **p = 2,8×10⁻⁵** | **Signifikan** |

Cara membaca temuan ini:

1. **Adaptasi terbesar terjadi pada modul yang paling jauh dari domainnya.**
   Model aspek berangkat dari bobot bahasa umum dengan recall 36%. Koreksi
   domain menaikkan recall ke 75%, dengan presisi turun dari 89% ke 74%. Model
   berubah dari sangat konservatif menjadi seimbang. Model sentimen sudah
   dilatih untuk tugas yang sama (SmSA), sehingga ruang perbaikannya lebih
   sempit.
2. **Kenaikan tidak monoton.** Puncak sentimen ada di iterasi 3, lalu turun
   di iterasi 4–5. Tiap putaran menambah lembaga baru, jadi adaptasi sekaligus
   perluasan domain. Laporkan kurvanya, bukan hanya titik awal dan akhir.
3. **Uji utama hanya satu** (awal lawan akhir). Perbandingan antariterasi
   lainnya deskriptif, karena menguji setiap pasangan adalah pengujian
   berganda.
4. **Pembanding kappa.** κ model (0,68–0,75) berada di sekitar κ
   antar-anotator 0,669. Kedua angka ini mengukur hal berbeda, jadi jangan
   ditulis "model setara manusia". Cukup tulis bahwa kinerja model sudah
   berada di wilayah ketidakpastian label manusia.

#### 3c. Fase 2: pengguna asli (belum ada di draf)

Alur yang dijalankan:

1. Aplikasi dipakai pengguna asli dengan **mode tunda** menyala. Pemicu
   pelatihan dicatat tetapi tidak langsung dijalankan.
2. Golden dataset 2 diambil, dianotasi, lalu dibekukan.
3. Iterasi 0 diukur.
4. Pemicu yang tertunda diputar ulang (iterasi 1).
5. Pelatihan dijalankan atas seluruh koreksi (iterasi 2).

**Golden dataset 2**

- 300 baris, acak berstrata dari **seluruh analisis** (enam korpus studi kasus
  ditambah analisis pengguna). Populasinya "seluruh penggunaan aplikasi".
- Kandidat 6.561. Ditolak: terlalu pendek 1.540, sudah di golden 1 sebanyak
  303, sudah dikoreksi dan dilatihkan 402, tidak dapat dinilai 252, duplikat
  226.
- 5 baris yang koreksinya belum dilatihkan ikut terambil. Baris ini otomatis
  disingkirkan dari pelatihan (kolam 560 − 5 = 555 sampel latih sentimen).
- Label: negatif 171, positif 87, netral 42.
- Anotasi memakai aplikasi anotasi dengan tebakan model yang **baru terlihat
  setelah anotator memutuskan**. Tercatat **44 label sentimen (14,7%) berubah
  setelah tebakan terlihat, 42 di antaranya ke arah tebakan model**, dan 66
  daftar aspek (22%) berubah. Lihat keterbatasan di §7.

**Hasil per iterasi (golden dataset 2, n = 300)**

| Iterasi | Data latih | Akurasi (±CI) | Macro-F1 | Weighted-F1 | Aspek F1 parsial | Aspek F1 ketat | Exact match |
|---|---|---|---|---|---|---|---|
| 0 (model akhir fase 1) | 481 koreksi fase 1 | 82,67 (±4,28) | 75,40 | 81,82 | 80,83 | 76,62 | 49,33 |
| 1 (pemicu tertunda, +79 pengguna) | 555 sentimen / 450 aspek | 83,33 (±4,21) | 75,16 | 82,39 | 80,33 | 76,45 | 49,33 |
| 2 (seluruh koreksi, +115 pengguna) | 591 sentimen / 456 aspek | 83,33 (±4,21) | **78,38** | **83,16** | 78,77 | 73,21 | 44,67 |

Uji yang sudah dihitung:

- Sentimen it. 1 → it. 2: 12 membaik, 12 memburuk, p = 1.
- Aspek it. 1 → it. 2: exact match 7 membaik, 21 memburuk, **p = 0,0125**;
  F1 ketat −3,23 poin, **CI 95% [−4,99; −1,56]**. Penurunannya signifikan.

**Keputusan model akhir.** Aspek iterasi 2 lebih buruk secara signifikan,
jadi **model aspek dikembalikan ke iterasi 1**. Model sentimen iterasi 2 tetap
dipakai. Pemulihan diverifikasi: prediksi aspek model pulihan identik 300/300
dengan evaluasi iterasi 1. Model yang dipakai aplikasi = **sentimen iterasi 2
+ aspek iterasi 1**, dan halaman depan aplikasi kini menampilkan angka model
aktif tersebut (akurasi 83,33%, F1 aspek 80,33%).

**Data fase 2 dari server (agregat, 27 September 2026)**

- Pengguna asli: 13 akun, 12 menjalankan analisis (16 analisis, 1.047 teks;
  10 combined, 5 sentimen, 1 topik), semuanya pada 24 September 2026.
- Koreksi: **115 koreksi dari 7 pengguna** pada 7 analisis (24 September,
  12:06–15:27 UTC). Pemicu tertunda tercatat sekali (12:48 UTC, 79 koreksi
  baru, kolam 560).
- Pemilihan berbasis ketidakpastian juga bekerja pada pengguna asli. Median
  keyakinan baris yang dikoreksi 0,589, dibanding 0,910 untuk seluruh 889
  prediksi di analisis mereka. **46,1% label sentimen berubah**, sebanding
  dengan fase 1 (49,7%).
- Pengguna asli **jarang mengoreksi aspek**: hanya 13,9% daftar aspek diubah
  (fase 1: 58,5%), dan rata-rata 0,97 aspek per teks (fase 1: 2,07).
- Golden 2 per kelompok: 273 baris korpus studi kasus dan 27 baris analisis
  pengguna asli.
- **Kappa antar-anotator golden 2: 0,698** (kuat), kesepakatan mentah 81,7%,
  n = 300. Tanpa baris yang diubah setelah tebakan terlihat (n = 233): κ 0,789.
  Kesepakatan aspek antaranotator: F1 62,97 (stem), exact match 42,0%.

**Uji statistik fase 2 (acuan: anotator 1)**

| Perbandingan | Sentimen | Aspek (F1 ketat) |
|---|---|---|
| it. 0 → it. 1 | 13 membaik / 11 memburuk, p = 0,84 | 76,62 → 76,45, CI [−1,49; +1,11] |
| it. 0 → it. 2 | 17 / 15, p = 0,86 | 76,62 → 73,21, CI [−5,17; −1,79], exact p = 0,020 |
| it. 1 → it. 2 | 12 / 12, p = 1 | 76,45 → 73,21, CI [−4,99; −1,56], exact p = 0,0125 |

**Analisis penyebab penurunan (salah anotasi golden atau koreksi pengguna?)**

| Acuan | Sentimen it. 0 / 1 / 2 | Aspek F1 ketat it. 0 / 1 / 2 |
|---|---|---|
| Anotator 1 (label emas) | 82,67 / 83,33 / 83,33 | 76,62 / 76,45 / **73,21** |
| Anotator 2 | 74,33 / 76,00 / 76,33 | 57,14 / 57,09 / 56,44 (CI it. 1→2: −2,25 s.d. +1,06) |
| Konsensus A1 = A2 (n = 245) | 86,12 / 87,76 / 88,16 | – |
| Irisan aspek A1 ∩ A2 | – | 59,63 / 60,02 / **56,73** (CI it. 1→2: −4,63 s.d. −1,94) |
| A1 tanpa baris yang diubah setelah tebakan | 85,41 / 87,12 / 86,27 (n = 233) | 77,99 / 77,48 / **74,51** (n = 234) |

1. **Sentimen tidak menurun.** Terhadap setiap acuan, iterasi 2 sama atau
   lebih baik dari iterasi 0: +2,0 poin terhadap anotator 2 dan +2,0 poin
   terhadap konsensus. Tidak ada yang perlu dijelaskan selain daya uji yang
   kecil.
2. **Penurunan aspek iterasi 2 nyata, bukan artefak label emas.** Penurunan
   bertahan pada baris yang tidak diubah anotator setelah melihat tebakan, dan
   pada aspek yang disepakati kedua anotator.
3. **Label emas memang terjangkar, tetapi arahnya tidak menjelaskan
   penurunan.** Anotator 1 setuju dengan tebakan yang ditampilkan 83,0%
   (sentimen) dan F1 64,6 (aspek), sedangkan anotator 2 hanya 70,3% dan 40,6.
   Model mencapai F1 aspek ±76 terhadap anotator 1, melampaui kesepakatan
   antarmanusia (63). Bias ini **menaikkan angka absolut** semua iterasi, tetapi
   tidak menciptakan selisih iterasi 1 → 2.
4. **Sumber penurunan: iterasi 2 menjadi terlalu royal menandai aspek.**
   Dibanding iterasi 1, muncul 105 aspek baru (86 di antaranya tidak ada di
   label emas) dan 25 hilang (9 di antaranya benar). Presisi turun. Contoh
   aspek keliru baru: *hukum mati*, *orang tua*, *ibu*, *kerja*, *sekolah*,
   *belanja*.
5. **Koreksi pengguna berperan kecil, pelatihan pada data kecil berperan
   besar.** Iterasi 2 hanya menambah **6 sampel latih aspek** (450 → 456).
   Dua di antaranya meragukan: `cucu` pada "Cucu krakatau tuh" dan `adik` pada
   "…dinamain adik KRAKATAU". Kata kekerabatan seperti itu cocok dengan aspek
   keliru baru (*ibu*, *orang tua*, *istri*). Tetapi 6 sampel tidak mungkin
   sendirian mengubah prediksi pada 76 dari 300 baris. Penjelasan yang lebih
   kuat adalah **ketidakstabilan pelatihan ulang pada kolam kecil**: pembagian
   latih/validasi berubah, epoch terbaik berbeda (2 lawan 3), dan kedua model
   **sama baiknya di data validasi** (F1 0,8056 lawan 0,8052) walaupun berbeda
   ±3 poin di golden. Set validasi 91 baris tidak mampu membedakan keduanya.
6. **Kualitas koreksi pengguna secara umum wajar.** Hanya 0,9% aspek yang tidak
   ditemukan di teks (fase 1: 2,6%). Kelemahannya ada pada **cakupan**, bukan
   kesalahan: pengguna jarang menambah aspek, sehingga koreksi mereka hampir
   tidak memberi informasi baru untuk modul aspek.
7. Kelompok pengguna asli di golden 2 hanya 27 baris (sentimen 81,48 → 70,37
   = selisih 3 baris). Terlalu kecil untuk disimpulkan.

Cara membaca fase 2:

1. **Masukan pengguna asli menggeser sentimen ke arah yang benar tetapi
   kecil.** Macro-F1 naik +2,98 poin; akurasi tetap. Wajar, karena 115 koreksi
   baru ditambahkan ke 481 koreksi lama, dan sebagian besar golden 2 berisi
   korpus studi kasus.
2. **Pelatihan ulang bisa memperburuk model, dan golden dataset
   menangkapnya.** Penjaga penerimaan checkpoint membandingkan model baru
   dengan **bobot dasar** pada data validasi kolam, bukan dengan iterasi
   sebelumnya, sehingga penurunan terhadap iterasi 1 lolos. Penurunan itu
   terlihat pada golden dataset dan ditangani dengan memulihkan versi
   sebelumnya. Ini bukti bahwa siklus *monitoring* bekerja sebagai pengaman.
3. **Pemilihan model akhir memakai golden dataset**, sehingga angka golden
   untuk model akhir sedikit optimistis. Nyatakan secara terbuka.
4. Aspek iterasi 0 fase 2 (80,83) dan iterasi 1 (80,33) praktis setara.
   Iterasi 1 dipilih karena sudah memuat koreksi pengguna asli; selisih
   0,5 poin tidak bermakna.

### T4 — Evaluasi

- Golden dataset 1: 300 baris, acak berstrata, *seed* 42, dibekukan
  7 September 2026 dengan sidik jari isi (SHA-256). Anotasi ganda seluruh
  baris dengan κ = 0,669 (kuat) dan kesepakatan mentah 81,3%. Tidak ada satu
  baris pun yang masuk data latih.
- Ukuran 300 dipilih lewat simulasi Wilson (±4,04 poin pada asumsi 85%).
  Realisasi margin 3,77–4,24 poin cocok dengan rancangan.
- Reliabilitas keyakinan pada golden it. 6: ECE 0,0585. Model terlalu percaya
  diri pada pita 0,50–0,70 (akurasi nyata 26,67% lawan rerata keyakinan
  59,93%). Tulis ini sebagai temuan: kalibrasi dari SmSA tidak sepenuhnya
  berpindah ke domain YouTube.
- CSUQ: lihat §8.

---

## 4. Yang perlu ditambahkan ke draf

1. **Tabel hasil iterasi fase 1** (tabel §3b) beserta uji awal–akhir, di
   subbab *Monitoring & Maintenance*, sebelum "Pembahasan Penerapan Active
   Learning". Tanpa tabel ini, kesimpulan "adanya peningkatan kinerja pada
   model ekstraksi aspek" tidak didukung angka.
2. **Subbab fase pengguna asli**, misalnya "Penerapan pada Pengguna Asli",
   memuat hosting, jumlah pengguna/analisis/koreksi, mode tunda dan pemutaran
   ulang, golden dataset 2, hasil iterasi 0–2, dan keputusan model akhir.
   Letakkan di *Deployment* (penerapan) dan *Monitoring & Maintenance*
   (siklus AL).
3. **Hasil CSUQ** di subbab *Evaluation* atau *Deployment*, untuk menjawab
   kriteria kegunaan ≥ 3,50.
4. **Tabel capaian quality gate yang memakai kriteria Bab III** (tabel §2),
   menggantikan tabel capaian *Evaluation* yang sekarang.
5. **Abstrak** (masih teks templat) — tulis paling akhir, ≤ 200 kata.
6. **Daftar pustaka** masih berisi contoh templat (Agresti, Astuti, BPS 2007,
   Cochran, dst.). Semua rujukan dalam teks harus masuk: Tewu et al., Kumar
   et al. (2023), Liu (2012), Mosqueira-Rey et al. (2023), Studer et al.
   (2021), Efstratiou (2026), Apriliani et al. (2025), Adrielvino & Ayunda
   (2026), Röder et al. (2015), Jacobs et al. (2021), Udayana et al. (2023),
   Lewis (1995), Onita (2023), Cohen (1960), Landis & Koch (1977), Guo et al.
   (2017), dan lainnya.
7. **Lampiran** masih templat. Kandidat isi: pedoman anotasi
   (`dokumen-skripsi/docs/PEDOMAN_ANOTASI.md`), kuesioner CSUQ, contoh
   keluaran aplikasi, dan laporan evaluasi.
8. **Daftar isi, daftar tabel, dan daftar gambar** belum diperbarui (masih
   "4.1 Judul Pokok Bahasan", "BAB V ANALISIS DAN PERANCANGAN", dst.).
9. **Kesimpulan** perlu disusun per tujuan (usulan di §9).

---

## 5. Yang perlu diperbaiki (kesalahan faktual dan inkonsistensi)

Rujukan memakai judul subbab di draf.

**Kesalahan isi**

| Lokasi | Tertulis | Seharusnya |
|---|---|---|
| Uji Ablasi Kebijakan *Preprocessing* | "tanpa *preprocessing* menghasilkan Macro-F1 0,9060, dengan 0,9160" | 0,9060 dan 0,9160 adalah **akurasi**. Macro-F1: 0,8716 → 0,8888. Selisih ±1 poin berada di dalam galat baku ±1,3 poin pada n=500, jadi klaim "memberikan pengaruh" harus dilunakkan. |
| Pemilihan Model Dasar Sentimen | crypter70 "secara konsisten mengungguli … pada seluruh instrumen", "keunggulan mutlak" | Tidak benar. mdhugol lebih baik pada ECE mentah, recall positif, dan F1 negatif. crypter70 unggul pada akurasi, macro-F1, kelas netral, ECE setelah kalibrasi, dan uji perilaku (kontrastif 0,417 → 0,917). Selisih ±2 dokumen dari 500 tidak signifikan. Pemilihan didasarkan pada **validasi** (SmSA valid n=1.260) dan uji perilaku, bukan data uji. |
| Hasil Evaluasi Sentimen | "sekitar sepertiga keputusan kelas memicu perbedaan pandangan" | Kesepakatan mentah 81,3%, jadi yang berbeda **18,7%** (±1 dari 5). |
| Hasil Evaluasi Sentimen | "kesepakatan prediksi model … tercatat pada ," | Nilai hilang: κ model = **0,6897**. |
| Hasil Evaluasi Sentimen | "murni berasal dari kecerdasan linguistik", "membuktikan" | Nada berlebihan; cukup "melampaui baseline dengan batas bawah CI 78,34%". |
| Kalibrasi *Confidence Score* dan Kinerja Model Terpilih | Satu subbab memakai akurasi 0,9120, subbab berikutnya 0,9160 | Dua versi pipeline: 0,9120 sebelum audit kamus slang, 0,9160 sesudahnya. Pakai satu versi (0,9160) atau beri keterangan. |
| Hasil Pemrosesan Asinkron | "batas panjang masukan 256 token" | Sentimen memakai 512, aspek 128. Bab IV awal menyebut 128. Samakan. |
| Capaian Kriteria *Deployment* | "Pekerjaan terbesar sekitar 950 detik" | Tabel waktu mencatat TNI 31 menit 25 detik (**1.885 detik**). |
| Capaian Kriteria *Evaluation* | "814 test cases" | Tabel pengujian menulis 880. Angka terbaru 1.027 (483 + 544). Pilih satu titik waktu. |
| Capaian Kriteria *Monitoring* | Tabel: semua "Terpenuhi"; paragraf: "kriteria terakhir terpenuhi sebagian" | Samakan. Usulan: "Terpenuhi sebagian". |
| Golden Dataset (Bab III, Prosedur Anotasi) | "tanpa menampilkan prediksi model"; anotasi ganda seluruh data | Benar untuk golden 1. Golden 2 memakai tebakan yang terlihat setelah anotator memutuskan, dan anotator kedua belum masuk. Tambahkan keterangan. |
| Bab III, Rancangan Golden Dataset | "Seluruh komponen … termasuk pemodelan topik … dievaluasi menggunakan Golden Dataset" | Topik dievaluasi pada korpus penuh, bukan golden (Bab IV sudah benar). Samakan Bab III. |

**Salah salin judul tabel/gambar** (judul menduplikasi tabel lain)

- Tabel ablasi berjudul "Parameter Akuisisi Data melalui YouTube Data API v3".
- Tabel capaian *Data Preparation* berjudul "… *Business and Data Understanding*".
- Tabel per kelas sentimen berjudul "Hasil Pengukuran Komparatif Checkpoint IndoBERT".
- Tabel uji kata dikenal/tak dikenal berjudul "F1 Span Setiap Kondisi pada Empat Korpus Uji".
- Tabel contoh polaritas klausa berjudul "Perbandingan Model Spesialis dengan Model Gabungan".
- Tabel kualitas topik per lembaga berjudul "Enam Kombinasi *Hyperparameter Clustering* Terbaik".
- Tabel UAT berjudul "Keadaan Layanan Sebelum dan Sesudah Pemuatan Bobot".
- Tabel capaian *Monitoring* berjudul "… Tahapan *Deployment*".
- Keterangan gambar polaritas klausa tertulis "lustrasi pelabelan BIO …".

**Penomoran dan kelengkapan**

- Nomor tabel melompat atau ganda: "Tabel 4.3" dipakai dua kali, "Tabel 4.24"
  dua kali, "Tabel 4.x", dan "Tabel 7".
- Banyak baris "Sumber:" kosong, dan "Ket:" kosong di hampir semua tabel.
- Kemungkinan artefak konversi Word ke Markdown: persamaan dan simbol (c_v, κ,
  n, *learning rate* "()"). Periksa di berkas .docx.

---

## 6. Yang sebaiknya dipangkas

Sesuai permintaan: temuan dan pemenuhan tujuan, bukan cerita perbaikan.

- Kalimat seperti "Beberapa ketidaksesuaian ditemukan … kemudian diperbaiki"
  (*Data Preparation*) dan "dengan tiga kriteria dicapai setelah dilakukan
  perbaikan" (*Modelling*). Sajikan hasil akhirnya saja.
- Subbab "Hasil Perbaikan Perhitungan Polaritas Tingkat Klausa". Pertahankan
  sebagai **rancangan yang diterapkan** (polaritas dihitung per klausa) dengan
  contoh, tanpa narasi "sebelum perbaikan".
- Paragraf "Nilai sejati rangkaian pengujian …" tentang anjloknya akurasi
  0,9160 → 0,6740. Itu cerita *debugging*; cukup sebut bahwa pengujian
  mencegah kegagalan senyap.
- Semua insiden hosting (memori, swap, restart) tidak perlu masuk naskah.

---

## 7. Keterbatasan yang harus dinyatakan

1. **Tanpa pembanding *random sampling*.** Yang dibuktikan adalah konsentrasi
   kesalahan (2,9×), bukan keunggulan AL atas pemilihan acak.
2. **Daya uji n = 300.** Perubahan sentimen < ±5 poin tidak dapat dideteksi.
   Hasil "tidak signifikan" bukan bukti "tidak berubah".
3. **Kriteria macro-F1 aspek** tidak cocok untuk ruang label yang didominasi
   aspek yang muncul sekali (§2).
4. **Penjaga penerimaan checkpoint** membandingkan dengan bobot dasar, bukan
   iterasi sebelumnya. Penurunan antariterasi baru terlihat di golden dataset.
   Pembanding yang adil membutuhkan *dev set* berlabel terpisah (saran).
5. **Model akhir dipilih dengan golden dataset**, sehingga angka golden-nya
   sedikit optimistis.
6. **Bias jangkar pada anotasi golden 2**: 44 label berubah setelah tebakan
   terlihat, 42 ke arah model. Uji ketahanan dengan hanya memakai 256 baris
   yang tidak berubah dapat disiapkan bila diperlukan.
7. **Kedua anotator golden 2 melihat tebakan model** setelah memutuskan,
   walaupun tidak saling melihat. κ = 0,698, tetapi kesepakatan aspek
   antaranotator hanya F1 62,97, sehingga definisi aspek masih longgar.
8. **Ketidakstabilan pelatihan ulang pada kolam kecil.** Dua model yang sama
   baiknya di validasi (91 baris) dapat berbeda ±3 poin F1 aspek di golden.
   Penambahan data sedikit (6 sampel) dapat mengubah prediksi pada seperempat
   baris.
9. **Relevansi topikal korpus** 3,5–35%. Hasil per lembaga menggambarkan
   percakapan di sekitar pencarian, bukan opini publik terhadap lembaga (sudah
   ada di draf; pertahankan).
10. **Polaritas per aspek belum pernah diukur** terhadap label manusia. Yang
   diukur adalah ekstraksi aspek.
11. **Teks berhuruf Unicode bergaya** (mis. 𝙂𝙖𝙠 𝙨𝙚𝙞𝙢𝙗𝙖𝙣𝙜) terbaca kosong
    oleh pembersih teks. Satu baris golden terdampak.
12. **Satu anotator** untuk koreksi pengguna selama fase 1 (peneliti sendiri).

---

## 8. Data yang masih harus dilengkapi

| Data | Sumber | Keterangan |
|---|---|---|
| Hasil CSUQ (rerata per dimensi dan total, jumlah responden) | Kuesioner fase uji coba | **Hanya pemilik penelitian yang punya.** Dibutuhkan untuk kriteria kegunaan ≥ 3,50. |
| Rentang tanggal fase uji coba pengguna | Pemilik penelitian | Data sistem: seluruh analisis dan koreksi pengguna terjadi pada 24 September 2026. |

Sudah terisi 27 September 2026: jumlah pengguna, analisis, dan koreksi;
konsentrasi ketidakpastian fase 2; uji statistik fase 2; rincian per kelompok;
κ golden 2 (0,698); dan analisis penyebab penurunan (§3c).

---

## 9. Usulan kerangka kesimpulan (satu butir per tujuan)

1. **(T1)** Model IndoBERT untuk sentimen dan aspek serta BERTopic untuk topik
   berhasil dibangun dan dijalankan pada komentar YouTube berbahasa
   Indonesia. Pada golden dataset berlabel manusia, sentimen mencapai akurasi
   85,67% (macro-F1 78,39), ekstraksi aspek micro-F1 80,06% (parsial), dan
   topik c_v 0,4998. Kriteria macro-F1 ≥ 0,80 belum terpenuhi pada titik
   akhir. Kesenjangan domain dari data ulasan ke komentar media sosial
   (±8 poin akurasi) menjadi dasar kebutuhan adaptasi.
2. **(T2)** Aplikasi web *end-to-end* berhasil direalisasikan dan diterapkan:
   8.385 komentar diproses tanpa kegagalan, 10/10 skenario penerimaan
   berhasil, seluruh pengujian otomatis lulus, dan aplikasi di-*hosting* serta
   dipakai pengguna asli. [+ hasil CSUQ]
3. **(T3)** *Active learning* berbasis *uncertainty sampling* berhasil
   diterapkan. Baris yang dipilih 2,9× lebih sering keliru. Setelah enam
   putaran, ekstraksi aspek meningkat signifikan (F1 +22,54 poin, CI
   +18,25–26,79; recall 36% → 75%). Sentimen naik 2,7 poin tetapi belum
   signifikan pada n = 300. Pada fase pengguna asli, siklus pemantauan
   mendeteksi penurunan signifikan model aspek setelah pelatihan ulang dan
   mempertahankan versi sebelumnya.
4. **(T4)** Evaluasi dilakukan pada golden dataset beku berlabel manusia
   (κ 0,669) dengan uji statistik berpasangan yang ditetapkan di muka, disertai
   pengujian perangkat lunak dan uji penerimaan. Hasil studi kasus
   menggambarkan percakapan di sekitar pencarian nama lembaga, bukan opini
   publik secara keseluruhan.

Saran yang diturunkan dari temuan (melengkapi saran di draf):

- *dev set* berlabel terpisah agar penerimaan checkpoint membandingkan dengan
  iterasi sebelumnya;
- golden dataset lebih besar dan kelompok pembanding *random sampling*;
- ketidakpastian tingkat *span* untuk aspek;
- kriteria aspek berbasis micro-F1 atau macro ber-*support* minimum;
- normalisasi Unicode pada pembersihan teks;
- evaluasi polaritas per aspek.
