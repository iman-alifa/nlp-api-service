# Apakah yang dijalankan proyek ini benar-benar active learning?

Ditulis 10 September 2026. Semua angka di bawah diukur pada data proyek
sendiri, bukan dikutip dari literatur.

## Kesimpulan

**Ya.** Yang berjalan adalah *pool-based batch-mode active learning* dengan
strategi kueri *least-confidence uncertainty sampling* di atas posterior yang
terkalibrasi, dan protokol pelatihan ulang dari bobot dasar atas seluruh kolam
berlabel. Itu formulasi baku pada Settles (2009) dan Lewis & Gale (1994).

Tiga hal perlu **dinamai jujur** di naskah - bukan diperbaiki, karena dua di
antaranya tidak bisa diperbaiki lagi tanpa merusak perbandingan antar-iterasi.

---

## 1. Yang menentukan sebuah sistem disebut active learning

Bukan pelatihan ulangnya. Yang mendefinisikan active learning adalah **strategi
kueri**: bagaimana baris yang diminta dilabeli manusia itu dipilih. Kalau
dipilih acak, itu *passive learning*, betapapun canggih pelatihannya.

Pada proyek ini pemilihan terjadi di `AnalysisFeedbackController` - antrean
koreksi diurutkan `confidence_score ASC` dengan penyaring "ragu" dan "sangat
ragu" pada ambang yang sama dengan `settings.sentiment_review_threshold` (0,94).

**Terukur pada 330 koreksi nyata:**

| | dikoreksi | tidak dikoreksi |
|---|---|---|
| median keyakinan (analisis #1) | **0,5754** | 0,9486 |
| median keyakinan (analisis #2) | **0,5919** | 0,9485 |
| di bawah ambang 0,94 | **330/330 = 100%** | - |
| proporsi kolam di bawah 0,94 | - | 41,4% / 39,7% |

Kalau anotasinya acak, sekitar 40% koreksi akan jatuh di bawah ambang. Yang
terjadi **100%**, dan seluruhnya di bawah 0,70. Ini bukan sekadar tersedianya
fitur pengurutan - anotator memang bekerja dari puncak antrean.

## 2. Strategi kueri itu terbukti bekerja, pada korpus ini

Asumsi yang mendasari uncertainty sampling: baris berkeyakinan rendah lebih
sering salah. Diuji langsung dengan membandingkan `predicted_sentiment` dan
`corrected_sentiment`:

| pita keyakinan | n | label berubah | |
|---|---|---|---|
| 0,00 - 0,50 | 56 | 32 | **57,1%** |
| 0,50 - 0,70 | 274 | 132 | **48,2%** |
| **seluruh koreksi** | **330** | **164** | **49,7%** |

Model yang sama memberi akurasi 83,00% pada golden dataset (Iterasi 0), jadi
laju kesalahan dasarnya **17%**. Strategi kueri menemukan baris dengan laju
kesalahan **49,7%** - konsentrasi **2,9x**. Dan monoton: makin rendah
keyakinan, makin sering salah.

Untuk aspek, 193 dari 330 daftar aspek berubah (**58,5%**).

Ini angka yang layak dilaporkan sebagai bukti empiris strategi kueri, bukan
sekadar mengutip bahwa uncertainty sampling "biasanya bekerja".

## 3. Yang sudah sesuai praktik terbaik

- **Posterior dikalibrasi sebelum dipakai memilih** (temperature scaling, Guo
  dkk. 2017). Ini sering dilewatkan implementasi lain, dan di sini bukan
  hiasan: mentah, model ini berkeyakinan rerata 0,9926 terhadap akurasi 0,9120
  (ECE 0,0827). Least-confidence di atas posterior seoverconfident itu memilih
  nyaris sembarang baris. Terukur: pada ambang 0,94 antrean menangkap 66,2%
  kesalahan sambil meninjau 12,1% korpus (lift 5,45) di SmSA valid,
  dikonfirmasi 17,0% -> 69,1% (lift 4,06) di SmSA test.
- **Ambang dipilih di validasi, dikonfirmasi di test** - bukan dibaca dari
  tabel yang sama yang dikutip.
- **Pelatihan ulang dari bobot dasar atas seluruh kolam** - formulasi baku
  Settles (2009) §2. Membuat hasil iterasi N hanya bergantung pada isi kolam,
  sehingga satu titik bisa dihitung ulang tanpa mengulang rangkaian.
- **Hold-out beku, diperiksa kebocorannya** - golden 300 baris, dikunci hash,
  dan koreksi yang bertabrakan dengannya disingkirkan dari kolam
  (`RetrainingService::correctedItems()` menahannya).
- **Checkpoint ditolak bila memburuk** (`should_accept_checkpoint`).

**Redundansi batch diperiksa, dan ternyata bukan masalah di sini.** Kelemahan
klasik batch-mode uncertainty sampling adalah k baris paling tidak yakin sering
nyaris duplikat, sehingga anggaran anotasi terbuang - itu yang dijawab
BatchBALD (Kirsch dkk. 2019) dan metode density-weighted. Terukur pada 330
koreksi: hanya **1 kelompok duplikat persis**, dan teks pendek **tidak**
kelebihan terwakili (<=3 kata: 20,0% pada koreksi vs 21,5% pada kolam; median
9 kata vs 8). Jadi tidak perlu diversifikasi - dan ini pernyataan terukur,
bukan asumsi.

---

## 4. Tiga hal yang harus dinamai jujur

### A. Tidak ada pembanding pasif (acak) - ini yang terpenting

Kurva yang naik membuktikan **"lebih banyak label membantu"**, bukan
**"pemilihan aktif membantu"**. Demonstrasi baku pada literatur AL adalah kurva
belajar *active* versus *random* pada **anggaran label yang sama**. Tanpa itu,
klaim yang boleh ditulis adalah "loop active learning berjalan dan model
membaik", bukan "active learning mengungguli anotasi acak".

Perlu dinyatakan sebagai keterbatasan, atau ditutup - lihat §5.

### B. Modul aspek tidak punya sinyal ketidakpastiannya sendiri

`training_items.confidence_score` diisi dari `row['confidence']`, yaitu
**keyakinan sentimen** (`TrainingItemService::extractJsonToTable`). Koreksi
aspek menumpang pada baris yang dipilih karena sentimennya ambigu.

Untuk pelabelan sekuens, praktik yang tepat adalah ketidakpastian tingkat
sekuens - Least Confidence atas seluruh sekuens, entropi token teragregasi,
atau margin token terlemah (Settles & Craven 2008). Itu tidak ada di sini.

Secara empiris ia tetap bekerja: perbaikan aspek iterasi 1 adalah
**ΔF1 +16,03, CI 95% [+11,94, +20,18], p<0,0001** (paired bootstrap). Jadi ini
keterbatasan yang perlu disebut, bukan cacat yang membatalkan hasil - baris
yang membingungkan model sentimen ternyata juga baris yang aspeknya sulit
(58,5% daftar aspeknya berubah).

### C. Sumbu iterasi mencampur dua variabel

Tiap putaran menambahkan **satu lembaga**, jadi "iterasi" berarti *+label* DAN
*+domain*. Pada kurva AL baku, iterasi hanya berarti *+label* dari kolam yang
sama.

Efeknya bukan hipotetis - terlihat jelas pada iterasi 2:

| segmen | it-1 | it-2 | |
|---|---|---|---|
| DPR | 85,7 | 89,3 | +3,6 |
| Presiden | 82,9 | 85,7 | +2,9 |
| Kejaksaan Agung | 91,7 | 93,3 | +1,7 |
| **TNI** | 85,2 | 80,0 | **-5,2** |

Lima naik, satu turun - dan yang turun adalah lembaga yang tadinya menguasai
100% kolam. Itu efek *domain*, bukan efek *jumlah label*.

Penamaan yang jujur: **perluasan domain berurutan dengan uncertainty sampling
di dalam tiap domain**. Sertakan tabel per segmen supaya kedua efek terlihat,
jangan hanya angka agregat.

**Tambahan:** titik awalnya bukan model kosong. Sentimen berangkat dari
`crypter70` (sudah dilatih di SmSA 11.000 dokumen), aspek dari K6. Jadi ini
**active learning untuk adaptasi domain**, bukan AL dari nol. Perlu disebut,
karena pembaca yang melihat "active learning" mengharapkan kurva yang mulai
dari dekat tebakan acak.

---

## 5. Rekomendasi

### Yang JANGAN dilakukan sekarang

**Jangan mengubah strategi kueri di tengah studi.** Menambahkan ketidakpastian
aspek pada iterasi 3 akan membuat iterasi 1-2 dan 3-6 mengukur hal yang
berbeda, dan kurvanya berhenti bisa dibaca. Simpan sebagai saran penelitian
lanjutan.

### Yang layak dipertimbangkan: menutup celah A

Satu perbandingan anggaran-setara, dan ini satu-satunya tambahan yang mengubah
klaim naskah dari "loop berjalan" menjadi "pemilihan aktif terbukti lebih
baik":

1. Ambil **100 baris acak** dari analisis #1 (di luar golden dataset), anotasi.
2. Latih dari bobot dasar hanya atas 100 baris acak itu -> evaluasi di golden.
3. Latih dari bobot dasar atas **100 baris paling tidak yakin** dari 218
   koreksi analisis #1 yang sudah ada -> evaluasi di golden.
4. Bandingkan pada hold-out yang sama.

Biayanya **100 anotasi baru**; langkah 3 tidak butuh anotasi sama sekali karena
labelnya sudah ada. Dijalankan lewat `/api/retrain/sentiment` dengan muatan
eksplisit, `MODEL_PATH` diarahkan ke direktori sementara supaya checkpoint
produksi tidak tersentuh.

Kalau anggaran anotasi tidak ada, **nyatakan celah A sebagai keterbatasan** -
itu jujur dan lazim pada skripsi. Yang tidak boleh adalah menulis "active
learning terbukti lebih efisien daripada anotasi acak" tanpa mengukurnya.

## Rujukan yang dipakai

- Settles, B. (2009). *Active Learning Literature Survey.* CS Tech Report 1648,
  University of Wisconsin-Madison.
- Lewis, D. & Gale, W. (1994). *A sequential algorithm for training text
  classifiers.* SIGIR.
- Settles, B. & Craven, M. (2008). *An analysis of active learning strategies
  for sequence labeling tasks.* EMNLP.
- Guo, C. dkk. (2017). *On calibration of modern neural networks.* ICML.
- Kirsch, A. dkk. (2019). *BatchBALD: Efficient and diverse batch acquisition
  for deep Bayesian active learning.* NeurIPS.
