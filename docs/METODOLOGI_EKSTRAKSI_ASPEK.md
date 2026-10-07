# Metodologi Fine-Tuning Model Ekstraksi Aspek

Dokumen ini merekam keputusan metodologis, alasan di baliknya, dan bukti
pengukurannya, agar dapat dijelaskan ulang dalam buku skripsi. Setiap angka di
sini berasal dari skrip yang ada di repositori dan dapat dijalankan ulang.

---

## 1. Rumusan Masalah

Tugas: **aspect term extraction** — menandai rentang kata (span) di dalam kalimat
yang merupakan aspek, yaitu entitas atau atribut yang menjadi sasaran penilaian.

Model: `indobenchmark/indobert-base-p1` dengan kepala *token classification*,
skema pelabelan **BIO** (`O`, `B-ASPECT`, `I-ASPECT`).

Tujuan yang ditetapkan: model harus **mengenali pola** letak dan bentuk frasa
nomina yang menjadi sasaran pembicaraan, **bukan menghafal** daftar kata yang
kebetulan muncul di data latih. Konsekuensinya, keberhasilan tidak boleh diukur
hanya pada domain data latih.

---

## 2. Penilaian Sumber Data Anotasi

Empat berkas Label Studio tersedia di awal. Semuanya diperiksa dengan
`scripts/` dan metrik berikut.

| Sumber | span | istilah unik | berkepala verba | span >4 kata | kemunculan tak ditandai |
|---|---:|---:|---:|---:|---:|
| `label_studio_aspect_annotated` (Gemini) | 3.175 | 2.446 | 12,4% | 6,0% | **56,3%** |
| `label_studio_bio_aspect` (bio_v3) | 3.555 | 422 | 6,3% | 0,6% | 9,8% |
| `label_studio_preannotated` (IndoLEM NER) | 5.055 | 1.760 | 4,8% | 1,3% | 20,8% |

### 2.1 Temuan

**(a) Tidak ada anotasi manusia.** Keempat berkas memiliki `annotations: 0`;
seluruhnya keluaran model. Tidak ada label rujukan untuk mengukur kebenaran.

**(b) Ketidakkonsistenan sebagai cacat utama.** Pada berkas Gemini, 56,3%
kemunculan frasa yang di kalimat lain ditandai sebagai aspek justru dibiarkan
tanpa label. Untuk *token classification* ini adalah supervisi yang saling
bertentangan: token yang sama diajarkan sebagai `B-ASPECT` dan `O` sekaligus.

Penyebabnya ditemukan pada skrip anotasi asal (`label_aspect.py`):

```python
def find_substring_indices(text, substring):
    start = text.find(substring)      # hanya kemunculan PERTAMA
    ...
```

ditambah *batching* 200 kalimat per satu prompt, yang membuat model kehilangan
keajekan antar-kalimat.

**(c) Kesepakatan antar sumber sangat rendah**, sehingga penggabungan
(*ensembling*) tidak dapat dilakukan:

| pasangan | Jaccard rata-rata per kalimat |
|---|---:|
| Gemini ↔ bio_v3 | 0,037 |
| Gemini ↔ NER | 0,013 |
| bio_v3 ↔ NER | 0,333 |

Hanya 11 istilah disepakati ketiganya, dan seluruhnya nama entitas
(*jonan, madrid, wolfsburg, blue bird, facebook*) — bukti bahwa ketiga sumber
memiliki definisi "aspek" yang berbeda.

**(d) Kerusakan offset.** Pada `label_studio_preannotated`, 36,7% span memiliki
field `text` yang tidak sesuai dengan `start`/`end`-nya.

### 2.2 Kesimpulan

Keempat berkas tidak layak dipakai apa adanya. Namun **penilaian per-kalimat**
model Gemini umumnya masuk akal; yang rusak adalah keajekan dan kaidah bentuk.
Karena itu strategi yang dipilih bukan menganotasi ulang dari nol, melainkan
memperbaiki cacat sistematisnya.

---

## 3. Metode Anotasi

Tiga jalur pembentukan data latih dibangun, masing-masing dengan skrip sendiri.

### 3.1 Proyeksi leksikon dari koreksi manusia (`build_aspect_dataset.py`)

Untuk domain produksi (komentar YouTube bertema pajak dan korupsi).

1. **Leksikon awal** diambil dari 292 item yang dikoreksi manusia melalui
   antarmuka *feedback* aplikasi Laravel → 34 istilah aspek terverifikasi.
2. **Perluasan** 23 istilah, masing-masing diperiksa pada konteks kalimatnya
   sebelum diterima. Penolakan dicatat beserta alasannya di
   `data/aspect_lexicon.json` bagian `_rejected`.
3. **Proyeksi konsisten**: setiap kemunculan istilah yang diterima ditandai di
   seluruh korpus, dengan batas kata dan penyelesaian tumpang tindih
   (span terpanjang menang).

Hasil: 1.159 teks, 885 beraspek (76,4%), 2.692 span, 0,07% anotasi gagal
dicocokkan.

### 3.2 Penyaringan + proyeksi anotasi LLM (`annotate_aspects.py`)

Untuk domain umum (berita). Alur: **panen kandidat → normalisasi → saring
linguistik → proyeksi konsisten**.

Normalisasi:
- kurung penjelas dibuang: `kantor akuntan publik ( KAP )` → `kantor akuntan publik`
- angka/tahun di ekor dibuang: `kinerja 2003` → `kinerja`
- frasa >4 kata dipangkas ke kepala + dua pewatas

Penyaringan (dengan jumlah penolakan aktual):

| aturan | ditolak | alasan |
|---|---:|---|
| berkepala verba | 288 | aspek adalah sasaran opini, bukan predikat |
| nama diri | 85 | itu ranah NER; melatihnya mengajari model menandai kapitalisasi |
| mengandung angka | 33 | penanggalan/spesifikasi bukan aspek |
| berkepala kata opini | 25 | `lolos`, `menang` adalah opini, bukan sasarannya |
| berkepala kata fungsi | 14 | `tak terkalahkan` bukan frasa nomina |
| berkepala satuan waktu | 1 | — |

Deteksi verba tidak mengandalkan awalan semata, melainkan memverifikasinya
dengan *stemmer* Sastrawi (awalan hanya dianggap verbal bila pengupasannya
menghasilkan akar yang berbeda), ditambah daftar pengecualian nomina agar
`Mentari` tidak dikira `men-` + `tari`.

Akronim konsep (`RUPS`, `BUMN`, `KAP`) sengaja **dipertahankan**, sedangkan nama
diri berkapital (`Manchester City`) dibuang.

Hasil: 2.487 kandidat → 2.041 diterima; 2.126 teks, 1.939 beraspek (91,2%),
4.203 span.

| metrik | Gemini asli | setelah diperbaiki |
|---|---:|---:|
| span berkepala verba | 12,4% | **4,6%** |
| span >4 kata | 6,0% | **0%** |
| kemunculan benar-benar tak ditandai | 56,3% | **0,0%** |

Angka terakhir diverifikasi terpisah: dari seluruh kemunculan 40 frasa
tersering, 69,3% ditandai persis dan 30,7% berada di dalam span yang lebih
panjang (perilaku yang benar); **tidak ada** yang tertinggal tanpa alasan.

### 3.3 Dataset berlabel manusia (`prepare_external_dataset.py`)

Dua dataset IndoNLU diunduh dan dikonversi:

| dataset | domain | dok (train/valid/test) | span aspek |
|---|---|---|---:|
| **TermA** | ulasan hotel AiryRooms | 3.000 / 1.000 / 1.000 | 8.806 |
| **KEPS** | tweet perbankan (keyphrase) | 800 / 200 / 247 | 5.002 |

Keduanya berformat CoNLL IOB dan **dianotasi manusia** — inilah yang tidak
dimiliki keempat berkas awal.

#### Kebocoran split pada distribusi TermA

Pemeriksaan tumpang tindih antar-split menemukan bahwa berkas `train` TermA yang
didistribusikan IndoNLU **memuat seluruh dokumen split `valid`**:

| dataset | valid ada di train | test ada di train |
|---|---:|---:|
| TermA | **996 / 996 (100%)** | 21 / 995 (2,1%) |
| KEPS | 1 / 200 | 0 / 247 |

Akibatnya, bila split resmi dipakai apa adanya, F1 validasi diukur pada data yang
sudah dilatih — angkanya naik monoton (0,9604 → 0,9849 → 0,9955) sehingga
pemilihan epoch terbaik selalu jatuh ke epoch terakhir dan kehilangan fungsinya.

Perbaikan: `prepare_external_dataset.py` membuang seluruh teks yang muncul di
`valid` atau `test` dari `train`. TermA train menjadi 1.959 dokumen unik
(dari 2.970), KEPS 799 (dari 800), dan tumpang tindihnya nol.

Temuan ini perlu disebut di skripsi: angka F1 validasi TermA yang dilaporkan
sebagian pustaka berpotensi terlalu tinggi karena persoalan yang sama.

#### Masalah fidelitas yang ditemukan dan diperbaiki

Endpoint retraining semula menerima `{text, aspects: [...]}` lalu menurunkan
ulang label BIO dengan pencocokan string. Diukur terhadap label emas TermA,
cara itu menghasilkan **104,8%** dari jumlah span aslinya — artinya menandai
kemunculan yang oleh anotator **sengaja** dibiarkan `O`. Contoh:

> "**kamar mandi** kotor tapi kamar tidur bersih" — hanya kemunculan pertama
> yang merupakan aspek.

Perbaikan: payload boleh membawa kunci `spans` berisi offset karakter, dan
`_generate_bio_tags` memakainya apa adanya. Data berlabel manusia karena itu
dilatih tanpa kerusakan label.

---

## 4. Regimen Pelatihan

Diterapkan pada seluruh kondisi agar hasilnya sebanding.

| aspek | nilai | alasan |
|---|---|---|
| model dasar | `indobert-base-p1` | pralatih bahasa Indonesia |
| panjang token | 128 | sama dengan panjang saat inferensi |
| batch | 16 | standar fine-tuning BERT pada CPU |
| learning rate | 3e-5 | rentang lazim BERT (2e-5–5e-5) |
| epoch | 3 | dengan pemilihan epoch terbaik |
| optimizer | AdamW, weight decay 0,01 | standar |
| scheduler | warmup linear 10% lalu peluruhan | menstabilkan langkah awal saat kepala klasifikasi masih acak |
| grad clipping | 1,0 | mencegah ledakan gradien |
| seed | 42 (python, numpy, torch, CUDA) | reproduktibilitas |

### Keputusan yang perlu dijelaskan di skripsi

**Pemilihan epoch terbaik.** Bobot yang disimpan bukan bobot epoch terakhir,
melainkan epoch dengan F1 validasi tertinggi. Setelah beberapa epoch model mulai
*overfit* dan F1 validasi menurun; menyimpan epoch terakhir berarti menyimpan
model yang lebih buruk. Bobot terbaik ditulis ke berkas sementara, bukan disalin
di RAM, karena *state dict* IndoBERT ±440 MB.

**Split validasi.** Untuk TermA/KEPS dipakai **split resmi** dataset agar
angkanya sebanding dengan literatur. Untuk data hasil proyeksi leksikon yang
tidak punya split resmi, dipakai pembagian **stratifikasi 80/20** berdasarkan
ada-tidaknya token aspek — pembagian acak biasa berisiko menghasilkan set
validasi tanpa satu pun aspek, sehingga F1-nya nol semu.

**Penolakan otomatis checkpoint yang memburuk.** Setelah pelatihan, F1 diukur
pada set validasi yang sama seperti sebelum pelatihan. Bila menurun, checkpoint
**tidak ditulis** dan bobot lama dipulihkan dengan memuat ulang dari disk
(bukan menyalin *state dict*). Ini memenuhi fase *Evaluation* pada kerangka
CRISP-ML(Q) yang dipakai skripsi.

**Metrik.** Dilaporkan dua tingkat:
- **token-level** — F1 atas token berlabel aspek. Longgar, sebanding dengan
  angka yang dilaporkan proses retraining.
- **span-level batas persis** — sebuah span dihitung benar hanya bila batas awal
  dan akhirnya sama persis dengan anotasi manusia. Ini metrik yang lazim pada
  literatur ABSA dan jauh lebih ketat.

*Accuracy* sengaja tidak dipakai: 87% token berlabel `O`, sehingga model yang
memprediksi semuanya `O` akan tampak akurat >85% padahal tidak berguna.

---

## 5. Rancangan Eksperimen

Empat kondisi latih, seluruhnya diuji pada tiga set uji yang **sama**.

| kondisi | data latih | label | domain |
|---|---|---|---|
| K1 | 885 dok komentar YouTube | proyeksi leksikon | pajak/korupsi |
| K2 | 1.939 dok berita | LLM disaring + proyeksi | berita umum |
| K3 | 2.483 dok TermA | **manusia** | ulasan hotel |
| K4 | 3.281 dok TermA+KEPS | **manusia** | hotel + perbankan |

Set uji: `terma-test` (1.000 dok, manusia), `keps-test` (247 dok, manusia),
`youtube` (885 dok, proyeksi).

**Logika pengujian.** Setiap kondisi diuji pada domain yang tidak dilatihnya.
Model yang menghafal kosakata akan runtuh; model yang menangkap pola tidak.
Selisih inilah bukti empiris untuk klaim "model membaca pola".

**Catatan penting tentang kolom `youtube`.** Set uji `youtube` berisi seluruh 885
dokumen beraspek, dan 708 di antaranya adalah data latih K1. Karena itu, angka
K1 pada kolom tersebut **bukan** ukuran generalisasi dan tidak boleh dilaporkan
sebagai demikian; ukuran in-domain yang sah untuk K1 adalah **F1 validasi
(0,9607)** yang dihitung pada 178 dokumen tertahan saat pelatihan. Untuk K3 dan
K4 — yang dilatih pada TermA/KEPS — seluruh 885 dokumen itu belum pernah dilihat,
sehingga kolom `youtube` sah sebagai uji lintas domain. Perlakuan yang sama
berlaku untuk K2 terhadap set ujinya sendiri.

Dijalankan dengan `python scripts/run_experiments.py`, setiap pelatihan dan
evaluasi sebagai subproses terpisah agar memori bersih antar kondisi.

### Bukti awal

Sebelum matriks lengkap dijalankan, checkpoint K1 (F1 validasi 0,971 pada
datanya sendiri) diuji pada TermA test:

| | span-level | token-level |
|---|---:|---:|
| F1 | **0,029** | 0,094 |
| precision / recall | 0,364 / 0,015 | 0,741 / 0,050 |
| TP / FP / FN | 8 / 14 / 523 | — |

Kesenjangan 0,971 → 0,029 menunjukkan model tersebut **menghafal leksikon**.
Ini sekaligus peringatan metodologis: F1 tinggi pada data yang labelnya
dihasilkan oleh fungsi deterministik sebagian besar mengukur kemampuan model
meniru fungsi anotasi itu, bukan generalisasi.

---

## 6. Hasil

<!-- AUTO:BEGIN - dihasilkan scripts/render_results.py, jangan disunting -->

Dihasilkan oleh `scripts/run_experiments.py`, dirender dengan
`scripts/render_results.py`. Seluruh kondisi diuji pada set uji yang sama.

### 6.1 F1 span-level (batas persis)

| Kondisi latih | TermA test (hotel, manusia) | KEPS test (perbankan, manusia) | YouTube (pajak, proyeksi) | YouTube held-out (bersih) |
|---|---:|---:|---:|---:|
| **K1** Leksikon (YouTube pajak) | 0,0390 | 0,0100 | 0,9726 ² | 0,9622 |
| **K2** Leksikon+LLM (berita) | 0,5064 | 0,2347 | 0,2849 | 0,2785 |
| **K3** TermA (label manusia) | 0,8870 ¹ | 0,0304 | 0,1097 | 0,1191 |
| **K4** TermA+KEPS (manusia, 2 domain) | 0,8763 ¹ | 0,7217 ¹ | 0,1974 | 0,2028 |
| **K5** K4 + negatif + substitusi + bobot kelas | 0,8650 ¹ | 0,7058 ¹ | 0,2022 | 0,2383 |
| **K6** TermA + YouTube in-domain | 0,8835 ¹ | 0,0372 | 0,9827 ² | 0,9558 ¹ |

¹ In-domain: domain set uji ini ada di data latih kondisi tersebut.
  Angkanya sah sebagai kinerja, tetapi **bukan** ukuran generalisasi.

² Tumpang tindih data latih: 708 dari 885 dokumen set uji YouTube adalah
  data latih K1, sehingga angkanya bahkan bukan uji tertahan. Nilai
  in-domain K1 yang sah adalah F1 validasi pada tabel 6.2.

### 6.2 Ringkasan pelatihan

| Kondisi | train / val | split | epoch terbaik | F1 val sebelum | F1 val sesudah |
|---|---:|---|---:|---:|---:|
| K1 | 707 / 178 | `stratified_80_20` | 3 | 0,1683 | 0,9607 |
| K2 | 1551 / 388 | `stratified_80_20` | 3 | 0,2774 | 0,8367 |
| K3 | 1638 / 822 | `official` | 3 | 0,2710 | 0,9175 |
| K4 | 2435 / 1020 | `official` | 3 | 0,3503 | 0,9081 |
| K5 | 5198 / 1020 | `official` | 3 | 0,3503 | 0,8931 |
| K6 | 2346 / 822 | `official` | 3 | 0,2710 | 0,9188 |

### 6.3 Kurva pelatihan per epoch (F1 validasi)

| Kondisi | epoch 1 | epoch 2 | epoch 3 |
|---|---:|---:|---:|
| K1 | 0,9346 | 0,9595 | 0,9607 |
| K2 | 0,8024 | 0,8290 | 0,8367 |
| K3 | 0,9028 | 0,9159 | 0,9175 |
| K4 | 0,8880 | 0,8990 | 0,9081 |
| K5 | 0,8676 | 0,8835 | 0,8931 |
| K6 | 0,9120 | 0,9123 | 0,9188 |

### 6.4 F1 token-level (metrik longgar, sebagai pembanding)

| Kondisi latih | TermA test (hotel, manusia) | KEPS test (perbankan, manusia) | YouTube (pajak, proyeksi) | YouTube held-out (bersih) |
|---|---:|---:|---:|---:|
| K1 | 0,1001 | 0,0170 | 0,9852 | 0,0000 |
| K2 | 0,6299 | 0,4478 | 0,4302 | 0,4236 |
| K3 | 0,9062 | 0,0842 | 0,1234 | 0,0000 |
| K4 | 0,8995 | 0,8683 | 0,2299 | 0,0000 |
| K5 | 0,8857 | 0,8594 | 0,2224 | 0,2405 |
| K6 | 0,9061 | 0,0926 | 0,9894 | 0,9660 |

<!-- AUTO:END -->

### 6.5 Pembahasan

**Keragaman leksikal data latih**

| kondisi | TTR | 10 istilah teratas menutupi | F1 lintas domain rata-rata |
|---|---:|---:|---:|
| K1 | 3,8% | 75,4% | 0,025 |
| K2 | 45,5% | 13,0% | 0,342 |
| K3 | 20,8% | 36,8% | 0,070 |
| K4 | 29,1% | 20,3% | 0,197 |
| K5 | 18,9% | 10,2% | 0,202 |

**(a) Label manusia tidak menjamin generalisasi.** K3 dilatih anotasi manusia dan
unggul telak in-domain (F1 0,8870 pada TermA test), tetapi kalah dari K2 yang
labelnya otomatis pada kedua uji lintas domain (0,0304 vs 0,2347 di KEPS;
0,1097 vs 0,2849 di YouTube). Asal-usul label bukan penentu tunggal.

**(b) Yang runtuh di luar domain adalah recall, bukan precision.** K3 pada
YouTube masih benar 76,4% ketika menandai sesuatu, tetapi hanya menandai 5,9%
aspek yang ada (TP 159, FN 2.533). Model yang dilatih pada kosakata sempit
menjadi konservatif: ia hanya menembak kata yang dikenalinya. Ini tanda khas
menghafal, dan diagnosis yang lebih berguna daripada F1 tunggal.

**(c) Pelatihan multi-domain memberi generalisasi dengan biaya kecil.**
K3 → K4 menaikkan F1 YouTube dari 0,1097 ke 0,1974 (+80%) sementara F1 TermA
hanya turun 0,0107. Satu model K4 menangani dua domain (0,8763 dan 0,7217)
nyaris setara dengan model spesialisnya.

**(d) Hasil negatif: augmentasi tidak mengonfirmasi hipotesis keragaman.**
Urutan generalisasi K2 > K4 > K3 > K1 mengikuti urutan TTR mereka, sehingga
sempat diduga keragaman leksikal adalah penyebabnya. K5 dirancang sebagai uji
kausal: data K4 dengan ambiguitas tipe aspek dinaikkan 50,0% → 64,0% lewat
substitusi istilah, ditambah contoh negatif dan bobot kelas, tanpa mengubah
domain sumbernya.

Hasilnya **tidak mendukung hipotesis tersebut**:

| | TermA test | KEPS test | YouTube |
|---|---:|---:|---:|
| K4 | 0,8763 | 0,7217 | 0,1974 |
| K5 | 0,8650 | 0,7058 | **0,2022** |
| selisih | −0,0113 | −0,0159 | **+0,0048** |

Kenaikan +0,0048 pada satu-satunya uji lintas domain berada dalam rentang derau,
sementara biayanya nyata di dua set uji lain. Kesimpulan yang jujur: hubungan
TTR–generalisasi pada tabel di atas adalah **korelasi, bukan kausalitas**.
Penjelasan alternatif yang lebih mungkin untuk keunggulan K2 adalah **kedekatan
domain** — teks berita lebih dekat ke komentar publik YouTube daripada ulasan
hotel atau tweet perbankan.

Efek yang benar-benar terukur dari K5 adalah pergeseran precision–recall akibat
bobot kelas: recall naik di seluruh set uji, precision turun. Model menjadi lebih
agresif menandai, bukan lebih mampu menggeneralisasi.

**(e) Batasan.** Tiga perlakuan pada K5 (contoh negatif, substitusi istilah,
bobot kelas) diterapkan bersamaan, sehingga kenaikan atau penurunannya tidak
dapat diatribusikan ke salah satunya. Ablasi terpisah diperlukan untuk itu.
Selain itu KEPS adalah tugas *keyphrase extraction* yang definisinya lebih luas
daripada aspek ABSA, sehingga sebagian selisih lintas domain berasal dari
perbedaan definisi tugas, bukan semata perbedaan domain.

Catatan: rekomendasi model dipindahkan ke §6.8 setelah kondisi K6 diuji.
Kesimpulan berbasis F1 lintas domain saja ternyata tidak memadai - lihat §6.8.


### 6.7 Uji langsung pembacaan pola

F1 lintas domain menunjukkan gejala, bukan mekanisme. `scripts/probe_pattern.py`
mengujinya langsung: lima pola kalimat yang identik, hanya nominanya diganti.

- **seen** — kata yang sering menjadi aspek di data latih
- **unseen** — nomina Indonesia wajar yang tidak pernah menjadi aspek di data latih
- **nonce** — kata yang tidak ada dalam bahasa Indonesia (`blarum`, `kentrasi`, `molusa`)

Model yang menghafal hanya menandai kelompok *seen*. Model yang membaca pola
menandai ketiganya, karena posisi sintaktisnya sama.

| kondisi | seen | unseen | nonce |
|---|---:|---:|---:|
| K1 | 24,0% | 24,0% | **0,0%** |
| K2 | 100,0% | 100,0% | 40,0% |
| K3 | 96,0% | 100,0% | 84,0% |
| K4 | 100,0% | 92,0% | 100,0% |
| K5 | 92,0% | 100,0% | 100,0% |
| **K6** | **100,0%** | **96,0%** | **72,0%** |

K1 tidak pernah menandai kata bentukan: ia hanya mengenali kosakata leksikonnya.
K3–K6 menandai kata yang mustahil dihafal, semata karena posisinya benar. Inilah
bukti mekanistik bahwa model mengekstraksi berdasarkan pola kalimat, bukan
mencocokkan daftar kata.

Uji ini juga mengoreksi kesimpulan sementara pada §6.5: K3 ternyata **membaca
pola dengan baik** (84% pada kata bentukan). Kegagalannya di KEPS karena itu
bukan gejala menghafal, melainkan perbedaan definisi anotasi antar-dataset.

Satu catatan metodologis: selisih *seen* − *unseen* saja tidak cukup untuk
menyimpulkan. Model yang jarang menembak menghasilkan selisih nol pada tingkat
sama-sama rendah (K1: 24% dan 24%), dan itu bukan tanda membaca pola melainkan
tanda diam. Tingkat absolut harus diperiksa lebih dulu.

### 6.8 Model produksi: K6

Evaluasi K1–K5 memunculkan persoalan yang tidak terlihat dari F1 saja. Diuji
pada 80 komentar produksi:

| model terpasang | dokumen beraspek | span ditemukan |
|---|---:|---:|
| K1 | 76 / 80 | 333 |
| K3 | 9 / 80 | 9 |

K3 membaca pola dengan baik tetapi kosakatanya ulasan hotel, sehingga recall di
domain produksi hanya 0,064. Akibatnya perhitungan PMI kekurangan data:
`association` mengembalikan `{}` karena tidak ada aspek yang mencapai ambang 3
dokumen. K1 padat tetapi menghafal (0% pada uji nonce). Keduanya tidak layak.

K6 dilatih pada gabungan **TermA** (mengajarkan pola dan definisi aspek ABSA yang
ketat) dan **708 komentar produksi**, dengan 177 komentar ditahan sebagai set uji
bersih yang tidak pernah dilatih kondisi mana pun.

| | TermA test | YouTube held-out | nonce |
|---|---:|---:|---:|
| K1 spesialis produksi | 0,0390 | 0,9622 | 0,0% |
| K3 spesialis TermA | 0,8870 | 0,1191 | 84,0% |
| **K6 gabungan** | **0,8835** | **0,9558** | **72,0%** |

K6 menyamai kedua spesialis pada domain masing-masing — selisihnya 0,0035 dan
0,0064, keduanya di bawah 0,01 — sekaligus membaca pola.

**Verifikasi ujung-ke-ujung** dengan K6 terpasang, pada 80 komentar produksi:

- 45 aspek ditemukan, 77 dari 80 dokumen beraspek
- `document_aspects` dan `document_topics` sama-sama berukuran 80 (kontrak
  keselarasan dengan Laravel terjaga)
- **16 asosiasi PMI berhasil dihitung** — fitur asosiasi aspek–topik berjalan
  penuh untuk pertama kalinya sejak awal proyek
- `php artisan nlp:test` dari sisi Laravel mengembalikan `aspect_retrained: true`
- 125 tes Python dan 18 tes Laravel pada jalur aspek/asosiasi lulus

**Kesimpulan yang menggantikan rekomendasi awal pada §6.5:** melatih pada
gabungan dataset berlabel manusia dan data domain sendiri lebih baik daripada
memilih salah satunya. Biayanya terhadap masing-masing domain kurang dari 0,01
F1, sementara keuntungannya adalah satu model yang sekaligus akurat di domain
produksi dan mampu menggeneralisasi ke kalimat berpola baru.

### 6.9 Sentimen per-aspek dihitung dari klausa

Verifikasi ujung-ke-ujung memunculkan cacat pada inti ABSA yang tidak terlihat
dari metrik ekstraksi mana pun: sentimen setiap aspek dihitung dari **kalimat
penuh** tempat aspek itu muncul, sehingga semua aspek dalam satu kalimat
mendapat polaritas yang sama.

Diukur sebelum perbaikan, seluruh enam aspek berikut dilaporkan `negative` 100%,
termasuk yang jelas dipuji:

| kalimat | aspek | sebelum | seharusnya |
|---|---|---|---|
| "Pelayanannya bagus sekali tapi harganya mahal" | pelayanannya | negative | positive |
| "Kamarnya bersih dan nyaman, sayangnya wifi lemot" | kamarnya | negative | positive |
| "Pajak naik terus padahal fasilitas umum memadai" | fasilitas umum | negative | positive |

Perbaikannya (`AspectService._clause_around`): polaritas dihitung dari klausa
tempat aspek berada. Pemisahan dilakukan pada tanda baca kalimat **dan**
konjungsi pertentangan (`tapi`, `namun`, `padahal`, `sayangnya`, `sedangkan`,
…), karena di situlah polaritas berbalik.

Dua detail yang menentukan hasilnya:

1. **Konjungsi di awal klausa dibuang.** `padahal fasilitas umum memadai`
   terbaca `negative` oleh IndoBERT, sedangkan `fasilitas umum memadai` terbaca
   `positive` dengan keyakinan 0,99. Konjungsi menandai hubungan antar-klausa,
   bukan opini terhadap aspeknya. Kata negasi sengaja **tidak** ikut dibuang -
   membuangnya akan membalik polaritas, kesalahan yang sama seperti pada
   penghapusan stopword di §4.
2. **Ambang klausa dibandingkan terhadap panjang aspek**, bukan angka tetap.
   Ambang tetap "minimal 3 kata" membuang klausa benar seperti `harganya mahal`
   yang sudah memuat opininya.

Setelah perbaikan, sembilan dari sembilan aspek pada kalimat uji majemuk
dinilai benar. Pada analisis nyata melalui antarmuka Laravel dengan 60 komentar
yang sama:

| aspek | sebelum | sesudah |
|---|---|---|
| negara | netral 0 · negatif 84,1 | netral 11,4 · negatif 72,7 |
| uang | netral 0 · negatif 77,8 | netral 33,3 · negatif 55,6 |
| korupsi | negatif 70 · positif 30 | negatif 60 · positif 40 |

Sebaran menjadi bermakna: dari 32 aspek, 4 dominan positif dan 9 campuran -
sebelumnya hampir seluruhnya negatif seragam karena hanya menyalin sentimen
kalimat.


### 6.10 Finalisasi modul: keselarasan, mode rule-based, dan kinerja

Audit menyeluruh setelah model terpasang menemukan empat persoalan yang tidak
tampak dari metrik model mana pun.

**(a) Kontrak keselarasan indeks putus pada baris kosong.** Validator schema
membuang teks kosong dari daftar dan `TopicService` menyaring teks yang menjadi
kosong setelah pembersihan. Akibatnya lima teks masuk dan hanya empat hasil
keluar, sehingga setiap baris setelah baris kosong tersimpan dengan teks asli
yang salah - `ProcessTextAnalysis::saveResults` memetakan hasil ke `raw_data`
berdasarkan posisi. Kini baris kosong dipertahankan di tempatnya dan
mengembalikan `[]` serta `-1`. Diverifikasi lewat Laravel dengan 118 teks
(dua baris kosong di tengah): `raw_data` = `document_aspects` = `predictions`
= 118, dan baris tepat setelah baris kosong cocok persis dengan `raw_data`.

**(b) Mode rule-based rusak diam-diam.** `_extract_rule_based` mengembalikan
struktur per-dokumen sementara seluruh hilir mengharapkan struktur per-aspek,
sehingga `aspect_sentiments` kosong, statistik nol, dan `document_aspects`
kosong - tanpa satu pun error. Strukturnya kini disamakan, dan mode ini ikut
mendapat pencocokan toleran-imbuhan serta sentimen per-klausa.

**(c) Aspek berulang menumpuk di `document_aspects`.** Satu komentar panjang
bisa mencatat 'pelayanan' dua belas kali. Frekuensinya sudah tersimpan di
`aspects[].count`, jadi daftar per-dokumen kini memuat nama unik saja.

**(d) Kinerja.** Profil pada 80 komentar: ekstraksi aspek 9,0 detik,
pemotongan klausa 8 milidetik, sentimen per-aspek 19,6 detik. Dua perbaikan
diterapkan:

1. Seluruh klausa dari semua aspek dikirim dalam satu panggilan sentimen,
   menggantikan satu forward pass per aspek.
2. Batch diurutkan berdasarkan panjang sebelum dipotong (*length bucketing*).
   Padding dinamis mem-pad seluruh batch sepanjang anggota terpanjangnya,
   sehingga satu komentar panjang membuat 255 teks pendek ikut dihitung pada
   panjang itu. Urutan keluaran dikembalikan ke urutan masukan dan diuji.

Hasil: tahap sentimen 19,6 -> 11,3 detik, total analisis 80 komentar
24,9 -> 18,5 detik. Batas token inferensi juga dipisahkan dari batas pelatihan
(256 vs 128); karena padding dinamis, batas lebih tinggi tidak menambah biaya
pada teks pendek dan justru terukur sedikit lebih cepat.

Setelah keempatnya, sembilan dari sembilan aspek pada kalimat majemuk uji
dinilai benar, termasuk kasus ambigu yang sebelumnya salah.


### 6.11 Penanganan hasil analisis lama

Perbaikan sentimen per-klausa dan penjajaran indeks mengubah MAKNA angka yang
tersimpan, bukan sekadar memperbaikinya. Hasil lama karena itu tidak bisa
ditambal dari basis data - polaritas per-klausa tidak dapat direkonstruksi dari
distribusi yang sudah teragregasi - sehingga analisisnya harus dijalankan ulang.

Perintah `analysis:reprocess-aspect` mengantre ulang analisis yang terpengaruh.
Terdeteksi 41 analisis pada basis data pengembangan.

**Deteksi berbasis penanda versi, bukan tebakan dari isi data.** Percobaan
pertama menandai analisis sebagai "versi lama" bila tidak satu pun aspeknya
berkelas netral, dengan asumsi itu ciri polaritas kalimat yang disalin. Asumsi
itu salah: analisis yang sudah benar pun tidak menghasilkan kelas netral bila
kalimatnya memang tegas. Contohnya analisis #27 - lima kalimat pendek satu
klausa - yang seluruh aspeknya benar (pelayanan positive, harga negative)
tetapi tetap ditandai perlu diproses ulang.

Penggantinya: `ProcessTextAnalysis::PIPELINE_VERSION` ditulis ke
`AnalysisResult.metrics.pipeline_version` setiap kali hasil disimpan. Perintah
membandingkan versi tersimpan dengan versi terkini, sehingga deteksinya pasti
dan tidak menghasilkan positif palsu.

Catatan operasional: `php artisan queue:work` menyimpan kode di memori.
Setelah mengubah job, `queue:restart` harus dijalankan DAN worker baru
dinyalakan - sinyal restart hanya membuat worker yang sedang berjalan keluar.

### 6.12 Deteksi kategori aspek pada mode rule-based

**Masalah.** Mode rule-based semula hanya mencocokkan string: sebuah kalimat
masuk ke kategori yang dideklarasikan pengguna hanya bila kata kategori itu
muncul harfiah. Diukur pada enam kalimat dengan tiga kategori (`pelayanan`,
`harga`, `infrastruktur`), recall pemetaannya **2/6**. "Antriannya lama sekali,
petugasnya lambat merespon" tidak masuk ke `pelayanan`; "Jalan rusak parah"
tidak masuk ke `infrastruktur`. Ini bukan *aspect category detection*
sebagaimana lazimnya ABSA, melainkan pencarian kata.

**Rancangan.** Ekstraksi rule-based dijadikan dua lapis:

| Lapis | Mekanisme | Sifat | Keluaran |
|---|---|---|---|
| 1 | Pencocokan span harfiah + stem Sastrawi | Deterministik, presisi tinggi | `match_type: lexical` |
| 2 | Ekstraktor terlatih (K6) mencari istilah aspek, lalu istilah dipetakan ke kategori terdekat lewat kemiripan kosinus | Probabilistik | `match_type: semantic`, `similarity` |

Lapis 2 memakai dua gerbang independen, dan keduanya diperlukan karena
**gagal pada contoh yang berbeda**: gerbang ekstraktor menolak "Film itu
durasinya terlalu panjang" (tidak ada istilah aspek) yang justru lolos ambang
kemiripan dengan skor 0,291 - di atas kandidat benar terendah 0,271; sebaliknya
gerbang kemiripan menolak `cuaca` (0,07) dan `kucing` (0,12) yang lolos
gerbang ekstraktor.

**Tiga keputusan rancangan, semuanya berdasar pengukuran.**

*(a) Encoder dasar, bukan encoder hasil fine-tuning.* Meski checkpoint K6 sudah
ada di memori, penyematan memakai IndoBERT dasar. Fine-tuning penandaan BIO
menata ulang ruang embedding ke arah "apakah token ini aspek" dan meruntuhkan
struktur topikalnya:

| Encoder | Akurasi | Celah pemisah (benar-min vs asing-maks) |
|---|---|---|
| IndoBERT dasar | 9/10 | **+0,138** (0,260 vs 0,122) |
| K6 hasil fine-tuning | 7/10 | +0,001 (0,201 vs 0,200) |

Dengan celah +0,001 tidak ada ambang yang memisahkan klausa dalam-aspek dari
klausa asing, sehingga jalur semantik menjadi tak berguna. Encoder tambahan
dimuat *lazy*, jadi mode `automatic` tidak menanggung biayanya.

*(b) Representasi kandidat adalah span berkonteks, bukan klausa.* Menyematkan
klausa membuat penugasan digerakkan topik: pada korpus yang seluruhnya membahas
pajak, `DPR` dipetakan ke `pajak` (0,52) alih-alih `pemerintah`. Mean-pool atas
token milik istilah - dihitung di dalam kalimat penuh sehingga tetap peka
konteks - memperbaiki kasus itu.

| Representasi | Domain berbeda | Domain bertumpang-tindih | Total |
|---|---|---|---|
| Klausa penuh | 9/9 | 5/8 | 14/17 |
| **Span berkonteks** | 9/9 | 6/8 | **15/17** |
| Istilah lepas | 8/9 | 6/8 | 14/17 |

*(c) Penyaringan terhadap lapis 1 dilakukan per-span, bukan per-dokumen.*
Melewati seluruh dokumen yang sudah cocok harfiah tampak hemat, tetapi
menghilangkan aspek lain di kalimat yang sama. Pada "Pelayanannya bagus tapi
antriannya lama sekali", `pelayanannya` cocok harfiah lalu `antriannya` tidak
pernah dinilai, sehingga aspek `pelayanan` tercatat 100% positif; setelah
diperbaiki menjadi 50/50 dengan dua klausa yang dinilai terpisah.

**Pendekatan yang diuji lalu ditolak.** Dua alternatif diukur dan tidak dipakai,
karena keduanya tidak menghasilkan ambang yang bisa memisahkan kandidat dalam
aspek dari kandidat asing:

| Pendekatan | Akurasi | Celah pemisah |
|---|---|---|
| Kemiripan kata-vs-kata, IndoBERT mentah | 12/15 | −0,117 |
| Kemiripan kata-vs-kata, `multilingual-e5-small` | 10/15 | −0,064 |
| *Whitening* penuh atas embedding | 11/15 | −0,200 |

Pada pendekatan pertama, kata asing (`cuaca` 0,83; `makanan` 0,82) menyamai
kandidat benar (`angkutan umum` 0,72). Gejalanya adalah anisotropi embedding
BERT - seluruh vektor menumpuk pada kerucut sempit sehingga kosinus mentah
nyaris tak informatif. *Centering* terhadap latar generik memulihkan sebagian
kontras (rasio pisah 1,13x menjadi 2,37x) dan dipertahankan; whitening penuh
justru memperburuk.

**Hasil.** Pada set uji 13 kalimat dengan tiga kategori berbeda-beda: **12/13**
benar, dan keempat kalimat asing ditolak seluruhnya. Recall pemetaan naik dari
2/6 menjadi 6/6. Satu-satunya kegagalan, "Sudah menunggu dua jam belum juga
dilayani", tidak memuat kata benda aspek sama sekali - ini persoalan *implicit
aspect* yang memang di luar jangkauan sistem berbasis ekstraksi istilah.

**Batasan yang harus dilaporkan apa adanya.** Presisi lapis 2 bergantung pada
seberapa berbeda kategori yang dideklarasikan. Pada empat kategori politik yang
saling bertumpang-tindih (`pajak`, `korupsi`, `pemerintah`, `pelayanan publik`),
seluruh istilah hasil ekstraksi jatuh pada pita skor sempit **0,254-0,428**, dan
`rakyat`→`pemerintah` (0,381) justru **lebih tinggi** daripada `dpr`→
`pemerintah` (0,378) maupun `koruptor`→`korupsi` (0,364). Artinya **tidak ada
ambang yang memisahkan pemetaan benar dari yang salah**: menaikkan
`aspect_semantic_min_score` akan membuang pemetaan yang benar lebih dahulu.

Ini bukan cacat implementasi yang bisa disetel, melainkan batas metodologis.
Membedakan "DPR adalah bagian dari pemerintah" dari "rakyat bukan" menuntut
pengetahuan keanggotaan kategori, bukan kemiripan distribusional - dan itulah
sebabnya ABSA arus utama (SemEval-2014/2016 Task 4/5) melakukan *aspect
category detection* secara **terawasi**. Konsekuensinya untuk sistem ini:

1. Setiap kemunculan membawa `match_type` dan `similarity`, dan setiap aspek
   membawa `lexical_matches`/`semantic_matches`, sehingga komposisi yang timpang
   (`pemerintah`: harfiah 9, semantik 117) terlihat, bukan tersembunyi.
2. `aspect_semantic_mapping=false` mengembalikan perilaku harfiah-saja.
3. Jalur andal menuju presisi tetap lingkar *active learning* yang sudah ada:
   koreksi manusia lewat antarmuka Laravel menjadi `training_items`, lalu
   dipakai melatih ulang model.

**Biaya.** Lapis 2 menjalankan ekstraktor atas seluruh teks, bukan hanya yang
belum cocok. Diukur pada 80 komentar: 18,9 detik (automatic) menjadi 46,8 detik
(rule-based). Kenaikan ini diterima karena mode rule-based dipakai untuk analisis
terarah, bukan pemrosesan massal.


---

## 7. Reproduksi

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -q                      # 119 tes

# data
python scripts/build_aspect_dataset.py --corpus data/corpus.json
python scripts/annotate_aspects.py \
    --source C:/Downloads/label_studio_aspect_annotated.json \
    --target C:/Downloads/label_studio_preannotated.json \
    --out data/aspect_news
python scripts/prepare_external_dataset.py --dataset terma
python scripts/prepare_external_dataset.py --dataset keps

# augmentasi (kondisi K5)
python scripts/augment_aspect_dataset.py     --data data/external/combined_train_retrain.json     --negatives data/external/terma/train_gold.json data/external/keps/train_gold.json     --out data/external/k5_train_retrain.json --factor 1

# eksperimen (K1..K5) lalu render tabel hasil
python scripts/run_experiments.py
python scripts/render_results.py
```

Pantau kemajuan pelatihan dari terminal lain:

```powershell
Get-Content data\experiments\progress.log -Wait -Tail 20
```

## 8. Rujukan Data

- IndoNLU (TermA, KEPS): <https://huggingface.co/datasets/indonlp/indonlu>
- Berkas mentah: <https://github.com/IndoNLP/indonlu/tree/master/dataset>
- NusaCrowd, katalog sumber daya NLP Indonesia: <https://arxiv.org/pdf/2212.09648>
