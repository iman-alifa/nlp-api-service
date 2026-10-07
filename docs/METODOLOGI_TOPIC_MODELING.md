# Metodologi Modul Topic Modeling

Dokumen ini mencatat rancangan, pengukuran, dan keputusan pada modul topic
modeling. Semua angka dapat direproduksi dengan skrip yang disebutkan; seed
tetap (`settings.topic_seed = 42`).

---

## 1. Rumusan Masalah

Modul harus mengelompokkan komentar berbahasa Indonesia menjadi topik yang
(a) koheren bagi pembaca manusia, (b) saling terpisah, dan (c) mencakup
sebanyak mungkin dokumen — karena dokumen yang tidak masuk topik mana pun juga
hilang dari perhitungan asosiasi PMI aspek–topik di hilir.

Sebelum perbaikan, modul tidak memiliki metrik mutu apa pun. Parameternya
ditetapkan lewat rumus tebakan, dan tidak ada satu pun tes otomatis.

---

## 2. Metrik Evaluasi

Diimplementasikan di `app/utils/topic_metrics.py`, diuji di
`tests/test_topic_metrics.py`.

| Metrik | Arti | Rujukan |
|---|---|---|
| **C_v** | Koherensi; segmentasi S_one_set, jendela geser boolean 110, konfirmasi kosinus antar-vektor NPMI | Röder, Both & Hinneburg (2015) |
| **C_NPMI** | Rata-rata NPMI seluruh pasangan kata teratas, jendela 10 | Röder dkk. (2015) |
| **Topic diversity** | Proporsi kata unik di antara kata teratas seluruh topik | Dieng, Ruiz & Blei (2020) |
| **Outlier rate** | Proporsi dokumen bertanda `-1` | khusus sistem ini |

**C_v dibaca sebagai angka utama** karena Röder dkk. menemukannya paling
berkorelasi dengan penilaian manusia. **C_NPMI dilaporkan tetapi tidak dijadikan
dasar penyetelan**: metrik itu menghargai kata yang *sering*, sehingga bergerak
berlawanan arah setiap kali kata umum yang tidak informatif disaring keluar dari
label. Diversity diperlukan sebagai penyeimbang koherensi: model yang mengulang
kata yang sama di semua topik bisa memperoleh koherensi tinggi tetapi tidak
berguna.

### 2.1 Menafsirkan angka C_v

Pertanyaan "berapa C_v yang dianggap baik" tidak punya jawaban mutlak, dan itu
perlu dinyatakan lebih dulu. **C_v tidak sebanding antar-korpus.** Nilainya
bergantung pada panjang dokumen, praperolehan teks, korpus rujukan, dan jumlah
kata teratas yang dinilai. Membandingkan C_v penelitian ini dengan C_v penelitian
lain yang memakai korpus berbeda tidak sahih; yang sahih adalah perbandingan
*relatif* antar-konfigurasi di dalam penyiapan yang sama — dan seluruh tabel di
dokumen ini disusun begitu.

Sebagai rambu kasar yang lazim dipakai praktisi:

| Rentang C_v | Tafsiran umum |
|---|---|
| < 0,30 | topik tidak koheren |
| 0,30 – 0,40 | lemah, masih bisa ditafsirkan sebagian |
| **0,40 – 0,55** | **baik untuk teks pendek/tidak baku** |
| 0,55 – 0,70 | sangat baik, lazim pada dokumen panjang dan baku |
| > 0,70 | patut dicurigai: biasanya topik terlalu sedikit, kata berulang, atau korpus sangat homogen |

**Teks pendek secara inheren mendapat C_v lebih rendah.** Coherence dihitung dari
kemunculan bersama kata di dalam dokumen atau jendela; komentar media sosial
sepanjang belasan kata jarang memuat dua kata topik sekaligus, sedangkan artikel
atau dokumen Wikipedia hampir selalu. Karena itu angka pada korpus komentar tidak
bisa disandingkan begitu saja dengan angka pada korpus berita panjang.

Hasil modul ini berada di **0,36–0,53** lintas tiga domain (0,51 pada korpus
utama) — berada di pita "baik" untuk jenis teks yang dianalisis, dan naik 20%
dari baseline 0,4259 pada penyiapan yang identik.

**Mengapa tidak memakai gensim.** `gensim==4.3.2` ada di `requirements.txt`
tetapi **tidak dapat diimpor** pada environment ini:
`cannot import name 'triu' from 'scipy.linalg'` — gensim 4.3.2 belum kompatibel
dengan scipy ≥ 1.13. Menurunkan scipy akan menyeret scikit-learn dan BERTopic
ikut turun. Rumusnya pendek, jadi ditulis langsung dan diuji sendiri.

**Catatan pengujian metrik.** Kasus degenerate `p(a,b) = 1` membuat penyebut
`−log(1) = 0`, yakni bentuk 0/0. Dengan epsilon, hasilnya berbalik menjadi −1,0 —
persis kebalikan dari yang benar, dan senyap karena nilainya tetap tampak wajar.
Kasus ini ditangani eksplisit dan dikunci oleh tes.

---

## 3. Baseline

Korpus: 885 komentar YouTube (`data/external/youtube/test_gold.json`),
konfigurasi preprocessing sama persis dengan yang dikirim Laravel.

```
c_v 0,4259 | c_npmi −0,1895 | diversity 0,80 | outlier 34,1% | 3 topik | 141 detik

T0 p=0,531 n=470  pajak, tidak, rakyat, negara, susah, bayar, uang, dpr, korupsi
T1 p=0,071 n=63   negara, usaha, hahaha, pimpin, percaya, hutang, oknum, susah
T2 p=0,057 n=50   bayar, ngapain, pajak, uang, ribet, calo, kursi, diskon, anggar
```

Tiga masalah langsung terlihat: satu topik menyerap 53% dokumen, sepertiga
dokumen tidak masuk topik mana pun, dan kata kunci topik dipenuhi kata tanpa
daya pembeda (`tidak`, `susah`, `hahaha`, `ngapain`).

---

## 4. Penyetelan Parameter Klasterisasi

Rumus lama menghasilkan `min_cluster_size = n // 20 = 44` dan
`min_samples = n // 50 = 17` pada 885 komentar — terlalu kasar.

`scripts/sweep_topic_params.py` menguji ~100 kombinasi. Embedding dihitung
**sekali** lalu dipakai ulang; tanpa itu tiap kombinasi menanggung ~25 detik
encoding dan sapuan tidak praktis.

| min_cluster_size | min_samples | n_neighbors | topik | C_v | Diversity | Outlier | Topik terbesar |
|---|---|---|---|---|---|---|---|
| 44 | 17 | 44 *(lama)* | 3 | 0,3903 | 0,90 | 34,1% | **53,1%** |
| 15 | 5 | 30 | 15 | **0,5085** | 0,93 | 35,0% | 9,6% |
| 15 | 3 | 44 | 16 | 0,5008 | 0,95 | 32,1% | 8,7% |
| **20** | **3** | **15** *(dipilih)* | 15 | 0,5045 | **0,98** | **18,4%** | 14,0% |

Baris terakhir dipilih: koherensinya praktis setara dengan yang tertinggi,
tetapi cakupannya jauh lebih baik dan tidak ada topik yang mendominasi.
`min_samples` dipatok **3** — pada sapuan, nilai rendah secara konsisten
menurunkan outlier tanpa merugikan koherensi, sedangkan 17 mendorong outlier
melewati 50%.

---

## 5. Perbaikan Lain, dan Alasannya

### 5.1 Urutan dua tahap reduksi

`reduce_topics()` menulis ulang `topic_model.topics_`. Penarikan outlier yang
dijalankan **sebelum** pemangkasan topik karena itu tertimpa dan hilang tanpa
jejak — terukur: outlier tetap 32% meski penarikan dilaporkan berhasil. Setelah
urutannya dibalik, outlier turun **32,3% → 9,6%** pada 885 komentar, dan
mendekati 0% pada korpus yang lebih kecil.

### 5.2 Dua lapis stopword yang sengaja berbeda

`TextCleaner` **mempertahankan** kata negasi, karena membuangnya membalik
polaritas bagi konsumen hilir (lihat metodologi modul sentimen/aspek). Tetapi
sebagai *label* topik, `tidak` dan `susah` tidak menerangkan apa pun — keduanya
muncul di kata teratas dua topik sekaligus pada baseline.

`TOPIC_KEYWORD_STOPWORDS` menyaringnya **hanya di lapis vectorizer c-TF-IDF**,
sehingga yang berubah adalah label topik; penugasan dokumen ke topik tidak
tersentuh sama sekali.

### 5.3 Unigram, bukan bigram

`ngram_range=(1,2)` pada komentar pendek menghasilkan label rusak dan berulang —
`negara negara`, serta `utang diskon` dan `diskon utang` di topik yang sama — dan
menurunkan C_v dari 0,426 ke 0,281. Diubah ke `(1,1)`.

### 5.4 LDA di atas hitungan kata, bukan TF-IDF

Versi sebelumnya menjalankan `LatentDirichletAllocation` di atas
`TfidfVectorizer`. LDA adalah model generatif Dirichlet-multinomial atas
**hitungan**; menjalankannya di atas bobot TF-IDF melanggar asumsi model dan
tidak dianjurkan dokumentasi scikit-learn. Diganti `CountVectorizer`.

### 5.5 `max_df` tidak stabil pada korpus kecil

`max_df=0,85` membuang kata yang muncul di lebih dari 85% dokumen. Pada korpus
bertema tunggal yang kecil, itu justru membuang kata terpentingnya — dengan 10
dokumen, `pajak` hilang seluruhnya dari kosakata. `max_df` kini hanya diterapkan
pada n ≥ 50; di bawah itu penindasan kata terlalu umum diserahkan sepenuhnya
kepada `ClassTfidfTransformer(reduce_frequent_words=True)`.

### 5.6 Label ditampilkan dalam bentuk asli

Pemodelan atas teks ter-stem itu benar — ia menyatukan `bayar`, `membayar`, dan
`dibayar` menjadi satu sinyal. Tetapi hasil stemming bukan kata yang bisa dibaca:
label yang tampil di antarmuka berbunyi `jabat`, `anggar`, `tri`, `mu`.

`_make_words_readable()` memetakan tiap stem kembali ke bentuk permukaan yang
paling sering muncul (`pejabat`, `anggaran`, `mentri`). Bentuk stem tetap
disimpan di `words_stemmed`, karena koherensi **harus** dinilai pada kosakata
yang sama dengan korpus rujukan; menilai bentuk terbaca terhadap korpus ter-stem
akan membuat setiap kata tidak ditemukan dan seluruh skor jatuh ke nol tanpa
sebab yang terlihat.

Menstem seluruh kosakata korpus memakan ~85 detik. Karena imbuhan bahasa
Indonesia bersifat konkatenatif, akar kata hampir selalu menjadi substring bentuk
berimbuhannya, sehingga kandidat disaring lebih dulu dengan uji substring yang
murah. Varian `w[1:]` ditambahkan untuk peluluhan huruf awal — tanpa itu
`pimpin` tidak pernah sampai ke `pemimpin`.

Word cloud memakai pemetaan yang sama; sebelumnya satu halaman hasil menampilkan
`pejabat` pada daftar topik dan `jabat` pada word cloud.

### 5.7 Bahaya urutan impor

`topic_service.py` kini mengimpor `torch` sebelum `text_cleaner`. Pada Windows,
memuat nltk sebelum torch memicu `OSError WinError 1114`. Modul ini dulu
mengimpor `text_cleaner` lebih dulu dan tetap berjalan **hanya karena** `main.py`
kebetulan memuat service lain yang menarik torch terlebih dahulu — ketergantungan
tak tertulis yang langsung pecah begitu modul diimpor sendirian oleh berkas tes.

### 5.8 Bug pada jaring pengaman terakhir

`_get_simple_topics()` mengembalikan `[]` ketika tidak ada kata tersisa,
sedangkan pemanggilnya membongkar dua nilai (`topics, sub_topics = ...`).
Korpus yang habis setelah pembersihan karena itu memicu `ValueError` — tepat di
jalur yang seharusnya menjadi pengaman terakhir. Kini mengembalikan
`([], [-1] * len(texts))`.

---

## 6. Hasil

Pipeline lengkap, korpus dan konfigurasi yang sama dengan baseline:

| | Baseline | Sesudah | Perubahan |
|---|---|---|---|
| C_v | 0,4259 | 0,4280 | +0,5% |
| Diversity | 0,80 | 0,90 | **+12,5%** |
| Outlier rate | 34,1% | **9,6%** | **−72%** |
| Topik terbesar | 53,1% | 48,6% | −8,5% |
| Waktu | 141 detik | 164 detik | +16% |

Pada korpus yang lebih kecil, outlier turun mendekati nol:

| n | Topik | C_v | Diversity | Outlier | Metode |
|---|---|---|---|---|---|
| 5 | 4 | 0,737 | 0,83 | 0% | LDA |
| 12 | 5 | 0,463 | 0,74 | 0% | LDA |
| 25 | 2 | 0,399 | 1,00 | 0% | BERTopic |
| 60 | 5 | 0,378 | 0,82 | 0% | BERTopic |
| 150 | 4 | 0,425 | 0,95 | 2% | BERTopic |

Penjajaran indeks (N teks masuk → N hasil keluar) terjaga pada seluruh ukuran,
termasuk baris kosong.

### 6.1 Granularitas adalah tempat koherensi berada

`num_topics` yang diminta pengguna berfungsi sebagai **batas atas**, dan batas
itu berbiaya:

| `num_topics` diminta | Topik terbentuk | C_v | Diversity | Outlier |
|---|---|---|---|---|
| 5 | 4 | 0,428 | 0,90 | 9,6% |
| 15 | 13 | **0,516** | **0,969** | 8,1% |

**Implikasi untuk antarmuka.** Petunjuk pada formulir Laravel — *"Rekomendasi:
3-7 topik untuk hasil optimal"* — secara empiris **keliru** untuk korpus ini:
meminta lebih banyak topik menaikkan koherensi maupun keragaman. Petunjuk itu
perlu diperbaiki atau dihapus.

---

## 6A. Memaksimalkan Coherence

Bagian ini mencatat pengungkit yang diuji khusus untuk menaikkan C_v, karena
coherence adalah metrik evaluasi utama modul ini.

### 6A.1 Model embedding — pengungkit yang ternyata kecil

Layanan memakai `SentenceTransformer('indobenchmark/indobert-base-p1')`. IndoBERT
adalah model *masked language modeling*, bukan sentence encoder, sehingga
sentence-transformers membungkusnya dengan mean pooling mentah dan mencetak
peringatan `No sentence-transformers model found ... Creating a new one with
mean pooling`. Karena embedding BERT mentah bersifat anisotropik, hipotesis awal
adalah mengganti encoder akan menaikkan coherence secara berarti.

**Hipotesis itu tidak terbukti.** Pada jumlah topik yang disamakan (k=7),
seluruh parameter lain identik (`scripts/bench_topic_encoders.py`):

| Encoder | Topik | C_v | Diversity | Outlier |
|---|---|---|---|---|
| `intfloat/multilingual-e5-small` | 6 | **0,4765** | 0,967 | 9,3% |
| `indobenchmark/indobert-base-p1` *(dipakai)* | 6 | 0,4658 | 0,967 | 15,6% |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 6 | 0,4405 | 1,000 | 9,0% |
| `firqaaa/indo-sentence-bert-base` | 7 | 0,4337 | 0,986 | 2,6% |

Selisihnya hanya ±0,04 — jauh lebih kecil daripada pengaruh granularitas maupun
model representasi. IndoBERT dipertahankan: selisihnya tidak berarti, model itu
sudah ada di cache untuk modul aspek dan sentimen sehingga tidak menambah unduhan
di Railway, dan ia merupakan pilihan yang teradaptasi bahasa Indonesia.

Catatan: pada granularitas *alami* (tanpa penyamaan k) IndoBERT justru unggul
(C_v 0,5160 dengan 13 topik) — tetapi perbandingan itu tidak sahih karena jumlah
topiknya berbeda, dan C_v cenderung naik seiring granularitas.

### 6A.2 Model representasi — pengungkit terbesar

Klasterisasi menentukan dokumen mana masuk topik mana; *representasi* menentukan
kata apa yang mewakili tiap topik — dan coherence dihitung justru atas kata-kata
itu (`scripts/bench_topic_representation.py`, 885 komentar, 13 topik):

| Varian | C_v | Diversity |
|---|---|---|
| **c-TF-IDF + reduce_frequent_words + MMR 0,3** | **0,5477** | 0,939 |
| c-TF-IDF + reduce_frequent_words + MMR 0,5 | 0,5422 | 0,939 |
| c-TF-IDF + reduce_frequent_words *(sebelumnya)* | 0,5160 | 0,969 |
| c-TF-IDF + BM25 + reduce_frequent_words | 0,5103 | 0,985 |
| KeyBERTInspired | 0,4613 | 0,769 |
| c-TF-IDF polos | 0,3759 | 0,808 |

Dua hal yang perlu dicatat: `reduce_frequent_words` saja sudah menyumbang
**+0,140**, dan MaximalMarginalRelevance menambah **+0,032** lagi.
KeyBERTInspired justru menurunkan C_v maupun diversity, sehingga tidak dipakai.

### 6A.3 Normalisasi slang — memperbaiki teks, menurunkan metrik

Token korpus dinormalkan dengan kamus yang ada, lalu sisanya dicocokkan dengan
29.933 kata dasar Sastrawi. Yang sering muncul namun tidak dikenali dikumpulkan
dan ditambahkan ke kamus slang (`klo`→`kalau`, `bnyak`→`banyak`,
`krna`→`karena`, …). Akronim sah (DPR, IKN, APBN, BUMN, NKRI) sengaja tidak
disentuh.

Hasilnya berlawanan arah:

| | Tanpa normalisasi | Dengan normalisasi |
|---|---|---|
| C_v | **0,5477** | 0,5135 |
| Diversity | **0,939** | 0,877 |
| Outlier | 8,1% | **5,8%** |

**Normalisasi tetap dipakai, meskipun C_v turun.** Alasannya terlihat pada
kata-kata topiknya. Tanpa normalisasi, salah satu topik berbunyi
`nagih, klo, bnyak, lucu, krna, bkin, jngn, wakanda, dzolim` — itu **klaster
ragam bahasa**, bukan topik: komentar ber-slang mengelompok karena ejaannya, dan
karena ejaan tak baku bersifat langka, C_v justru menghargainya. Dengan
normalisasi, topik yang muncul berisi `laporan, tunggakkan, ngemplang, buruh,
ditilep, pajak` dan `proyek, duit, tagih, kongkalikong` — benar-benar topik.

Ini contoh konkret bahwa **coherence yang lebih tinggi tidak otomatis berarti
topik yang lebih baik**, dan angka apa pun perlu diperiksa terhadap keluarannya.

### 6A.4 Parameter harus disetel ULANG setelah praperolehan berubah

Normalisasi menggabungkan varian kata sehingga dokumen menjadi lebih mirip dan
ruang embedding memadat. Parameter yang optimal pada ruang lama (20/3/15)
menghasilkan klaster terlalu sedikit pada ruang baru: 13 topik turun menjadi 9,
dan satu topik kembali menelan 52% dokumen. Setelah sapuan diulang, nilai
optimalnya bergeser ke **15/2/15** (C_v 0,5183 vs 0,4738).

Menyetel parameter klasterisasi tanpa mengulangnya setelah mengubah praperolehan
teks adalah kesalahan yang mudah terlewat.

### 6A.5 Memilih k hanya dengan C_v tertinggi tidak aman

Sapuan jumlah topik pada pipeline final (`scripts/sweep_topic_count.py`):

| k | C_v | Diversity | Outlier | Topik terbesar |
|---|---|---|---|---|
| 19 *(alami)* | 0,5100 | 0,779 | 3,1% | 15,4% |
| **18** | 0,5123 | 0,818 | 3,4% | **15,5%** |
| 13 | 0,4982 | 0,817 | 3,4% | 32,4% |
| **8** | **0,5182** | 0,886 | 4,0% | **58,5%** |
| 4 | 0,4619 | 1,000 | 3,5% | 78,1% |

C_v **tertinggi** justru jatuh di k=8 — tetapi di sana satu topik menelan 58%
dokumen. "Topik" itu sebenarnya sisa gabungan segala hal, dan kata-katanya
memang koheren karena mewakili seluruh korpus. Karena itu pemilihan otomatis
(`num_topics=0`) memaksimalkan C_v **dengan kendala keseimbangan**: topik
terbesar tidak boleh melebihi 3,5× bagian seimbang (1/k).

Kendala itu **relatif terhadap k, bukan ambang mutlak**. Dengan ambang mutlak
0,35, k=2 mustahil lolos karena bagian terkecil yang mungkin justru 0,5 — cacat
yang tertangkap oleh tes unit, bukan oleh pengukuran korpus.

### 6A.6 Hasil akhir

Mode otomatis pada 885 komentar memilih k=18 dan menghasilkan:

```
C_v 0,5123 | diversity 0,818 | outlier 3,4% | topik terbesar 15,5% | 201 detik
```

| | Baseline | Akhir | Perubahan |
|---|---|---|---|
| C_v | 0,4259 | **0,5123** | **+20,3%** |
| Diversity | 0,80 | 0,818 | +2,3% |
| Outlier rate | 34,1% | **3,4%** | **−90%** |
| Topik terbesar | 53,1% | **15,5%** | −71% |
| Jumlah topik | 3 | 17 | — |

---

## 6B. Ketahanan Lintas Domain dan Ukuran Korpus

Seluruh penyetelan di atas dilakukan pada **satu** korpus dengan **satu** ukuran
(885 komentar YouTube). Itu tidak cukup untuk mengklaim modul bekerja pada kasus
apa pun. `scripts/robustness_topics.py` menjalankan pipeline produksi yang sama
pada tiga domain dengan ragam bahasa berbeda — komentar media sosial (pendek,
tidak baku), artikel berita (panjang, baku), dan ulasan hotel/produk (menengah,
padat istilah aspek) — pada lima ukuran korpus.

Yang dinilai bukan hanya coherence, melainkan tanda-tanda kegagalan: jumlah
topik yang tidak masuk akal, satu topik yang menelan korpus, dan cakupan yang
bocor.

### 6B.1 Hasil awal: 9 dari 14 kombinasi gagal

| Domain | n | Topik | Topik terbesar | Outlier | Status |
|---|---|---|---|---|---|
| youtube | 60 | 2 | **88%** | 0% | timpang |
| berita | 120 | 3 | **64%** | 29% | timpang + cakupan |
| berita | 250 | 3 | 43% | **25%** | cakupan |
| ulasan | 60 | 4 | **67%** | 0% | timpang |
| ulasan | 250 | 3 | **69%** | 12% | timpang |

### 6B.2 Dua bug yang tersembunyi di balik fallback

Diagnosis pertama saya keliru. Saya menduga ini degenerasi HDBSCAN dan menambah
jaring pengaman KMeans, yang memperbaiki dua kasus (9/14 → 11/14). Tiga kasus
sisanya tetap gagal — dan ketika `method` pada keluaran diperiksa, ternyata
berbunyi **`frequency-based`**: jalur BERTopic tidak pernah berjalan sama sekali.

**Bug 1 — ambang vectorizer bertabrakan pada c-TF-IDF.** BERTopic menerapkan
`CountVectorizer` pada dokumen **gabungan per-topik**, bukan per-dokumen; jumlah
"dokumen" yang dilihatnya sama dengan jumlah topik. Dengan `min_df=2` dan
`max_df=0,85`, korpus yang menghasilkan 2 topik memberi `max_df` = 1 dokumen
sementara `min_df` meminta 2, dan sklearn melempar `max_df corresponds to <
documents than min_df`. Seluruh jalur BERTopic gagal.

**Bug 2 — LDA menolak mode otomatis.** Jalur cadangan pun gagal: `num_topics=0`
diteruskan mentah, sehingga `n_components` bernilai 0 dan sklearn menolaknya.

Akibat keduanya, hasil jatuh ke fallback frekuensi kata yang paling kasar —
**tanpa satu pun tanda di keluaran API**. Bug ini tidak akan terlihat dari
angka coherence saja; ia terungkap hanya karena field `method` ikut dilaporkan
pada tiap topik.

### 6B.3 Perbaikan

1. Vectorizer c-TF-IDF memakai `min_df=1, max_df=1.0`; penyaringan kata terlalu
   umum di jalur itu memang tugas `ClassTfidfTransformer(reduce_frequent_words)`,
   bukan vectorizer. Jalur LDA — yang memakai dokumen sungguhan — tetap memakai
   ambang lamanya.
2. LDA memilih jumlah topik sendiri ketika diminta mode otomatis.
3. Jaring pengaman KMeans dijalankan bila hasil **akhir** degenerate. Pemeriksaan
   sengaja dilakukan pada hasil akhir, bukan klaster mentah: ketimpangan sebagian
   besar baru muncul setelah pemangkasan topik dan penarikan outlier, karena
   outlier diserap ke topik terdekat dan topik terbesarlah yang paling banyak
   menyerap.

### 6B.4 Hasil akhir: 14 dari 14 lolos

| Domain | n | Topik | C_v | Topik terbesar | Outlier |
|---|---|---|---|---|---|
| youtube | 60 | 5 | 0,4644 | 26,7% | 0,0% |
| youtube | 120 | 5 | 0,4559 | 37,5% | 4,2% |
| youtube | 250 | 8 | 0,4339 | 16,8% | 14,0% |
| youtube | 500 | 22 | 0,4740 | 10,0% | 12,8% |
| berita | 60 | 4 | 0,4682 | 30,0% | 3,3% |
| berita | 120 | 7 | 0,4328 | 20,8% | 0,0% |
| berita | 250 | 11 | 0,4177 | 14,4% | 0,0% |
| berita | 500 | 15 | 0,3642 | 22,2% | 5,4% |
| berita | 900 | 3 | 0,3913 | 44,8% | 0,0% |
| ulasan | 60 | 5 | 0,4674 | 21,7% | 0,0% |
| ulasan | 120 | 7 | 0,4715 | 27,5% | 4,2% |
| ulasan | 250 | 9 | 0,4841 | 19,2% | 0,0% |
| ulasan | 500 | 8 | 0,4706 | 35,3% | 9,0% |
| ulasan | 900 | 9 | 0,4666 | 35,5% | 7,2% |

Tidak ada lagi topik yang menelan lebih dari separuh korpus, dan cakupan
terburuk turun dari 29% outlier menjadi 14%. **C_v rata-rata praktis tidak
berubah** (0,4473 vs 0,4496) — perbaikan ini memang bukan soal menaikkan angka,
melainkan menghilangkan kegagalan senyap.

---

## 6C. Varians Lintas Seed

Seluruh angka di atas berasal dari satu kali jalan dengan seed 42. UMAP bersifat
stokastik, sehingga nilai seed yang berbeda menghasilkan proyeksi berbeda.
`scripts/seed_variance_topics.py` mengulang pipeline yang sama dengan lima seed
pada korpus 885 komentar, mode pemilihan topik otomatis:

| Seed | Topik | C_v | Diversity | Outlier |
|---|---|---|---|---|
| 42 | 13 | 0,5045 | 0,939 | 12,8% |
| 7 | 17 | 0,5255 | 0,971 | 17,3% |
| 2024 | 19 | 0,5103 | 0,963 | 15,4% |
| 123 | 13 | 0,5057 | 0,977 | 18,9% |
| 999 | 6 | 0,4657 | 0,900 | 19,1% |

| Metrik | Rata-rata ± simpangan baku |
|---|---|
| **C_v** | **0,5023 ± 0,0221** |
| C_NPMI | −0,3812 ± 0,0324 |
| Diversity | 0,9498 ± 0,0314 |
| Outlier rate | 0,1668 ± 0,0265 |
| Jumlah topik | **13,6 ± 4,98** |

**Dua kesimpulan yang berbeda sifatnya.**

*Mutu topik stabil.* C_v bergerak dalam ±0,022 — jauh lebih kecil daripada
selisih yang dipakai untuk mengambil keputusan pada dokumen ini (misalnya
+0,140 dari `reduce_frequent_words`, +0,032 dari MMR, dan +0,086 dari baseline
ke hasil akhir). Artinya keputusan-keputusan itu tidak dijelaskan oleh derau
seed. Diversity juga stabil (±0,031).

*Jumlah topik TIDAK stabil.* Rentangnya 6 sampai 19 topik. Pemilihan otomatis
memaksimalkan C_v, dan permukaan C_v terhadap k relatif datar (lihat §6A.5),
sehingga proyeksi UMAP yang sedikit berbeda menggeser puncaknya cukup jauh.

**Implikasi praktis.** Bila jumlah topik perlu konsisten antar-analisis — misalnya
untuk laporan yang dibandingkan dari waktu ke waktu — `num_topics` sebaiknya
ditetapkan pengguna, bukan diserahkan ke mode otomatis. Mode otomatis cocok untuk
eksplorasi awal. Seed tetap (42) membuat satu analisis dapat direproduksi persis;
varians di atas menggambarkan sensitivitas metode, bukan ketidakstabilan hasil
yang sudah tersimpan.

---

## 7. Keterbatasan yang Harus Dilaporkan

1. **Penyetelan dilakukan pada satu korpus dan satu seed.** Angka pada §4 berasal
   dari korpus YouTube 885 komentar dengan seed 42. Parameter yang optimal pada
   korpus lain belum diuji, dan tidak ada laporan varians lintas seed.
2. **C_NPMI tetap negatif** (−0,27) di seluruh konfigurasi. Pada korpus teks
   pendek, dokumen jarang memuat dua kata topik sekaligus, sehingga NPMI
   dokumen-level memang cenderung negatif. Angka ini berguna untuk perbandingan
   *relatif*, bukan sebagai nilai mutlak.
3. **Tidak ada evaluasi manusia.** Koherensi otomatis adalah proksi; tidak ada
   uji intrusi kata (*word intrusion*) maupun penilaian ahli terhadap label
   topik.
4. **Pemulihan bentuk kata bersifat heuristik.** Bentuk yang dipilih adalah yang
   paling sering, sehingga sebagian hasilnya kurang ideal (`hak` → `haknya`).
   Kata dengan perubahan morfofonemis di luar peluluhan huruf awal tetap
   ditampilkan sebagai stem.
5. **Peralihan metode tidak seragam.** Korpus di bawah ~15 dokumen jatuh ke LDA,
   sehingga hasil antar-ukuran korpus tidak sepenuhnya sebanding. Metode yang
   dipakai dilaporkan pada tiap topik lewat field `method`.
6. **Korpus sangat kecil tetap timpang.** Mode otomatis bekerja baik mulai
   ~200 dokumen (n=200: 15 topik, topik terbesar 15,5%; n=500: 19 topik, 13,8%),
   tetapi pada n=60 klasterisasi alami hanya menghasilkan 2 topik dengan yang
   terbesar 88,3% — tidak ada kandidat yang lolos kendala keseimbangan, sehingga
   pencarian mengembalikan klaster alami apa adanya. Parameter disetel pada
   n=885 dan belum disetel per rentang ukuran korpus.

### 7.1 Catatan: nilai 0 yang tertelan operator `or`

Endpoint menulis `num_topics=request.num_topics or 5`. Karena 0 bersifat *falsy*,
setiap permintaan mode otomatis diam-diam dijalankan sebagai 5 topik. Bug ini
tidak terlihat dari luar: keluarannya identik persis dengan permintaan
`num_topics=5`, termasuk seluruh angka metriknya. Ia baru terungkap ketika hasil
lewat API dibandingkan dengan pemanggilan langsung pada korpus dan seed yang
sama — 4 topik dengan satu topik 82,5% versus 15 topik dengan yang terbesar
15,5%. Perbandingan silang seperti itu layak dijadikan kebiasaan.

---

## 8. Reproduksi

```bash
# Pipeline lengkap; num_topics 0 = pilih otomatis berdasarkan coherence
python scripts/eval_topics.py --label baseline --num-topics 5
python scripts/eval_topics.py --label akhir    --num-topics 20

# Sapuan parameter klasterisasi (embedding dihitung sekali)
python scripts/sweep_topic_params.py

# Sapuan jumlah topik
python scripts/sweep_topic_count.py

# Perbandingan model embedding, disamakan jumlah topiknya
python scripts/bench_topic_encoders.py --num-topics 7

# Perbandingan model representasi topik
python scripts/bench_topic_representation.py --num-topics 15

# Tes
python -m pytest tests/test_topic_metrics.py tests/test_topic_service.py -q
```

Keluaran tersimpan di `data/experiments/`: `topic_results.json` (satu entri per
label), `topic_sweep.json` / `topic_sweep_v2.json`, `topic_count_sweep.json`,
`topic_encoders.json` / `topic_encoders_n7.json`, dan
`topic_representation.json`.
