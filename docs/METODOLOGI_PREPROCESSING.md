# Metodologi Modul Preprocessing

Dokumen ini mencatat rancangan, pengukuran, dan keputusan pada
`app/preprocessing/text_cleaner.py` — sumber kebenaran tunggal untuk seluruh
pembersihan teks pada sistem ini.

---

## 1. Rumusan Masalah

Modul ini melayani **tiga modul analisis dengan kebutuhan yang saling
bertentangan**:

| modul | kebutuhan | alasan |
|---|---|---|
| Sentimen | pertahankan morfologi, negasi, kata fungsi | IndoBERT memakai tokenisasi subword dan dilatih pada teks alami |
| Topik | pembersihan agresif (stemming + stopword) | model bag-of-words diuntungkan penyeragaman |
| Aspek | **jangan bersihkan sama sekali** | ekstraksi berbasis offset karakter |

Karena itu tidak ada satu konfigurasi yang benar untuk semuanya. Yang dibutuhkan
adalah mekanisme agar kebutuhan satu modul **tidak bisa** merusak modul lain.

---

## 2. Profil per Tugas

Sebelumnya aturan itu tersebar: sentimen memaksanya di dalam service-nya
sendiri (`_sanitize_config`), aspek melewatinya diam-diam, topik memakai apa
adanya. Tidak ada satu tempat pun yang menyatakan kebijakannya.

`PROFILES` di `text_cleaner.py` menyatukannya:

```python
TextCleaner.for_task('transformer', config_pengguna)   # sentimen
TextCleaner.for_task('bag_of_words', config_pengguna)  # topik
TextCleaner.for_task('span', config_pengguna)          # aspek -> None
```

| kunci | transformer | bag_of_words | span |
|---|---|---|---|
| `stemming` | False | True | *(tidak membersihkan)* |
| `remove_stopwords` | False | True | |
| `lemmatization` | False | — | |
| `min_token_length` | **1** | 2 | |
| `collapse_repeated_words` | **False** | True | |

**Kunci profil selalu menang atas konfigurasi pemanggil, dan penimpaannya
dicatat ke log.** Itulah yang menjamin permintaan pengguna dari formulir Laravel
tidak bisa merusak modul yang tidak menoleransinya.

`'span'` mengembalikan **`None`**, bukan konfigurasi kosong: ekstraksi aspek
memakai offset karakter, sehingga pembersihan apa pun akan menggeser seluruh
offset. Nilai `None` merekam keputusan itu secara eksplisit alih-alih
membiarkannya tersirat pada `AspectService` yang memang tidak memanggil
`TextCleaner` sama sekali.

### 2.1 Dua parameter baru pada profil transformer

- **`min_token_length = 1`.** Ambangnya dulu 2 untuk semua jalur, sehingga
  `"a"`, angka, dan kata sependek `"di"` lenyap — padahal IndoBERT justru
  memakai kata fungsi dan angka sebagai sinyal. Terukur: `"a"` berubah menjadi
  string kosong dan diperlakukan sebagai baris tanpa isi.
- **`collapse_repeated_words = False`.** Penggabungan kata identik berurutan
  mengubah `"sangat sangat bagus"` menjadi `"sangat bagus"` — itu penekanan,
  bukan duplikasi yang perlu dirapikan.

---

## 3. Audit Kamus Slang

Kamus dipangkas dari **313 menjadi 265 entri**. Ia melakukan empat hal yang
bukan normalisasi, masing-masing diukur pada empat korpus proyek (YouTube,
berita, ulasan, SmSA).

### 3.1 Angka telanjang diterjemahkan menjadi kata

Kamus memetakan `'9'→'yang'`, `'4'→'untuk'`, `'8'→'delapan'`. Setiap bilangan
yang berdiri sendiri karena itu ikut tergantikan:

```
"sesuai pasal 4 ayat 9"   ->  "sesuai pasal untuk ayat yang"
"naik 4 persen tahun ini" ->  "naik untuk persen tahun ini"
```

**Terukur: 161 token di 145 dokumen.** Hanya pola leet yang mengandung huruf
(`j4d1`, `4ku`, `d1`, `k3`, `y4`) dipertahankan, karena pola itu tidak mungkin
merupakan bilangan.

### 3.2 Kata baku diganti sinonimnya

`'mantap'→'bagus'`, `'keren'→'bagus'`, `'jelek'→'buruk'`. Keempatnya kata baku
dengan **kadar yang berbeda**. Ini substitusi makna, bukan normalisasi, dan
dampaknya langsung pada jalur sentimen: token yang dilihat model bukan lagi
token yang ditulis penulisnya, sementara `mantap` adalah penanda positif yang
jauh lebih kuat daripada `bagus`.

**Terukur: 178 token di 163 dokumen.** Aturan sekarang: **kunci harus berupa
bentuk tidak baku.**

### 3.3 Bentuk dasar dipetakan ke berimbuhan

`'cari'→'mencari'`, `'lihat'→'melihat'`, `'coba'→'mencoba'` — arah terbalik dari
normalisasi. Pada jalur topik, stemming melucutinya kembali (kerja sia-sia);
pada jalur sentimen yang tidak memakai stemming, kalimatnya benar-benar berubah:

```
"coba lihat dulu"  ->  "mencoba melihat dulu"
```

**Terukur: 207 token di 187 dokumen.**

### 3.4 Kata ambigu dipaksa satu makna

`'lagi'→'sedang'` adalah pemetaan yang paling sering kena — **604 token di 514
dokumen** — padahal `lagi` juga berarti "kembali" atau "lebih":

```
"dia datang lagi kemarin"  ->  "dia datang sedang kemarin"
```

Juga `'mana'→'dimana'` (`mana` berdiri sendiri: "mana yang benar", "ke mana")
dan `'aku'→'saya'` (`aku` adalah kata baku). Ketiganya dibuang.

### 3.5 Entri identitas dan penghapusan lewat pintu belakang

- **11 entri identitas** (`'beli'→'beli'`, `'makan'→'makan'`) yang tidak
  melakukan apa pun selain memperbesar kamus.
- **14 entri memetakan kata ke string kosong** (`'dong'→''`, `'kok'→''`,
  `'anjay'→''`). Ini kebocoran kebijakan: kata-kata itu **terhapus walaupun
  pemanggil mematikan `remove_stopwords`** — padahal jalur sentimen sengaja
  mempertahankan kata fungsi. Semuanya dipindahkan ke
  `_get_social_media_stopwords()` agar penghapusannya tunduk pada kebijakan
  yang benar.

### 3.6 Dampak terukur

Pada SmSA test (500 dokumen), jalur produksi penuh:

| | sebelum audit | sesudah audit |
|---|---|---|
| Accuracy | 0,9120 | **0,9160** |
| Macro F1 | 0,8855 | **0,8888** |
| Weighted F1 | 0,9091 | **0,9127** |
| ECE | 0,0240 | **0,0209** |
| Uji perilaku | 237/249 | 237/249 *(tetap)* |

Perbaikan ini kecil dan berada di sekitar batas derau, tetapi **arahnya
konsisten dan tidak ada yang memburuk** — dan alasan utama perubahannya memang
kebenaran, bukan angka: `"pasal 4 ayat 9"` tidak boleh menjadi
`"pasal untuk ayat yang"` berapa pun akurasinya.

---

## 3.7 Dua kerusakan lagi pada tahap pembersihan

### Angka dipangkas oleh normalisasi huruf berulang

Pola `(.){2,}` berlaku untuk **karakter apa pun**, termasuk digit:

```
"utang 1000 triliun"     ->  "utang 10 triliun"
"anggaran 100000 rupiah" ->  "anggaran 10 rupiah"
"tahun 2000 lalu"        ->  "tahun 20 lalu"
```

Fatal untuk korpus yang justru membahas nominal pajak dan utang. Polanya kini
`([a-zA-Z]){2,}` — hanya huruf.

### Memangkas ke satu huruf merusak kata berhuruf ganda

`"maaaaf"` menjadi `"maf"`, bukan `"maaf"`. Keputusannya kini **berbasis kamus**:
bentuk dua-huruf dicoba lebih dulu dan diterima bila kata utuhnya ada di
29.933 kata dasar Sastrawi; kalau tidak, baru dipangkas menjadi satu huruf.

| masukan | sebelum | sesudah |
|---|---|---|
| `bagusss` | bagus | **bagus** |
| `maaaaf` | maf | **maaf** |
| `saaangat` | sangat | **sangat** |
| `lamaaa` | lama | **lama** |
| `1000` | 10 | **1000** |

### Tagar dibuang beserta katanya

Pola `#\w+` menghapus seluruh tagar, sehingga
`"baca #pajak dan #korupsi"` menjadi `"baca dan"` — **kata paling topikal
justru hilang, tepat pada modul yang paling membutuhkannya**. Kini hanya tanda
pagarnya yang dibuang. Mention (`@budi`) tetap dihapus seluruhnya: nama akun
bukan isi.

---

## 4. Opsi Konfigurasi yang Ternyata Mati

### 4.1 `remove_punctuation` tidak pernah dipakai

Dibaca di `__init__` lalu tidak pernah dirujuk lagi; tanda baca **selalu**
dibuang oleh `re.sub(r'[^a-zA-Z0-9\s]', ' ', text)` tanpa syarat. Ini penting
bagi jalur transformer: IndoBERT dilatih pada teks bertanda baca, dan tanda seru
membawa sinyal intensitas pada komentar.

Sekarang tanda batas kalimat (`. , ! ?`) dipertahankan bila opsi dimatikan;
sisa emoji, simbol, dan karakter kontrol tetap dibuang tanpa syarat.

### 4.2 `lemmatization` tidak diimplementasikan

Diterima skema API tetapi tidak melakukan apa pun. **Tetap tidak
diimplementasikan** — tidak ada lemmatizer bahasa Indonesia yang mapan, dan
Sastrawi adalah *stemmer* (memotong imbuhan), bukan *lemmatizer* (memetakan ke
bentuk kamus). Bedanya kini ia **mencatat peringatan** alih-alih diabaikan
diam-diam, sehingga pemanggil tidak mengira lemmatisasi berjalan.

### 4.3 Kunci typo multi-kata tidak pernah cocok

`fix_typos` mencocokkan per token hasil `split()`, sehingga kunci seperti
`'terima kasi' → 'terima kasih'` **tidak mungkin cocok**. Frasa kini
dicocokkan lebih dulu pada teks utuh dengan batas kata (`\b`), baru sisanya
per token.

---

## 5. Bahaya Urutan Impor Diperbaiki di Sumbernya

Pada Windows, mengimpor `nltk` sebelum `torch` memicu
`OSError WinError 1114` saat memuat `c10.dll`.

Modul ini dulu mengandalkan **pemanggilnya** untuk mengimpor torch lebih dulu,
dan itu bertahan hanya karena `main.py` kebetulan memuat service ber-torch
duluan. Begitu sebuah tes mengimpor `text_cleaner` sendirian, **seluruh koleksi
tes gagal** — dan itu benar-benar terjadi saat menjalankan
`pytest tests/test_preprocessing.py tests/test_preprocessing_policy.py`.

`text_cleaner.py` kini mengimpor torch lebih dulu sendiri, sehingga modul ini
mandiri tanpa bergantung urutan impor di tempat lain. Perbaikan yang sama sudah
pernah dilakukan pada `topic_service.py`.

---

## 6. Filter Token Langka: Diukur, lalu Ditolak

Masalahnya nyata. Pada 885 komentar YouTube, **63,4% kosakata hanya muncul
sekali** (1 523 dari 2 403 tipe), sebagian besar berupa salah ketik. Karena
c-TF-IDF memberi bobot tinggi pada kata langka, kata-kata itu bisa naik menjadi
label topik — satu kali jalan menghasilkan topik berbunyi:

```
mahakuasah, majak, kendaran, lillahi, fuul, rakya, kabbah, sahroni
```

yaitu **klaster salah ketik**, bukan klaster makna.

`TextCleaner.filter_rare_tokens()` membuang token yang muncul kurang dari
`min_freq` kali di seluruh korpus. Hasil pengukurannya:

| n dokumen | c_v tanpa filter | c_v dengan filter | diversity tanpa → dengan | token terbuang |
|---|---|---|---|---|
| 60 | **0,4884** | 0,3228 | 0,933 → 0,800 | 30,9% |
| 120 | **0,4602** | 0,3940 | 0,930 → 0,686 | 27,3% |
| 250 | **0,4972** | 0,4244 | 0,950 → 0,787 | 21,7% |
| 885 | 0,5139 | 0,5205 | 0,936 → 0,875 | 15,6% |

**Ditolak sebagai default.** Ia hanya menolong pada 885 dokumen, dan selisih
+0,0066 itu jauh di bawah varians antar-seed (±0,0221) — jadi bukan perbaikan
yang nyata. Di bawah itu ia jelas merugikan, dan pada 20 dokumen ia membuang
**separuh token**, yang merusak embedding sekaligus label topik. Diversity turun
di **semua** ukuran.

Fungsinya dipertahankan untuk korpus yang jauh lebih besar, tetapi
`settings.topic_min_token_freq` bernilai **1 (mati)**.

Ini contoh temuan yang layak ditulis di skripsi: **masalah yang nyata tidak
otomatis berarti perbaikannya menolong**, dan satu kali jalan yang buruk bukan
bukti masalah sistematis.

---

## 7. Yang Sengaja Tidak Diubah

- **Normalisasi slang tetap aktif** meski pada SmSA efeknya kecil. Alasannya
  domain: normalisasi menyentuh **5,0%** token SmSA tetapi **11,2%** token
  korpus YouTube — 2,2 kali lebih banyak. Menyetel kebijakan pada korpus yang
  efeknya separuh domain sasaran adalah kesalahan yang sudah terdokumentasi di
  modul topik.
- **Perlindungan negasi** (`NEGATION_WORDS` dikurangkan dari stopword) tetap
  seperti semula. Ini pengaman kebenaran, bukan penyetelan: 5 dari 6 kalimat
  bernegasi berbalik polaritas pada ~0,99 tanpanya.
- **Akronim dan nama diri** (DPR, IKN, APBN, BUMN, NKRI, Kemenkeu, Jokowi)
  tetap tidak dinormalkan — keduanya memang bukan kata dasar, tetapi merupakan
  istilah sah yang justru bernilai sebagai kata kunci topik.
- **Stemming hanya pada jalur bag-of-words.** Tidak pernah pada sentimen, dan
  pada modul aspek hanya dipakai untuk pencocokan span, bukan pembersihan.

---

## 8. Verifikasi Lintas Modul

Setiap perubahan diuji terhadap ketiga modul analisis, karena inti masalahnya
memang "jangan sampai satu modul merusak yang lain":

| modul | metrik | sebelum | sesudah |
|---|---|---|---|
| Sentimen | accuracy / macro F1 (SmSA test) | 0,9120 / 0,8855 | **0,9160 / 0,8888** |
| Sentimen | uji perilaku (249 kasus) | 0,952 | 0,952 |
| Aspek | span F1 (TermA test, 1 000 dok) | 0,8835 | **0,8835** *(identik)* |
| Topik | c_v (885 komentar, auto-k) | 0,5135 | 0,5139 |

Modul aspek **identik sampai empat desimal**, sebagaimana mestinya: ia tidak
memanggil `TextCleaner` sama sekali. Itu sekaligus bukti bahwa pemisahan
profilnya benar-benar bekerja.

### 8.1 Ketahanan lintas domain tetap utuh

`scripts/robustness_topics.py`, 3 domain x 3 ukuran, **8/8 kombinasi lolos**
(kombinasi youtube n=900 dilewati karena korpusnya 885 dokumen):

| domain | n | topik | c_v | diversity | outlier | terbesar |
|---|---|---|---|---|---|---|
| youtube | 60 | 3 | 0,4884 | 0,933 | 0,133 | 0,333 |
| youtube | 250 | 10 | 0,4972 | 0,950 | 0,096 | 0,200 |
| berita | 60 | 4 | 0,4501 | 0,925 | 0,100 | 0,283 |
| berita | 250 | 11 | 0,4492 | 0,964 | 0,000 | 0,152 |
| berita | 900 | 3 | 0,3764 | 1,000 | 0,000 | 0,448 |
| ulasan | 60 | 5 | 0,3725 | 0,920 | 0,033 | 0,433 |
| ulasan | 250 | 11 | 0,4394 | 0,864 | 0,028 | 0,196 |
| ulasan | 900 | 8 | 0,4313 | 0,925 | 0,118 | 0,249 |

Tidak ada topik yang melampaui separuh korpus dan tidak ada kombinasi yang
gagal. Jaring pengaman KMeans tetap bekerja sebagaimana mestinya — ia menyala
pada berita n=250 saat HDBSCAN hanya menghasilkan 2 klaster.

---

## 9. Reproduksi

```powershell
# Tes unit modul preprocessing (tanpa memuat bobot model)
python -m pytest tests/test_preprocessing.py tests/test_preprocessing_policy.py -q

# Dampak pada modul sentimen
python scripts/evaluate_sentiment.py
python scripts/behavioral_sentiment.py

# Dampak pada modul topik
python scripts/eval_topics.py --num-topics 0 --label cek-preprocessing
python scripts/robustness_topics.py

# Modul aspek (harus tidak berubah - ia tidak memakai TextCleaner)
python scripts/evaluate_aspect.py --gold data/external/terma/test_gold.json
```
