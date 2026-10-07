# Tinjauan Evaluasi Hold-out

Menanggapi `text-analysis-web/docs/PANDUAN_EVALUASI.md` dan implementasinya
(`app/Services/Evaluation/*`, 5 kelas, 2 228 baris, 28 test hijau).

**Kesimpulan singkat: rancangan evaluasinya benar, dan statistiknya saya
verifikasi ulang secara numerik — tidak ada yang salah.** Ini sudah di atas
standar skripsi S1 pada umumnya. Yang perlu diperbaiki ada **satu cacat
pengukuran serius** (bukan pada model, pada cara mengukurnya), **satu korpus
yang salah pilih**, dan beberapa angka yang sebaiknya ditulis lebih dulu di
naskah supaya tidak jadi bahan pertanyaan penguji.

---

## Bagian 1 — Yang sudah benar, dan sudah saya verifikasi

Ini bukan basa-basi: saya menghitung ulang, bukan membaca komentarnya.

**Uji binomial eksak McNemar cocok sampai 1e-12** dengan referensi eksak
`comb(n,k)` untuk tujuh pasangan (b,c) termasuk kasus tepi b=c dan n=1:

| b | c | implementasi PHP | referensi eksak |
|---|---|---|---|
| 6 | 19 | 0,01463330 | 0,01463330 |
| 3 | 12 | 0,03515625 | 0,03515625 |
| 25 | 40 | 0,08168153 | 0,08168153 |
| 10 | 10 | 1,00000000 | 1,00000000 |

Contoh di §5 panduan (19 membaik, 6 memburuk → p = 0,015) juga benar: p = 0,0146.

**Tabel margin Wilson di §2 cocok persis.** n=100 → ±7,0; n=200 → ±4,9;
n=300 → ±4,0; n=400 → ±3,5; n=1000 → ±2,2. Peringatan per lembaga juga benar:
n=50 pada akurasi 0,85 memberi ±9,9 poin, dan melebar ke ±13,4 poin pada
akurasi 0,50 — persis "±10 sampai ±14" yang ditulis panduan.

Keputusan desain yang tepat dan layak dipertahankan apa adanya:

- **Binomial eksak, bukan khi-kuadrat, untuk b+c kecil.** Alasan di komentar
  benar: pada jumlah pembalikan sedikit — keadaan normal setelah satu-dua
  putaran koreksi — khi-kuadrat memberi p terlalu kecil.
- **Wilson, bukan Wald.** Benar untuk akurasi tinggi pada n beberapa ratus.
- **Bootstrap persentil untuk macro-F1**, karena macro-F1 bukan rerata per
  baris sehingga rumus tertutup tidak berlaku. Disemai tetap, jadi bisa
  dihitung ulang.
- **Tidak ada pengambilan sampel berbasis keyakinan.** Ini yang paling sering
  salah di skripsi sejenis, dan panduannya menolak dengan alasan yang benar.
- **`evaluation:report` hanya menguji iterasi awal vs akhir.** Satu perbandingan
  yang ditetapkan di muka — ini menghindari masalah pengujian berganda tanpa
  perlu koreksi Holm/Bonferroni. Pertahankan (lihat catatan 2.4).
- **Baris tak terbandingkan dihitung terpisah (`unscorable`)**, tidak dibuang
  diam-diam.
- **Kappa antar-anotator dipisahkan** dari kappa model-vs-emas. Keduanya ada dan
  keduanya berbeda artinya.

---

## Bagian 2 — Yang perlu diperbaiki

### 2.1 Metrik aspek salah ukur telak — **prioritas tertinggi**

`ClassificationMetrics::normalizeSet()` mencocokkan nama aspek sebagai **string
persis** (hanya `mb_strtolower` + `trim`). Bahasa Indonesia aglutinatif, dan
mode `automatic` mengembalikan **bentuk permukaan seperti yang tertulis di
teks** — sementara anotator manusia menulis **bentuk dasar**.

Terukur, memakai keluaran `document_aspects` yang sama persis:

| kalimat | emas (tulisan manusia) | prediksi model |
|---|---|---|
| Pelayanannya cepat dan jelas | pelayanan | pelayanan**nya** |
| Sistemnya error terus tiap dibuka | sistem | sistem**nya** |
| Pajaknya naik tapi fasilitasnya begitu saja | pajak, fasilitas | pajak**nya**, fasilitas**nya** |
| Antriannya lama sekali | antrian | antrian**nya** |
| Petugasnya ramah tapi gedungnya kotor | petugas, gedung | petugas**nya**, gedung**nya** |

| cara mencocokkan | TP | FP | FN | micro-F1 |
|---|---|---|---|---|
| string persis (**yang berlaku sekarang**) | 1 | 8 | 7 | **0,118** |
| setelah normalisasi bentuk dasar | 8 | 1 | 0 | **0,941** |

**Modelnya menemukan aspek yang benar pada kedua hitungan.** Selisih 0,118 vs
0,941 seluruhnya artefak pengukuran. Setiap pasangan yang benar dihitung
**dua kali salah**: sekali sebagai false positive (`pelayanannya` dianggap
aspek yang tidak ada di emas) dan sekali sebagai false negative (`pelayanan`
dianggap tidak ditemukan).

Kalau ini dibiarkan, naskah akan melaporkan modul aspek nyaris tidak berfungsi,
dan itu tidak benar. Layanan ekstraksi sendiri sudah memakai pencocokan
berbasis stem Sastrawi di `_find_aspect_spans` justru karena alasan ini —
evaluasinya harus memakai definisi kecocokan yang sama dengan ekstraksinya.

**Sudah saya perbaiki di sisi layanan** (bagian 3), karena akar masalahnya ada
di sana: stemmer Indonesia ada di Python, bukan di PHP.

### 2.2 Koherensi topik diukur pada korpus yang salah

`HoldoutEvaluationService::topicQuality()` menjalankan pemodelan topik pada
**300 baris golden dataset**, bukan pada korpus studi kasus.

Terukur pada korpus yang **sama** dan konfigurasi yang **sama**, hanya jumlah
teksnya berbeda (sampel acak dari korpus yang sama, `data/corpus.json`):

| korpus | c_v | jumlah topik | diversity | rambu sistem sendiri |
|---|---|---|---|---|
| 300 teks (seukuran golden dataset) | **0,3597** | 19 | — | **lemah** (0,30–0,40) |
| 1 139 teks (korpus penuh) | **0,4498** | 18 | 0,961 | **baik untuk teks pendek** (0,40–0,55) |

Selisihnya **0,09 c_v** — jauh di atas variasi seed yang terukur (±0,0221,
`scripts/seed_variance_topics.py`) — dan melintasi batas kategori penafsiran
yang dipakai sistem itu sendiri. Mengukur di test set membuat naskah melaporkan
pemodelan topik sebagai "lemah" padahal yang benar-benar dijalankan pengguna
berada di kategori "baik".

Sebabnya bukan mutu topik, melainkan dua hal sekaligus:

1. **Golden dataset adalah sampel acak berstrata.** Topik yang muncul pada
   sampel bukan topik korpusnya — kalimat-kalimat yang membentuk satu topik
   sebagian besar tidak ikut tersampel.
2. **c_v dihitung dari ko-okurensi kata pada korpus referensi.** Pada 300
   dokumen pendek, matriks ko-okurensinya jarang, dan koherensi tertekan.

**Saran: hitung koherensi pada korpus studi kasus penuh, bukan pada test set.**
Koherensi memang tidak butuh label emas, jadi tidak ada alasan membatasinya ke
baris yang dianotasi. Perannya sebagai kontrol tetap utuh — malah lebih kuat,
karena yang dilaporkan menjadi angka yang benar-benar dihasilkan sistem.

Catatan yang meluruskan dugaan saya sendiri: saya menyangka c_v akan bergoyang
antar iterasi karena UMAP stokastik. **Salah** — diukur tiga ulangan berturut,
c_v identik sampai empat desimal (rentang 0,0000) baik pada mode otomatis
maupun `num_topics` dipatok. Seed-nya memang dikunci. Jadi koherensi **layak**
dipakai sebagai kontrol; yang perlu diganti hanya korpusnya.

### 2.3 Berapa besar perbaikan yang sebenarnya bisa dideteksi

Ini bukan cacat, tetapi angka yang wajib diketahui **sebelum** studi kasus
dijalankan — kalau tidak, Anda bisa mengumpulkan koreksi selama berminggu-minggu
lalu mendapat "tidak signifikan" yang sebenarnya sudah bisa diramalkan.

Daya uji McNemar bergantung pada **jumlah pembalikan (b+c)**, bukan pada n.
Pada α = 0,05 dua sisi:

| baris memburuk (b) | butuh membaik (c) minimal | perbaikan bersih | = poin akurasi pada n=300 |
|---|---|---|---|
| 0 | 6 | 6 baris | 2,0 |
| 2 | 10 | 8 baris | 2,7 |
| 4 | 13 | 9 baris | 3,0 |
| 6 | 17 | 11 baris | 3,7 |
| 8 | 20 | 12 baris | 4,0 |

Artinya: **satu putaran active learning harus memperbaiki bersih sekitar 6–12
baris uji** agar terdeteksi. Perbaikan 1–2 poin akurasi **tidak akan pernah**
signifikan pada n=300, berapa pun rapinya pekerjaan Anda.

Dua konsekuensi praktis:

- Jangan mengukur iterasi terlalu sering. Lebih baik 2–3 iterasi dengan koreksi
  banyak daripada 6 iterasi dengan koreksi sedikit yang semuanya "tidak
  signifikan".
- Kalau hasilnya memang tidak signifikan, tulis apa adanya **beserta tabel di
  atas**. "Perbaikan 2,3 poin, tidak signifikan (p = 0,09); dengan n = 300 uji
  ini memang hanya mampu mendeteksi perbaikan bersih ≥ 6 baris" jauh lebih kuat
  daripada sekadar melaporkan p.

### 2.4 Pengujian berganda bila memakai `--compare-last`

`evaluation:report` benar: satu perbandingan awal-vs-akhir. Tetapi
`evaluation:run --compare-last` menguji tiap iterasi terhadap iterasi
sebelumnya. Bila hasil dari beberapa iterasi ikut dilaporkan sebagai temuan,
itu pengujian berganda — dengan 4 uji pada α = 0,05, peluang setidaknya satu
"signifikan" palsu naik ke sekitar 19%.

**Saran:** tetapkan di muka bahwa **uji utamanya adalah iterasi 0 vs iterasi
terakhir** (yang sudah dilakukan `evaluation:report`), dan sajikan
perbandingan antar-iterasi sebagai **deskriptif saja, tanpa mengklaim
signifikansi**. Tulis kalimat itu di bab metodologi sebelum menjalankan
evaluasinya. Itu praktik pra-registrasi sederhana, dan menutup pertanyaan
penguji tanpa perlu koreksi Bonferroni.

### 2.5 Populasi yang diukur bukan "seluruh korpus"

`GoldenSamplingService::MIN_PANJANG_BAWAAN = 20` membuang teks di bawah 20
karakter, dan duplikat juga dibuang. Keduanya keputusan yang **sah** dan
alasannya masuk akal (mutu anotasi; satu kesalahan tidak dihitung dua kali).
Rencana pengambilan sudah mencatat jumlah yang ditolak per alasan — bagus, itu
yang membuat hal ini bisa dilaporkan.

Yang perlu ditambahkan hanya penafsirannya, dan satu peringatan:

- Sebut populasinya apa adanya: **"akurasi pada komentar unik ≥ 20 karakter"**,
  bukan "akurasi pada korpus". Cantumkan persentase yang tersaring dari
  `plan.rejected` — angkanya sudah tersimpan.
- **Arah biasnya ke atas.** Komentar pendek justru tempat model paling lemah,
  karena konteksnya sedikit: terukur di layanan ini, `"Pajak naik"` terbaca
  positif pada 0,53 sementara `"Pajak naik terus"` terbaca negatif pada 0,73 —
  itu sebabnya `_resolve_low_confidence` ada. Membuang teks terpendek
  menyingkirkan sebagian kasus tersulit, jadi akurasi hold-out sedikit
  **melebih-lebihkan** kinerja pada korpus utuh.

Menyebut ini sendiri di bagian batasan jauh lebih kuat daripada menunggu
ditanya.

### 2.6 Kolom "Kappa" di tabel utama ambigu

Ada **dua** kappa dan keduanya sudah dihitung, tetapi §5 hanya punya satu kolom:

- `metrics.sentiment.cohen_kappa` — kesepakatan **model vs label emas**.
- `metrics.annotation.cohen_kappa` — kesepakatan **antar dua anotator manusia**.

Keduanya harus masuk naskah, dan yang kedua justru yang lebih penting: ia
**batas atas** kinerja yang masuk akal. Bila dua manusia hanya sepakat pada
kappa 0,70, model dengan kappa 0,72 sudah berada di batas kebisingan label, dan
selisih apa pun di atas itu tidak bermakna. Beri dua kolom terpisah dengan nama
yang jelas — mis. "Kappa model" dan "Kappa antar-anotator (n=…)".

### 2.7 Dua hal kecil

- **Simpan sidik jari isi golden dataset**, bukan hanya `item_count` dan
  `frozen_at`. Hash dari seluruh `text_hash` terurut sudah cukup, dicetak di
  bagian Reproduksi. Pembekuan sudah dijaga di aplikasi, tetapi hash membuktikan
  kepada pembaca bahwa iterasi 0 dan iterasi terakhir benar-benar mengukur baris
  yang sama — tanpa perlu mempercayai aplikasinya.
- **Peringatkan di UI koreksi bila baris ada di golden dataset.**
  `RetrainingService::correctedItems()` sudah menyingkirkannya dari pelatihan
  (benar), tetapi pengguna tidak diberi tahu, jadi tenaganya terbuang pada baris
  yang hasilnya dibuang diam-diam. Pada studi kasus dengan active learning nyata
  ini terasa.

---

## Bagian 3 — Yang sudah saya kerjakan di `nlp-api-service`

Akar masalah 2.1 ada di layanan, bukan di Laravel: stemmer Sastrawi hidup di
Python. Memaksa PHP menebak bentuk dasar akan melahirkan definisi kecocokan
kedua yang berbeda dari definisi yang dipakai ekstraksi — persis kesalahan yang
sedang diperbaiki.

**1. `document_aspects_normalized` pada respons `/api/analyze/aspect`.**
Sejajar indeks dan isinya dengan `document_aspects`, berisi bentuk dasar tiap
aspek. Dikirim **berdampingan**, tidak menggantikan: bentuk permukaan tetap yang
ditampilkan ke pengguna, karena `pelayanannya` memang kata yang ditulis penonton.
Duplikat yang muncul setelah di-stem (`pelayanan` + `pelayanannya` dalam satu
komentar) digabung jadi satu.

**2. `POST /api/normalize/aspects`** untuk sisi sebaliknya — label emas.
Mengembalikan bentuk dasar tiap nama aspek memakai stemmer yang sama.
Tidak memuat bobot model apa pun, jadi murah dan tidak ikut mengantre di
belakang retraining.

```json
POST /api/normalize/aspects
{"aspects": ["Pelayanan", "pajaknya"]}

{"status": "success",
 "results": {"normalized": {"Pelayanan": "layan", "pajaknya": "pajak"},
             "aspects": ["layan", "pajak"]}}
```

**15 test baru** (`tests/test_aspect_normalization.py`), suite kini **357 hijau**.

### Yang perlu dikerjakan di sisi Laravel

Kecil, dan hanya di dua tempat:

1. **Saat golden dataset diimpor**, panggil `/api/normalize/aspects` sekali
   untuk seluruh label aspek emas, simpan hasilnya di kolom baru
   (mis. `gold_aspects_normalized`). Datasetnya beku, jadi cukup sekali seumur
   dataset.
2. **Di `HoldoutEvaluationService::predictAspects()`**, baca
   `document_aspects_normalized` (dengan `document_aspects` sebagai cadangan
   bila kunci itu belum ada), lalu bandingkan bentuk dasar dengan bentuk dasar.

Penting: **kedua sisi harus dinormalkan.** Nilai stem-nya sendiri tidak selalu
sama dengan bentuk dasar yang ditulis manusia (`pelayanan` → `layan`), dan itu
tidak masalah — yang wajib benar adalah keduanya mendarat di token yang sama.
Normalisasinya idempoten, jadi menormalkan label yang sudah berbentuk dasar
aman. Ada test yang mengunci kedua sifat itu.

Selama perubahan ini belum ada, **jangan kutip angka aspek apa pun** — bukan
karena modelnya buruk, tetapi karena yang terukur bukan mutu modelnya.

---

## Bagian 4 — Urutan yang saya sarankan

1. **2.1** — perbaiki metrik aspek. Tanpa ini angka aspek di naskah salah telak,
   dan sisi layanannya sudah siap.
2. **2.2** — pindahkan pengukuran koherensi ke korpus penuh. Sekali ubah, dan
   selisihnya terukur 0,3597 → 0,4498, yang melintasi batas "lemah"/"baik"
   pada rambu penafsiran sistem sendiri.
3. **2.3 dan 2.4** — tulis di bab metodologi **sebelum** studi kasus jalan:
   uji utama adalah iterasi 0 vs terakhir, dan perbaikan bersih < 6 baris tidak
   akan terdeteksi. Keduanya tidak butuh perubahan kode.
4. **2.5 dan 2.6** — dua paragraf di bagian batasan, dan satu kolom tambahan di
   tabel.
5. **2.7** — kalau sempat.

Urutan langkah studi kasus di §4 panduan sudah benar dan mengikat; jangan
diubah. Yang saya tambahkan hanya: **kerjakan nomor 1 dan 2 di atas sebelum
langkah 6 (Iterasi 0)**, karena keduanya mengubah cara angka dihitung — bukan
modelnya. Mengubahnya setelah iterasi 0 terukur membuat iterasi 0 dan iterasi 1
mengukur hal yang berbeda, yaitu persis kesalahan kedua pada tabel §6.
