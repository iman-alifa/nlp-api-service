# Ringkasan temuan untuk penyusunan skripsi

Ditulis 11 September 2026, setelah enam iterasi active learning selesai.
Semua angka diukur pada golden dataset yang sama (300 baris, beku, sudah
diperiksa kebocorannya).

---

## 1. Keadaan akhir sistem — terverifikasi

```
sentiment_source     : ./models/sentiment_retrained   sentiment_retrained : true
aspect_checkpoint    : ./models/aspect_retrained.pt   aspect_retrained    : true
sentiment_temperature: 0.9336
```

**Model yang melayani permintaan adalah model hasil retraining, bukan bobot
publik.** Dikonfirmasi dari `/health` dan dari catatan asal-usul:

| | sentimen | aspek |
|---|---|---|
| bobot awal | `crypter70/IndoBERT-Sentiment-Analysis` | `models/aspect_baseline.pt` (K6) |
| `trained_from_baseline` | true | true |
| jumlah sampel kolam | 481 | 398 |
| sidik jari kolam | `53eaa04dcdd99a8b` | `f3679e342ad8f193` |
| set validasi | 96 | 79 |

---

## 2. Hasil enam iterasi

| iterasi | koreksi ditambahkan | kolam sentimen | akurasi sentimen | macro F1 | aspek micro-F1 | exact match |
|---|---|---|---|---|---|---|
| 0 | — (sebelum AL) | 0 | 83,00 | 75,68 | 51,84 | 27,67 |
| 1 | 218 TNI | 218 | 86,33 | 80,25 | 67,86 | 32,33 |
| 2 | +112 Presiden | 330 | 85,67 | 78,80 | 72,86 | 36,67 |
| 3 | +62 Kejaksaan Agung | 392 | **87,33** | **82,51** | 73,14 | 40,00 |
| 4 | +21 KPK | 413 | 85,67 | 79,93 | 73,15 | 37,67 |
| 5 | +22 Polri | 435 | 84,33 | 76,43 | 72,78 | 37,67 |
| 6 | +46 DPR | 481 | 85,67 | 78,39 | **74,38** | **40,67** |

### Uji signifikansi terhadap Iterasi 0

**Aspek — paired bootstrap 5000 resample, micro-F1:**

| | F1 | CI 95% | |
|---|---|---|---|
| It-1 | 67,86 | [+11,94, +20,18] | **signifikan** |
| It-2 | 72,86 | [+16,72, +25,43] | **signifikan** |
| It-3 | 73,14 | [+17,08, +25,60] | **signifikan** |
| It-4 | 73,15 | [+16,87, +25,75] | **signifikan** |
| It-5 | 72,78 | [+16,77, +25,22] | **signifikan** |
| **It-6** | **74,38** | **[+18,16, +26,94]** | **signifikan** |

**Sentimen — McNemar eksak dua sisi:**

| | akurasi | rusak | diperbaiki | p |
|---|---|---|---|---|
| It-1 | 86,33 | 14 | 24 | 0,1433 |
| It-2 | 85,67 | 21 | 29 | 0,3222 |
| It-3 | 87,33 | 14 | 27 | **0,0596** |
| It-4 | 85,67 | 17 | 25 | 0,2800 |
| It-5 | 84,33 | 25 | 29 | 0,6835 |
| It-6 | 85,67 | 21 | 29 | 0,3222 |

### Kesimpulan yang boleh ditulis

**Modul aspek: berhasil, dan signifikan.** 51,84 -> 74,38 micro-F1
(**+22,54 poin**, CI 95% [+18,16, +26,94], p<0,0001). Seluruh enam iterasi
mengungguli baseline secara signifikan.

**Modul sentimen: arah positif tetapi TIDAK signifikan pada iterasi mana pun.**
Puncaknya di Iterasi 3 (87,33; p=0,0596), akhirnya 85,67 (+2,67 poin dari
baseline, p=0,3222). Ini harus ditulis apa adanya. Yang benar: *"active
learning memperbaiki ekstraksi aspek secara signifikan; pada klasifikasi
sentimen arah perbaikan konsisten tetapi tidak terbukti signifikan pada
hold-out 300 baris."*

**Kurva belajarnya jenuh setelah iterasi 2** — perbandingan antar-langkah:

| langkah | sentimen (p) | aspek (CI 95%) |
|---|---|---|
| It-0 -> It-1 | 0,1433 | [+11,94, +20,18] **sig** |
| It-1 -> It-2 | 0,8506 | [+3,53, +6,55] **sig** |
| It-2 -> It-3 | 0,3018 | [-0,92, +1,48] |
| It-3 -> It-4 | 0,2668 | [-1,33, +1,29] |
| It-4 -> It-5 | 0,4545 | [-1,52, +0,77] |
| It-5 -> It-6 | 0,2891 | [+0,37, +2,90] **sig** |

Kejenuhan setelah ~330 koreksi adalah temuan yang layak dilaporkan, bukan
kegagalan: itu perilaku kurva belajar active learning yang normal, dan berguna
secara praktis (memberi tahu kapan berhenti menganotasi).

---

## 3. Bukti bahwa ini benar-benar active learning

Rinciannya di `TINJAUAN_KONSEP_ACTIVE_LEARNING.md`. Ringkasnya:

**Strategi kueri = least-confidence uncertainty sampling** di atas posterior
terkalibrasi. Terukur pada 330 koreksi pertama:

| | dikoreksi | kolam |
|---|---|---|
| median keyakinan | 0,575 / 0,592 | 0,949 |
| di bawah ambang 0,94 | **100%** | 41,4% / 39,7% |

**Strategi itu terbukti menemukan kesalahan**: 49,7% baris terpilih ternyata
berlabel salah, terhadap laju kesalahan dasar 17% — konsentrasi **2,9x**, dan
monoton terhadap keyakinan (57,1% pada pita 0,00-0,50; 48,2% pada 0,50-0,70).
Untuk aspek, 58,5% daftar aspeknya berubah.

**Redundansi batch diperiksa dan bukan masalah**: hanya 1 kelompok duplikat
persis dari 330, dan teks pendek tidak kelebihan terwakili (<=3 kata: 20,0%
pada koreksi vs 21,5% pada kolam).

**Protokol pelatihan**: dari bobot dasar tetap atas seluruh kolam, seed 42.
Terbukti deterministik — pelatihan #6 mengulang #5 dan menghasilkan
`delta 0,145`, `val 66`, `T 2,3238` yang identik.

---

## 4. Keterbatasan yang HARUS disebut di naskah

1. **Tidak ada pembanding pasif (acak).** Kurva yang naik membuktikan "lebih
   banyak label membantu", bukan "pemilihan aktif membantu". Klaim yang boleh
   ditulis terbatas pada yang pertama.
2. **Modul aspek tidak punya sinyal ketidakpastiannya sendiri.**
   `confidence_score` adalah keyakinan sentimen; koreksi aspek menumpang baris
   yang dipilih karena sentimennya ambigu (Settles & Craven 2008 menyarankan
   ketidakpastian tingkat sekuens).
3. **Sumbu iterasi mencampur dua variabel** — tiap putaran menambah satu
   lembaga, jadi "iterasi" berarti *+label* DAN *+domain*. Sertakan tabel per
   segmen supaya keduanya terlihat.
4. **Titik awalnya bukan model kosong** — ini active learning untuk *adaptasi
   domain*, bukan AL dari nol.
5. **Suhu kalibrasi tidak stabil antar-iterasi** — lihat §5.
6. **8 dari 329 sampel aspek dilatihkan sebagai "tanpa aspek"** karena seluruh
   anotasinya gagal dicocokkan dengan teksnya (4,82% anotasi tak cocok:
   campuran salah ketik anotator, salah ketik teks sumber, dan bentuk turunan
   yang sengaja ditolak penjaga prefiks).
7. **Golden dataset 300 baris tidak punya daya uji yang cukup** untuk selisih
   sentimen sebesar 2-4 poin. Ini penyebab langsung temuan "tidak signifikan"
   di atas, bukan bukti bahwa modelnya tidak membaik.

---

## 5. Yang perlu diperhatikan: suhu kalibrasi berayun liar

| iterasi | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| set validasi | 44 | 66 | 79 | 83 | 87 | 96 |
| **T** | 1,0 | 2,3238 | 1,6729 | **0,7617** | **4,1704** | 0,9336 |

T<1 **mempertajam** model yang sudah terlalu percaya diri; T=4,17 meratakannya
habis-habisan. Keduanya lolos penjaga [0,5; 5,0], tetapi keduanya dipasang dari
80-96 sampel — Guo dkk. (2017) memakai ribuan.

Konsekuensinya nyata: suhu menentukan ukuran antrean tinjauan, dan antrean itu
yang memberi makan iterasi berikutnya. Antrean yang mengembang-mengempis tanpa
alasan substantif adalah salah satu penjelasan kenapa kurva sentimen mendatar
setelah iterasi 3.

**Untuk naskah:** sebutkan sebagai keterbatasan. **Untuk deployment:** perlu
penjaga stabilitas — lihat §6.

---

## 6. Deployment: biaya pelatihan ulang dan cara membatasinya

### Masalahnya nyata dan terukur

Waktu pelatihan tumbuh linear terhadap ukuran kolam (CPU, tanpa GPU):

| modul | detik/sampel | n=481 | n=5.000 | n=50.000 |
|---|---|---|---|---|
| sentimen | ~1,7 | 13,5 menit | ~2,4 jam | ~24 jam |
| aspek | ~1,1 | 7 menit | ~1,5 jam | ~15 jam |

Satu putaran penuh sekarang **~20,5 menit**. Pada 50.000 koreksi menjadi
**~39 jam**. Melatih seluruh kolam selamanya memang tidak berkelanjutan.

### Usulan "bekukan di 500 lalu latih data baru saja" — inti idenya benar, tapi bentuknya berisiko

Yang benar dari usulan itu: biaya harus dibatasi, dan pada suatu titik model
saat ini layak jadi titik berangkat baru.

Yang berisiko: melatih **hanya** data baru menyebabkan **catastrophic
forgetting** (McCloskey & Cohen 1989; Kirkpatrick dkk. 2017). Bukti langsung
ada di studi ini — pada Iterasi 2, akurasi segmen TNI turun **5,2 poin**
semata-mata karena porsi TNI dalam kolam turun dari 100% ke 66%. Datanya
**masih ada**, hanya proporsinya mengecil. Kalau dibuang seluruhnya,
penurunannya jauh lebih besar.

Kerugian lain: hasil putaran jadi bergantung pada urutan pelatihan, sehingga
satu titik tidak bisa dihitung ulang tanpa mengulang seluruh rangkaian —
persis sifat yang kemarin membuat satu iterasi harus dibuang.

### Yang disarankan: batasi ukuran latih dengan replay, bukan dengan membekukan

```
ambang N (mis. 2.000 sampel)

kolam <= N  -> perilaku sekarang: seluruh kolam, dari bobot dasar
kolam >  N  -> himpunan latih = SELURUH koreksi baru sejak putaran terakhir
                              + sampel berstrata dari koreksi lama
                              hingga total N, tetap dari bobot dasar
```

Stratifikasi menurut **label** dan **analisis asal**, supaya tiap lembaga tetap
terwakili. Ini metode *rehearsal/replay* baku pada continual learning, dan ia
mempertahankan ketiganya sekaligus:

| | bekukan + data baru saja | replay berbatas |
|---|---|---|
| biaya terbatas | ya | **ya** |
| tahan catastrophic forgetting | tidak | **ya** |
| hasil dapat direproduksi | tidak | **ya** (seed + sidik jari kolam) |

Sidik jari kolam yang sudah dicatat di `provenance.json` cukup untuk membuat
subset terpilih itu dapat diverifikasi ulang.

### Kalau tetap ingin membekukan bobot

Ada bentuk yang sah: **re-baselining berkala** — promosikan model sekarang
menjadi `aspect_baseline.pt` / bobot dasar sentimen yang baru, **dan tetap
gunakan replay**. Yang harus dihindari adalah re-baselining *tanpa* replay.
Kalau ditempuh, catat di `provenance.json` bahwa baseline berpindah, karena
sejak titik itu angka "dari bobot dasar" berarti hal yang berbeda.

### Catatan praktis untuk deployment

- **Laju pertumbuhan kolam dibatasi manusia.** Koreksi datang dari anotator;
  pada 100 koreksi/minggu, kolam mencapai 5.000 dalam setahun. Pelatihan penuh
  bulanan pada n=2.000 memakan ~1 jam — masih wajar. Jadwal lebih menentukan
  daripada biaya per putaran.
- **GPU mengubah seluruh perhitungan** (20-50x). Angka di atas CPU, sesuai
  Railway free tier.
- **Suhu perlu penjaga stabilitas.** Sebelum dipakai produksi, batasi
  perubahan T antar-putaran (mis. tolak fit yang berbeda >50% dari T
  sebelumnya, atau tarik ke arah T lama secara proporsional terhadap ukuran
  set validasi). Ayunan 0,76 -> 4,17 tidak boleh sampai ke pengguna.
- **Ambang kelayakan sudah ada** (50 sampel kolam) dan sudah terbukti berguna.
