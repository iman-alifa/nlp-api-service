# Konteks proyek untuk sesi Claude (cloud / sesi baru)

Dokumen serah-terima. Ringkasannya diambil dari sesi pengembangan lokal yang
panjang (hingga 7 Oktober 2026) supaya sesi baru tidak mulai dari nol.
Baca bersama `CLAUDE.md` di akar repositori, yang memuat keputusan teknis
terperinci dan alasannya.

## Apa ini

Layanan analisis teks bahasa Indonesia (FastAPI): klasifikasi sentimen,
ekstraksi aspek, pemodelan topik, dan asosiasi aspek–topik. Ini adalah
**separuh backend** dari skripsi "Otomatisasi Analisis Sentimen Berbasis Aspek
dan Pemodelan Topik" (Politeknik Statistika STIS, Iman Alifa Novansyah).
Separuh lainnya, aplikasi web Laravel `text-analysis-web`, ada di repositori
terpisah dan **tidak ikut di sini**. Rujukan `../text-analysis-web` di
dokumen lama tidak akan ditemukan.

## Keadaan akhir (Oktober 2026)

- Aplikasi web dan layanan ini berjalan di satu VPS:
  `https://text-analyze.ing.biz.id` (nginx, HTTPS, layanan di `127.0.0.1:8001`).
  Sudah dipakai pengguna asli pada 24 September 2026.
- **Model yang dipakai:** sentimen = hasil pelatihan ulang iterasi 8
  (IndoBERT `crypter70`), aspek = hasil iterasi 7. Aspek iterasi 8 sempat lebih
  buruk secara signifikan, sehingga bobot aspek dikembalikan ke iterasi 7.
- Seluruh pengujian otomatis layanan ini lulus: 483 kasus (`python -m pytest tests/ -q`).
- Tidak ada model atau data pengguna di repositori. Bobot ada di Hugging Face
  (repositori privat) dan di server.

## Hasil utama untuk skripsi

Angka lengkap dan sumbernya: `docs/skripsi/JAWABAN_DATA_NLP.md` dan
`docs/skripsi/BAHAN_PEMBAHASAN_SKRIPSI.md`. Ringkasnya:

- **Fase 1** (enam putaran koreksi studi kasus, golden dataset 1, n = 300):
  F1 ekstraksi aspek 51,84 → 74,38 (+22,54 poin, CI 95% [+18,33; +27,03],
  bootstrap berpasangan 10.000 resampel seed 42), signifikan. Sentimen naik
  +2,67 poin tetapi tidak signifikan (McNemar p = 0,322).
- **Fase 2** (pengguna asli, golden dataset 2, n = 300, κ antar-anotator 0,698):
  sentimen macro-F1 75,40 → 78,38; aspek iterasi 8 turun signifikan
  (F1 76,45 → 73,21, CI [−4,96; −1,57]). Penyebab utama: ketidakstabilan
  pelatihan ulang pada kolam kecil, bukan salah anotasi.
- **Bias jangkar:** anotator 1 melihat tebakan model setelah memutuskan;
  44 label berubah, 42 di antaranya ke arah model. Ini keterbatasan yang harus
  dinyatakan.

## Keputusan desain yang sering disalahpahami

- Pelatihan ulang selalu dari **bobot dasar** atas seluruh kolam koreksi
  (protokol kolam), bukan melanjutkan checkpoint. Penjaga penerimaan
  membandingkan dengan bobot dasar, bukan iterasi sebelumnya.
- Evaluasi memprediksi ulang teks golden dengan model yang sedang aktif;
  tidak membaca prediksi yang tersimpan.
- Golden dataset tidak boleh bocor ke pelatihan: baris yang koreksinya sudah
  dilatihkan tidak diambil, dan baris golden otomatis disingkirkan dari
  muatan pelatihan.
- Layanan harus berjalan dengan `--workers 1` (keadaan model ada di memori
  proses).
- Memori dilepas setelah tiap pelatihan (`lepas_memori`); tanpa itu proses
  tertahan di puncak pelatihan.

## Yang TIDAK bisa dilakukan dari sesi cloud

- Tidak ada akses ke VPS, basis data, `.env`, bobot model, atau kunci API.
- Pengujian yang memuat bobot IndoBERT akan dilewati atau gagal bila bobot
  tidak diunduh. Sisanya berjalan tanpa jaringan.
- Jangan pernah menaruh rahasia, bobot model, atau data pengguna asli di
  repositori ini.

## Konvensi

- Komentar dan docstring campuran Bahasa Indonesia dan Inggris; teks yang
  tampil ke pengguna berbahasa Indonesia. Ikuti gaya yang sudah ada.
- Setiap perubahan perilaku disertai tes perilaku (bukan tes atas teks
  sumber), dan tes baru sebaiknya dibuktikan gagal tanpa perbaikannya.
- Catatan keterbatasan dalam dokumen skripsi harus jujur; jangan membulatkan
  hasil yang tidak signifikan menjadi klaim.
