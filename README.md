# NLP Analysis API

Layanan FastAPI untuk analisis teks berbahasa Indonesia: klasifikasi sentimen,
ekstraksi aspek, pemodelan topik, dan asosiasi aspek–topik. Menjadi backend NLP
bagi aplikasi Laravel `text-analysis-web`, yang memanggilnya lewat HTTP.

## Menjalankan secara lokal

```bash
python -m venv venv && ./venv/Scripts/Activate.ps1   # Windows
pip install -r requirements.txt
cp .env.example .env          # APP_ENV=development, API_KEY boleh kosong
python run.py
curl http://localhost:8001/health
```

Tes (`pip install -r requirements-dev.txt` lebih dulu):

```bash
python -m pytest tests/ -q
```

Suite tes sengaja tidak memuat bobot model, jadi ia berjalan tanpa jaringan.

## Deployment

**Dua hal wajib sebelum URL dibagikan ke publik:**

1. **`API_KEY` diisi.** Layanan menolak start bila `APP_ENV=production` dan
   kunci ini kosong. Tanpa kunci, `/api/retrain/*` terbuka untuk siapa pun yang
   tahu URL-nya — dan endpoint itu menimpa bobot model.
2. **`MODEL_PATH` menunjuk ke penyimpanan permanen**, atau `HF_WEIGHTS_REPO`
   diisi. `models/` tidak ikut image; tanpa salah satunya, hasil retraining
   hilang pada redeploy berikutnya dan layanan diam-diam kembali ke bobot
   publik.

Rencana lengkap, termasuk anggaran memori dan pemisahan peran melayani/melatih
untuk banyak pengguna, ada di `docs/RENCANA_DEPLOY.md` (tidak ikut ke image;
lihat repo).

### Yang wajib diperhatikan

- **`--workers 1` bersifat arsitektural.** Keadaan model — termasuk bobot hasil
  retraining — hidup di global tingkat proses. Dua worker berarti dua salinan
  bobot dan keadaan yang menyimpang setelah retraining. `Procfile`,
  `nixpacks.toml`, dan `railway.json` semuanya memakukannya.
- **RAM:** ~2,5 GB ketika seluruh bobot termuat, **~4 GB puncak** saat
  retraining (gradien + state optimizer).
- **Bobot dimuat secara malas.** Permintaan pertama tiap jenis membayar biaya
  muat model. Panggil `POST /api/warmup` setelah boot.

## Endpoint

| metode | jalur | kunci API |
|---|---|---|
| GET | `/`, `/health` | tidak |
| POST | `/api/warmup` | ya |
| POST | `/api/preprocess` | ya |
| POST | `/api/normalize/aspects` | ya |
| POST | `/api/analyze/{sentiment,aspect,topic,combined,association}` | ya |
| POST | `/api/retrain/{sentiment,aspect,preview}` | ya |

`/health` melaporkan bobot mana yang sedang hidup (`sentiment_retrained`,
`aspect_retrained`) dan suhu kalibrasi yang berlaku.

## Lisensi model dasar

Bobot layanan ini diturunkan dari `indobenchmark/indobert-base-p1` dan
`crypter70/IndoBERT-Sentiment-Analysis`. Periksa lisensi masing-masing sebelum
mendistribusikan ulang checkpoint hasil retraining — turunan mewarisi
kewajibannya.
