import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class Settings(BaseSettings):
    # `model_path` dkk. bentrok dengan protected namespace "model_" milik pydantic v2.
    # Namespace dikosongkan agar penamaan setting tetap deskriptif tanpa UserWarning.
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
        protected_namespaces=(),
    )

    # Application
    app_name: str = "NLP Analysis API"
    app_version: str = "1.0.0"
    app_env: str = "development"
    debug: bool = False
    # Bawaan 127.0.0.1: hanya menerima koneksi dari mesin yang sama. Di VPS
    # tunggal, 0.0.0.0 berarti port 8001 terbuka ke internet begitu firewall
    # lupa dipasang. Railway, HF Spaces, dan Dockerfile memberi `--host 0.0.0.0`
    # sendiri di perintah uvicorn-nya, jadi bawaan ini hanya berlaku untuk
    # `python run.py` - dan .env lokal yang menyetel HOST tetap menang.
    host: str = "127.0.0.1"
    port: int = int(os.getenv("PORT", "8001"))

    # ── Analisis saat pelatihan berjalan ─────────────────────────────────────
    # True: analisis yang butuh model yang sedang dilatih langsung dibalas 503
    # + Retry-After, alih-alih menunggu lock. Lihat `_tolak_bila_sedang_dilatih`
    # di main.py. Nyalakan hanya bila pemanggil mengantrekan ulang berdasarkan
    # Retry-After; tanpa itu analisis yang tadinya berhasil menunggu justru gagal.
    retrain_reject_inference: bool = False

    # Detik per sampel kolam untuk satu putaran penuh (3 epoch + evaluasi),
    # dipakai memperkirakan Retry-After. Terukur di Ryzen 7 5700U (8 core fisik,
    # CPU saja): sentimen 481 sampel 13,5 menit, aspek 398 sampel 7 menit.
    # Ukur ulang di mesin produksi - vCPU VPS bersama bisa 2-3x lebih lambat.
    retrain_sec_per_sample_sentiment: float = 1.7
    retrain_sec_per_sample_aspect: float = 1.1

    # ── Penyimpanan bobot di Hugging Face Hub ───────────────────────────────
    # Repo tujuan, mis. "namauser/skripsi-nlp-weights". Kosong berarti
    # sinkronisasi MATI dan bobot hanya hidup di disk lokal - sah bila ada
    # volume terpasang di model_path, berbahaya bila tidak.
    #
    # Lihat app/weights_store.py: `models/` masuk .gitignore, jadi tanpa salah
    # satu dari keduanya (volume atau Hub) redeploy menghapus hasil retraining.
    hf_weights_repo: str = ""

    # Token tulis Hugging Face. Simpan sebagai secret di platform hosting,
    # jangan di repo.
    hf_token: str = ""

    # ── Keamanan deployment publik ──────────────────────────────────────────
    # Kunci yang harus dikirim pemanggil lewat header `X-API-Key`. Kosong
    # berarti autentikasi MATI, dan itu hanya sah untuk pengembangan lokal:
    # `app/security.py` menolak start ketika APP_ENV=production dan kunci ini
    # masih kosong.
    #
    # Laravel satu-satunya pemanggil sah, jadi kunci bersama sudah memadai -
    # tidak perlu OAuth untuk dua layanan yang saling kenal.
    api_key: str = ""

    # Percayai header `X-Forwarded-For` untuk menentukan IP klien.
    # Nyalakan HANYA di belakang proxy (Railway, HF Spaces, nginx). Header itu
    # ditulis klien: kalau dipercaya tanpa proxy di depan, batas laju bisa
    # dilewati hanya dengan mengarang header.
    trust_proxy: bool = False

    # Batas laju per alamat IP pada endpoint analisis. Bukan pertahanan
    # anti-DDoS - itu tugas lapisan di depannya - melainkan pagar terhadap satu
    # pemanggil yang menghabiskan CPU. Satu analisis gabungan 80 komentar
    # terukur ~18,5 detik di CPU, jadi 30 per menit sudah longgar untuk
    # pemakaian sah lewat Laravel.
    rate_limit_max: int = 30
    rate_limit_window: int = 60

    # Sembunyikan /docs, /redoc, dan /openapi.json ketika produksi. Skema yang
    # terbuka memberi tahu penyerang persis bentuk muatan retraining.
    expose_docs: bool = False

    # CORS - allow Laravel frontend on Railway
    allowed_origins: list = [
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:8001",
    ]

    # ── Model & checkpoint ──────────────────────────────────────────────────
    # model_path adalah root penyimpanan checkpoint hasil retraining. Sebelumnya
    # setting ini ada di .env tapi tidak pernah dibaca: path checkpoint di-hardcode
    # di masing-masing service, sehingga MODEL_PATH tidak berefek apa pun.
    model_path: str = "./models"
    use_gpu: bool = False

    # Model dasar HuggingFace (dipakai bila belum ada checkpoint hasil retraining).
    # Kandidat yang sudah diverifikasi ada di SENTIMENT_MODELS di bawah; ganti
    # nilai ini (atau SENTIMENT_BASE_MODEL di .env) untuk berpindah, dan suhu
    # kalibrasi serta ambang peninjauannya ikut menyesuaikan otomatis.
    sentiment_base_model: str = "crypter70/IndoBERT-Sentiment-Analysis"
    aspect_base_model: str = "indobenchmark/indobert-base-p1"

    # Panjang token saat INFERENSI aspek. Pelatihan tetap 128 karena padding
    # label BIO mengasumsikannya; inferensi memakai dynamic padding sehingga
    # batas lebih tinggi hanya berbiaya pada teks panjang yang jarang muncul
    # (diukur: 0,3% komentar korpus produksi melebihi 128 token).
    aspect_inference_max_length: int = 256

    # Sentimen per-aspek dihitung dari klausa. Klausa yang sangat pendek bisa
    # kehilangan konteks: IndoBERT menilai "Pajak naik" positive dengan
    # keyakinan 0,53 saja, sedangkan "Pajak naik terus" negative 0,73. Bila
    # keyakinan di bawah ambang ini, prediksi teks penuh dipakai sebagai gantinya.
    aspect_clause_min_confidence: float = 0.6

    # Gabungkan aspek yang hanya berbeda bentuk infleksi (`gaji` vs `gajinya`,
    # `pelayanan` vs `pelayanannya`). Tanpa ini dasbor menampilkan aspek kembar
    # dengan jumlah terbelah, dan statistik per aspek ikut terbelah - keluhan
    # nyata dari studi kasus TNI. Terukur pada 22 komentar: 24 aspek -> 18.
    #
    # Penggabungan memakai stem Sastrawi DITAMBAH penjaga kelas awalan, karena
    # stem saja meratakan awalan pembentuk pelaku: `petugas` dan `tugas` sama-sama
    # menjadi `tugas` padahal aspeknya berbeda. Label yang ditampilkan adalah
    # bentuk permukaan paling sering, bukan stem-nya.
    #
    # Hanya berlaku pada mode automatic/zero-shot. Mode rule-based tidak pernah
    # digabung: kategorinya dideklarasikan pengguna.
    aspect_merge_variants: bool = True

    # ── Pemetaan aspek semantik (mode rule-based) ───────────────────────────
    # Mode rule-based dulu hanya mencocokkan string, sehingga "antriannya lama"
    # tidak pernah masuk ke aspek "pelayanan" yang dideklarasikan pengguna.
    # Tahap kedua memetakan istilah aspek temuan model ke kategori terdekat.
    aspect_semantic_mapping: bool = True

    # Ambang kemiripan kosinus (setelah centering) untuk menerima pemetaan.
    # Diukur pada 17 kasus berlabel di dua domain memakai representasi span
    # berkonteks: kandidat benar terendah 0,175, kandidat asing tertinggi 0,246.
    # Keduanya TUMPANG TINDIH - tidak ada ambang sempurna, jadi ambang ini
    # sengaja longgar dan penyaringan sesungguhnya bertumpu pada gerbang
    # pertama (ekstraktor harus menemukan istilah aspek di kalimat itu).
    aspect_semantic_min_score: float = 0.15

    # Selisih minimum terhadap kategori peringkat dua. Tanpa ini istilah yang
    # mirip dengan semua kategori akan ditempelkan ke salah satunya sembarangan.
    # Terukur: margin kandidat benar rata-rata 0,122 vs kandidat asing 0,035
    # (domain berbeda-beda) dan 0,069 vs 0,010 (domain bertumpang-tindih).
    aspect_semantic_min_margin: float = 0.01

    # ── Topic modeling ──────────────────────────────────────────────────────
    # Seed tetap agar UMAP/HDBSCAN/LDA menghasilkan topik yang sama pada korpus
    # yang sama - tanpa ini angka coherence di skripsi tidak bisa direproduksi.
    topic_seed: int = 42

    # Jumlah kata yang ditampilkan per topik.
    topic_top_n_words: int = 10

    # BERTopic menyisakan dokumen tak berklaster sebagai -1. Terukur pada 885
    # komentar: 34% dokumen tidak mendapat topik sama sekali, dan dokumen itu
    # juga hilang dari perhitungan PMI aspek-topik. Bila aktif, dokumen outlier
    # ditarik ke topik terdekat memakai c-TF-IDF.
    topic_reduce_outliers: bool = True

    # Ambang keyakinan penarikan outlier. Terlalu rendah berarti memaksa semua
    # dokumen masuk topik dan mengotori topik; ini menyisakan yang benar-benar
    # tidak cocok tetap -1.
    topic_outlier_threshold: float = 0.10

    # MaximalMarginalRelevance memilih kata topik yang saling melengkapi alih
    # alih sinonim yang berulang. Terukur pada 885 komentar, 13 topik
    # (scripts/bench_topic_representation.py):
    #   c-TF-IDF polos          c_v 0,3759
    #   + reduce_frequent_words c_v 0,5160
    #   + MMR diversity 0,3     c_v 0,5477  <- dipakai
    #   + MMR diversity 0,5     c_v 0,5422
    #   KeyBERTInspired         c_v 0,4613 (diversity ikut turun ke 0,77)
    # 0 mematikan MMR sepenuhnya.
    topic_mmr_diversity: float = 0.3

    # Pemilihan jumlah topik otomatis berdasarkan coherence (num_topics=0).
    #
    # Memilih k hanya dengan memaksimalkan C_v TIDAK aman. Terukur pada 885
    # komentar: C_v tertinggi jatuh di k=8 (0,5182), tetapi di sana satu topik
    # menelan 58% dokumen - solusi degenerate yang tampak bagus di metrik.
    # Pilihan berimbang ada di k=18 (C_v 0,5123, topik terbesar 15%). Karena itu
    # pencarian dibatasi: kandidat yang tidak seimbang dibuang lebih dulu, baru
    # C_v tertinggi diambil dari sisanya.
    #
    # Kendalanya RELATIF terhadap k, bukan ambang mutlak. Dengan k topik yang
    # sempurna seimbang, tiap topik memuat 1/k dokumen; ambang mutlak 0,35 akan
    # mustahil dipenuhi saat k=2 karena bagian terkecil yang mungkin justru 0,5.
    # Nilai di bawah berarti "topik terbesar boleh sampai 3,5x bagian seimbang".
    # Terukur: k=18 berada di 2,8x (diterima), k=8 di 4,7x (ditolak).
    topic_auto_max_imbalance: float = 3.5

    # Batas MUTLAK, diperlukan bersama batas relatif di atas. Keduanya sendirian
    # tidak cukup: pada 200 dokumen, k=4 dengan satu topik 82,5% tetap lolos
    # batas relatif (0,825 x 4 = 3,3 <= 3,5), padahal jelas degenerate. Batas
    # mutlak menolaknya, sementara batas relatif tetap diperlukan untuk menolak
    # ketimpangan pada k besar yang nilai mutlaknya masih tampak kecil.
    topic_auto_max_share: float = 0.50

    # Batas bawah pencarian; k di bawah ini hampir selalu degenerate pada korpus
    # komentar pendek.
    topic_auto_min_k: int = 4

    # ── Kosakata label topik ────────────────────────────────────────────────
    # Kata harus muncul minimal sekian kali di korpus untuk boleh menjadi LABEL
    # topik. Berbeda dari topic_min_token_freq di bawah: yang ini TIDAK membuang
    # token dari dokumen dan tidak memindahkan satu dokumen pun antar topik - ia
    # hanya membatasi kata mana yang boleh mewakili sebuah topik.
    #
    # Masalahnya terlihat langsung pada korpus produksi: dari 18 topik, enam
    # berlabel kata yang muncul sekali saja - `chuaaakksss`, `preetttt`,
    # `membaaaanguuuuun`, `indonesiiiiaaaa`, `mahpuuudd`, `gatel`. Itu klaster
    # salah ketik dan kata yang dipanjangkan, bukan topik.
    #
    # Penyebabnya struktural: c-TF-IDF memberi bobot tertinggi pada kata yang
    # JARANG, jadi hapax justru menang di topiknya sendiri, dan MMR memperkuatnya
    # karena kata langka selalu tampak melengkapi. Menambah stopword tidak bisa
    # menyelesaikannya - salah ketik tidak mungkin didaftar satu per satu.
    #
    # 1 = mematikan penyaringan.
    # BAWAANNYA MATI (1). Diukur dan ditolak sebagai default, dengan alasan
    # yang sama sekali berbeda dari niat awalnya.
    #
    # Masalah yang ingin diatasi nyata pada korpus pajak: 14 dari 20 topik
    # berlabel kata yang muncul sekali (`chuaaakksss`, `membaaaanguuuuun`).
    # Tetapi pada korpus studi kasus enam lembaga masalah itu hanya 4-12% kata
    # label, sementara biayanya terukur mahal - dan penempatannya di vectorizer
    # bahkan meruntuhkan klasterisasi (berita n=250: 9 topik -> 2, robustness
    # 14/14 -> 13/14).
    #
    # Pelajaran yang lebih umum: apa pun yang masuk ke vectorizer c-TF-IDF
    # BUKAN sekadar soal label. BERTopic memakainya untuk menggabungkan topik
    # dan menarik outlier, jadi mempersempit kosakata di sana menggeser
    # penugasan dokumen. Naikkan hanya bila label salah ketik benar-benar
    # mendominasi, dan ukur ulang robustness sesudahnya.
    topic_label_min_freq: int = 1

    # Penyaringan dimatikan di bawah jumlah dokumen ini: pada korpus kecil,
    # hampir seluruh kosakata memang hapax, dan ambang berapa pun akan membuang
    # kata terpentingnya. Alasan yang sama dengan min_df pada _build_vectorizer.
    topic_label_min_docs: int = 50

    # Bila kosakata yang lolos lebih sedikit dari ini, penyaringan dibatalkan:
    # label dari kosakata yang terlalu tipis lebih buruk daripada label yang
    # tidak disaring sama sekali.
    topic_label_min_vocab: int = 50

    # Token yang muncul kurang dari sekian kali di korpus dibuang SEBELUM
    # pemodelan topik. 1 = tanpa penyaringan (default).
    #
    # DIUKUR DAN DITOLAK SEBAGAI DEFAULT. Masalah yang ingin diatasi nyata:
    # pada 885 komentar, 63,4% kosakata hanya muncul sekali (1 523 dari 2 403
    # tipe), dan karena c-TF-IDF menghargai kata langka, salah ketik bisa naik
    # menjadi label topik - satu kali jalan menghasilkan topik berisi
    # "mahakuasah, majak, kendaran, rakya", yaitu klaster salah ketik.
    #
    # Tetapi penyaringannya merugikan, dan semakin kecil korpus semakin parah
    # (scripts/eval_topics.py, korpus YouTube, num_topics=0):
    #
    #   n dokumen   c_v tanpa filter   c_v dengan filter   token terbuang
    #   60          0,4884             0,3228              30,9%
    #   120         0,4602             0,3940              27,3%
    #   250         0,4972             0,4244              21,7%
    #   885         0,5139             0,5205              15,6%
    #
    # Hanya pada 885 dokumen ia sedikit menolong, dan selisih +0,0066 itu jauh
    # di bawah varians antar-seed (±0,0221) - jadi bukan perbaikan yang nyata.
    # Pada korpus kecil ia membuang separuh token (50% pada 20 dokumen) dan
    # merusak embedding sekaligus label topik. Diversity juga turun di semua
    # ukuran (mis. 0,930 -> 0,686 pada n=120).
    #
    # Fungsinya dipertahankan (TextCleaner.filter_rare_tokens) untuk korpus yang
    # jauh lebih besar, tetapi TIDAK diaktifkan secara default.
    topic_min_token_freq: int = 1

    # ── Jaring pengaman klasterisasi ────────────────────────────────────────
    # HDBSCAN berbasis KEPADATAN, dan asumsi itu runtuh pada korpus kecil atau
    # ruang embedding yang tidak terpisah rapi: ia mengembalikan 2-4 klaster
    # dengan satu klaster menelan sebagian besar korpus. Terukur pada uji
    # ketahanan 3 domain x 5 ukuran (scripts/robustness_topics.py), 5 dari 14
    # kombinasi degenerate - youtube n=60 (1 topik 88%), berita n=120 (64%),
    # ulasan n=250 (69%).
    #
    # Bila hasil HDBSCAN memenuhi salah satu tanda di bawah, klasterisasi
    # diulang memakai KMeans, yang menjamin k klaster berukuran seimbang.
    topic_degenerate_min_clusters: int = 3
    topic_degenerate_max_share: float = 0.50

    # Lantai jumlah klaster yang IKUT TUMBUH bersama ukuran korpus.
    #
    # Batas mutlak 3 di atas tidak cukup. Terukur pada studi kasus enam lembaga:
    # korpus Kejaksaan Agung berisi 1 493 komentar tetapi HDBSCAN hanya
    # menemukan 3 klaster - dan itu LOLOS kedua penjaga lama (tepat 3, bukan
    # kurang dari 3; klaster terbesar 40,4%, bukan di atas 50%). Bandingkan DPR
    # yang menghasilkan 15 topik hanya dari 778 komentar.
    #
    # Akibatnya bukan sekadar kasar. Mode otomatis hanya mencari k ke BAWAH
    # (reduce_topics memang cuma bisa mengurangi), jadi begitu HDBSCAN
    # under-cluster sejak awal, tidak ada jalan naik dan hasilnya terkunci.
    # Dengan lantai ini, hasil semacam itu dinyatakan degenerate sehingga
    # jaring pengaman KMeans - yang menerima k sebagai masukan - ikut dicoba,
    # dan hasilnya hanya dipakai bila memang tidak degenerate.
    #
    # Satu klaster per sekian dokumen; dibatasi topic_degenerate_max_floor agar
    # korpus besar tidak dipaksa memecah topik yang memang tidak ada.
    topic_degenerate_docs_per_cluster: int = 250
    topic_degenerate_max_floor: int = 8

    # ── Sentimen: kalibrasi confidence ──────────────────────────────────────
    # Klasifikasi berbasis transformer terkenal terlalu yakin (Guo, Pleiss, Sun
    # & Weinberger, 2017). Terukur pada model ini: rata-rata confidence 0,9568
    # sementara akurasinya 0,9080 - selisih +0,0381, dan pita 0,7-0,8 bahkan
    # menyatakan 0,75 pada kasus yang benar hanya 0,29.
    #
    # Temperature scaling membagi logit dengan T sebelum softmax. Karena
    # pembagian skalar positif TIDAK mengubah urutan logit, argmax tetap sama
    # persis: accuracy/F1/confusion matrix tidak berubah sedikit pun, hanya
    # keyakinannya menjadi jujur. Itu yang membuatnya aman dipakai.
    #
    # T dicari dengan LBFGS atas NLL pada SmSA valid (1 260 dokumen), lalu
    # diuji pada SmSA test yang tidak dipakai mencarinya:
    #
    #   set     T        ECE      accuracy
    #   valid   1,0000   0,0360   0,9310
    #   valid   1,3057   0,0249   0,9310   <- ECE turun 31%
    #   test    1,0000   0,0489   0,9080
    #   test    1,3057   0,0367   0,9080   <- ECE turun 25%, akurasi identik
    #
    # Ini penting bukan demi angka: `review_queue` memilih baris untuk dikoreksi
    # manusia berdasarkan confidence, sehingga confidence yang tidak jujur
    # membuat antrean koreksi salah sasaran.
    #
    # Setelah retraining, T dicari ulang pada set validasi retraining dan
    # disimpan bersama checkpoint (calibration.json). 1.0 = tanpa kalibrasi.
    sentiment_temperature: float = 1.3057

    # ── Sentimen: ambang peninjauan manusia ─────────────────────────────────
    # Confidence IndoBERT ternyata informatif, bukan sekadar angka hiasan.
    # Terukur pada SmSA test (500 dokumen, scripts/evaluate_sentiment.py):
    #
    #   confidence      porsi   akurasi
    #   >= 0,99         65,2%    0,9847
    #   0,90 - 0,99     22,4%    0,8661
    #   0,70 - 0,90      7,4%    0,7027
    #   < 0,70           5,0%    0,4340
    #
    # Artinya ~12% prediksi dengan confidence di bawah 0,90 memuat sebagian
    # besar kesalahan. Menandainya untuk koreksi manusia adalah cara termurah
    # mengisi data retraining (active learning), dibanding meninjau acak.
    #
    # Angka di bawah ini BUKAN yang berlaku. Ambang bersifat per bobot dan
    # ditimpa dari SENTIMENT_MODELS di akhir berkas ini; untuk `crypter70`
    # (model aktif) nilainya 0,94. Baca nilai efektifnya dari /health, jangan
    # dari baris ini.
    #
    # Pilihan 0,94 diukur, bukan ditebak - berapa porsi SELURUH kesalahan yang
    # tertangkap bila semua baris di bawah ambang ditinjau (SmSA test, n=500,
    # model crypter70):
    #
    #   ambang   ditinjau   porsi korpus   error tertangkap   lift vs acak
    #   0,80        26 baris     5,2%         29,5%              5,68
    #   0,90        45 baris     9,0%         47,7%              5,30
    #   0,94        86 baris    17,2%         63,6%              3,70
    #   0,95       137 baris    27,4%         79,5%              2,90
    #   0,96       377 baris    75,4%         97,7%              1,30
    #
    # 0,94 dipilih sebagai titik sebelum lift runtuh: meninjau 17% korpus untuk
    # menangkap 64% kesalahan. Di 0,95 biaya tinjauan naik 60% untuk tambahan
    # 16 poin, dan di 0,96 praktis seluruh korpus harus ditinjau.
    #
    # `text-analysis-web` memakai 0,80 lewat ANALYSIS_REVIEW_THRESHOLD. Itu
    # pilihan yang sah - beban pengguna jauh lebih ringan - tetapi harganya
    # terbaca di tabel: 29,5% kesalahan tertangkap, bukan 63,6%. Sebutkan
    # angka itu bila 0,80 dipakai di naskah. Sisi web menyaring hasil yang
    # SUDAH dibangun pada 0,94, jadi 0,80 menghasilkan himpunan bagian yang
    # konsisten - bukan permintaan ulang ke layanan.
    #
    # Ambang ini per bobot dan TIDAK boleh disalin antar model: 0,90 adalah
    # titik kerja setara pada `mdhugol` (18,0% -> 67,4%), tetapi pada
    # `crypter70` hanya menangkap 47,7% karena model ini lebih percaya diri
    # sehingga lebih sedikit baris yang jatuh di bawahnya.
    sentiment_review_threshold: float = 0.90

    # ── Protokol active learning ────────────────────────────────────────────
    # Setiap putaran dilatih dari BOBOT DASAR atas seluruh kolam koreksi yang
    # sudah terkumpul, bukan melanjutkan dari checkpoint putaran sebelumnya.
    #
    # Ini protokol active learning yang baku (Settles 2009): kolam berlabel L
    # bertambah tiap putaran, dan model dilatih ulang pada L. Melanjutkan dari
    # checkpoint sebelumnya SAMBIL melatih pada kolam yang sama membuat koreksi
    # lama ikut terlatih berkali-kali - koreksi putaran 1 akan dilihat model
    # dua kali pada putaran 2, tiga kali pada putaran 3 - sehingga bobotnya
    # membengkak tanpa alasan.
    #
    # Yang lebih penting untuk skripsi: dari-dasar membuat kurva iterasi bisa
    # ditafsirkan. Hasil putaran N hanya bergantung pada ISI kolam saat itu,
    # bukan pada urutan pelatihan yang mendahuluinya, sehingga siapa pun bisa
    # menghitung ulang satu titik tanpa mengulang seluruh rangkaian.
    #
    # Matikan hanya bila biaya komputasi jadi penghalang; pada korpus ini satu
    # putaran 218 sampel memakan ~3,7 menit di CPU.
    retrain_from_baseline: bool = True

    # Bobot dasar modul aspek untuk protokol di atas. Berbeda dari
    # `aspect_base_model` (IndoBERT mentah): titik awal yang benar adalah
    # checkpoint SEBELUM active learning dimulai - pada proyek ini K6, yang
    # F1-nya 0,92 pada label manusia sementara IndoBERT mentah hanya 0,17-0,35.
    # Memakai IndoBERT mentah sebagai dasar akan membuat "sebelum active
    # learning" berarti sistem yang tidak pernah dipakai siapa pun.
    #
    # Kosong berarti jatuh ke perilaku lama (melanjutkan dari checkpoint), dan
    # itu dicatat sebagai peringatan.
    aspect_baseline_checkpoint: str = ""

    # Jumlah minimum sampel validasi agar suhu kalibrasi boleh diukur ulang
    # setelah retraining. Di bawah ini suhu lama DIWARISI, bukan direset ke 1.0.
    #
    # Guo dkk. (2017) memakai ribuan sampel; 50 sudah longgar. Yang penting
    # bukan angkanya melainkan perilakunya saat tidak terpenuhi: retrain per
    # analisis pada studi kasus ini menghasilkan set validasi 4-44 baris, jadi
    # penjaga ini akan menyala hampir selalu - dan mereset ke 1.0 setiap kali
    # akan merusak antrean tinjauan yang memberi makan iterasi berikutnya.
    calibration_min_samples: int = 50

    # Nama artefak di dalam model_path
    sentiment_checkpoint_name: str = "sentiment_retrained"
    aspect_checkpoint_name: str = "aspect_retrained.pt"

    # ── Retraining ──────────────────────────────────────────────────────────
    # Seed tetap agar pembagian train/val dan inisialisasi head bisa direproduksi;
    # tanpa ini angka pada laporan skripsi berubah setiap kali training diulang.
    retrain_seed: int = 42
    retrain_val_ratio: float = 0.2
    # Checkpoint baru ditolak bila metrik utamanya turun lebih dari toleransi ini
    # dibanding model sebelumnya (fase Evaluation CRISP-ML(Q)).
    retrain_min_delta: float = 0.0

    # HuggingFace cache directory (use persistent storage on Railway)
    # Kunci model dasar ke revisi (commit) tertentu di Hugging Face Hub. Lihat
    # HF_MODEL_REVISIONS di bawah. Matikan hanya untuk sengaja menarik versi
    # terbaru - dan catat bahwa hasilnya tidak lagi sebanding dengan skripsi.
    hf_pin_revisions: bool = True

    hf_home: str = os.getenv("HF_HOME", "./.cache/huggingface")
    # Kosong = ikut `<hf_home>/hub`, lokasi yang dipakai huggingface_hub dan
    # `hf download`. Dulu bawaannya `./.cache/huggingface` (akar, bukan `hub/`):
    # transformers menyimpan dan mencari model di akar sementara `hf download`
    # di `hub/`, sehingga model dasar yang sengaja diunduh lebih dulu dengan
    # revisi terkunci TIDAK PERNAH terbaca - terukur di VPS produksi, pemuatan
    # tanpa internet gagal padahal berkasnya ada. Isi hanya untuk memaksa
    # lokasi lain.
    transformers_cache: str = os.getenv("TRANSFORMERS_CACHE", "")

    # ── Batas permintaan ────────────────────────────────────────────────────
    # Jumlah maksimum teks per permintaan. Dipakai schemas/request_schemas.py
    # sebagai batas validasi, sehingga permintaan yang terlalu besar ditolak
    # dengan pesan jelas alih-alih membuat service kehabisan memori di tengah
    # jalan. Sebelumnya nilai ini tidak pernah dibaca dan skema memakai angka
    # 10000 yang ditulis langsung.
    max_batch_size: int = 10000

    # Panjang maksimum satu teks (karakter). Teks yang melebihi batas TOKEN
    # model tetap dipotong dan dilaporkan lewat metrics.total_truncated; batas
    # ini menjaga hal yang berbeda, yaitu ukuran payload.
    max_text_length: int = 10000

    # CATATAN: `request_timeout` dihapus. Ia tidak pernah dibaca kode mana pun,
    # dan timeout permintaan memang milik pemanggil (Laravel: NLP_API_TIMEOUT)
    # serta server HTTP, bukan aplikasi. Menyimpannya sebagai setting membuat
    # seolah-olah service menegakkan batas yang sebenarnya tidak ada.

    # Logging
    log_level: str = "INFO"
    log_file: str = "./logs/api.log"

    # ── Derived paths ───────────────────────────────────────────────────────
    @property
    def sentiment_checkpoint_path(self) -> str:
        """Direktori HuggingFace hasil retraining sentimen."""
        return os.path.join(self.model_path, self.sentiment_checkpoint_name)

    @property
    def aspect_checkpoint_path(self) -> str:
        """File checkpoint torch hasil retraining aspek."""
        return os.path.join(self.model_path, self.aspect_checkpoint_name)


# ── Kandidat model sentimen yang sudah diukur ───────────────────────────────
# Setiap entri diukur lewat JALUR PRODUKSI yang sama (scripts/evaluate_sentiment.py
# dan scripts/behavioral_sentiment.py) pada SmSA valid (1 260) + test (500).
#
# `temperature` dicari terpisah untuk tiap model pada SmSA VALID - suhu adalah
# milik bobot tertentu, bukan konstanta global. `review_threshold` juga per model:
# model yang lebih percaya diri butuh ambang lebih tinggi untuk menangkap porsi
# kesalahan yang sama. Keduanya dipilih dari pengukuran, bukan angka bulat.
#
# Ringkasan (test / valid, setelah kalibrasi masing-masing):
#
#   model         dasar          acc     macroF1   ECE     perilaku  kontras  negasi
#   crypter70     IndoBERT p1    0,9120  0,8855    0,0240  0,952     0,917    0,542
#   mdhugol       IndoBERT p1    0,9080  0,8762    0,0367  0,924     0,417    0,500
#   w11wo         RoBERTa        0,9160  0,8853    0,0209  0,972     0,750    0,917
#   ayameRushia   BERT Cahya     0,9180  0,9010    -       0,956     0,833    0,625
#
# Dipakai: crypter70 - satu-satunya yang mengungguli mdhugol pada SETIAP metrik
# sambil tetap memakai model dasar IndoBERT yang sama (indobenchmark/indobert-base-p1),
# sehingga tidak menyimpang dari IndoBERT yang diajukan pada seminar proposal.
# w11wo unggul pada uji perilaku tetapi BUKAN IndoBERT (RoBERTa), jadi tidak dipakai.
SENTIMENT_MODELS = {
    "crypter70/IndoBERT-Sentiment-Analysis": {
        "base": "indobenchmark/indobert-base-p1",
        "arch": "bert",
        "temperature": 2.7748,
        "review_threshold": 0.94,
        "note": "IndoBERT p1 + IndoNLU SmSA. Kontras terbaik (0,917). DIPAKAI.",
    },
    "mdhugol/indonesia-bert-sentiment-classification": {
        "base": "indobenchmark/indobert-base-p1",
        "arch": "bert",
        "temperature": 1.3057,
        "review_threshold": 0.90,
        "note": "IndoBERT p1 + SmSA. Model awal proyek; kalah di semua metrik.",
    },
    "w11wo/indonesian-roberta-base-sentiment-classifier": {
        "base": "flax-community/indonesian-roberta-base",
        "arch": "roberta",
        "temperature": 1.6132,
        "review_threshold": 0.90,
        "note": "BUKAN IndoBERT. Perilaku terbaik (0,972) & negasi 0,917.",
    },
    "ayameRushia/bert-base-indonesian-1.5G-sentiment-analysis-smsa": {
        "base": "cahya/bert-base-indonesian-1.5G",
        "arch": "bert",
        "temperature": None,  # belum diukur; fit_temperature dipanggil bila dipakai
        "review_threshold": 0.90,
        "note": "BUKAN IndoBERT. macro F1 test tertinggi (0,9010).",
    },
}


@lru_cache()
def get_settings():
    return Settings()

settings = get_settings()

# Suhu dan ambang tinjau mengikuti model yang dipilih, kecuali .env menyebutkan
# nilainya secara eksplisit. Tanpa ini, mengganti model akan memakai kalibrasi
# milik model LAIN - dan confidence yang salah kalibrasi membuat antrean koreksi
# salah sasaran tanpa memunculkan galat apa pun.
_profil = SENTIMENT_MODELS.get(settings.sentiment_base_model, {})
if "SENTIMENT_TEMPERATURE" not in os.environ:
    _suhu = _profil.get("temperature")
    if _suhu:
        settings.sentiment_temperature = _suhu
    elif _profil:
        # Kandidat terdaftar tetapi suhunya belum diukur (mis. `ayameRushia`).
        # Pengecekan truthiness saja akan MELEWATI baris ini dan membiarkan
        # nilai bawaan kelas bertahan - yaitu suhu milik model LAIN, yang
        # diam-diam salah kalibrasi lalu menggeser seluruh antrean tinjauan.
        # Suhu itu milik bobot, bukan milik modul.
        raise ValueError(
            f"SENTIMENT_MODELS['{settings.sentiment_base_model}'] belum punya "
            "temperature hasil pengukuran. Jalankan "
            "`python scripts/evaluate_sentiment.py --model <id> --fit-temperature` "
            "lebih dulu, atau setel SENTIMENT_TEMPERATURE di .env secara sadar. "
            "Memakai suhu model lain akan salah kalibrasi tanpa galat."
        )
if "SENTIMENT_REVIEW_THRESHOLD" not in os.environ and _profil.get("review_threshold"):
    settings.sentiment_review_threshold = _profil["review_threshold"]

# Set HuggingFace cache directories before transformers is imported
os.environ.setdefault("HF_HOME", settings.hf_home)
# TRANSFORMERS_CACHE SENGAJA tidak disetel kecuali diminta eksplisit. Tanpa
# variabel itu transformers memakai `huggingface_hub.constants.HF_HUB_CACHE` -
# lokasi yang sama dengan `hf download` - apa pun urutan impornya. Menyetelnya
# ke jalur mana pun bisa menyimpang dari konstanta itu, karena konstanta
# tersebut dihitung saat huggingface_hub PERTAMA KALI diimpor: bila itu terjadi
# sebelum baris HF_HOME di atas, ia jatuh ke ~/.cache. Terukur di VPS: model
# dasar yang diunduh lebih dulu dengan revisi terkunci tidak terbaca.
if settings.transformers_cache:
    os.environ.setdefault("TRANSFORMERS_CACHE", settings.transformers_cache)
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


# ── Revisi model dasar di Hugging Face Hub ──────────────────────────────────
# `from_pretrained("nama")` tanpa revisi selalu menarik cabang `main`. Di mesin
# yang cache-nya masih kosong - server yang baru dipasang, atau dipasang ulang -
# itu berarti versi TERBARU, bukan versi yang dipakai skripsi. Kalau pemilik
# `crypter70` memperbarui modelnya, titik berangkat protokol kolam
# (`retrain_from_baseline`) ikut berganti diam-diam, dan putaran berikutnya
# tidak lagi sebanding dengan iterasi 0-6. Tidak ada galat apa pun yang
# memberi tahu: angkanya hanya berubah.
#
# Revisi di bawah adalah yang ada di cache mesin pengembang - mesin tempat
# seluruh eksperimen dan keenam iterasi active learning dijalankan.
#
# Hanya id Hub yang tercantum yang dikunci. Folder checkpoint lokal
# (`models/sentiment_retrained`) dan kandidat lain di SENTIMENT_MODELS tidak
# disentuh: memberi `revision` pada path lokal tidak bermakna.
HF_MODEL_REVISIONS = {
    "crypter70/IndoBERT-Sentiment-Analysis": "e4f806186f3e4adebe369bef85b328bc3c819eb2",
    "indobenchmark/indobert-base-p1": "c2cd0b51ddce6580eb35263b39b0a1e5fb0a39e2",
}


def hub_kwargs(nama: str) -> dict:
    """
    Argumen tambahan untuk `from_pretrained(nama, **hub_kwargs(nama))`.

    Mengembalikan `{"revision": sha}` untuk model dasar yang dikunci, dan
    `{}` untuk selainnya - termasuk path lokal - sehingga pemanggilnya tidak
    perlu tahu apakah `nama` itu id Hub atau folder di disk.
    """
    if not settings.hf_pin_revisions:
        return {}
    revisi = HF_MODEL_REVISIONS.get(nama)
    return {"revision": revisi} if revisi else {}
