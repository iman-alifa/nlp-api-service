"""
Kebijakan preprocessing per-tugas.

Aturan berbeda untuk tiap model, dan salah menerapkannya pernah membalik
hasil sentimen secara diam-diam.
"""

import pytest

from app.services.sentiment_service import SentimentService
from app.preprocessing.text_cleaner import NEGATION_WORDS, TextCleaner


AGGRESSIVE = {
    'case_folding': True,
    'remove_punctuation': True,
    'remove_stopwords': True,
    'stemming': True,
}


# ── Perlindungan negasi ─────────────────────────────────────────────────────

def test_kata_negasi_tidak_ikut_terbuang():
    """
    Sastrawi/NLTK memasukkan 'tidak', 'bukan', 'jangan' ke stopword.
    Membuangnya membalik polaritas: "tidak bagus" -> "bagus".
    """
    cleaner = TextCleaner(AGGRESSIVE)

    for kata in ('tidak', 'bukan', 'jangan', 'kurang', 'belum', 'tanpa'):
        assert kata not in cleaner.all_stopwords, f"'{kata}' seharusnya dilindungi"


def test_negasi_bertahan_setelah_pembersihan():
    cleaner = TextCleaner(AGGRESSIVE)

    assert 'tidak' in cleaner.clean_text("pelayanannya tidak bagus")


def test_perlindungan_negasi_bisa_dimatikan():
    cleaner = TextCleaner({**AGGRESSIVE, 'protect_negation': False})

    assert 'tidak' in cleaner.all_stopwords


def test_stopword_biasa_tetap_dibuang():
    """Perlindungan negasi tidak boleh melumpuhkan stopword removal."""
    cleaner = TextCleaner(AGGRESSIVE)

    assert 'yang' in cleaner.all_stopwords
    assert 'dengan' in cleaner.all_stopwords


def test_daftar_negasi_mencakup_ragam_informal():
    for kata in ('gak', 'nggak', 'tdk', 'jgn', 'blm'):
        assert kata in NEGATION_WORDS


# ── Konfigurasi khusus jalur BERT ───────────────────────────────────────────

def test_sentimen_menonaktifkan_stemming_dan_stopword():
    """
    IndoBERT memakai tokenisasi subword dan dilatih pada teks alami; membuang
    imbuhan dan kata fungsi menghilangkan sinyal yang justru dipakainya.
    """
    config = SentimentService._sanitize_config(AGGRESSIVE)

    assert config['stemming'] is False
    assert config['remove_stopwords'] is False
    assert config['lemmatization'] is False


def test_sentimen_mempertahankan_normalisasi_permukaan():
    """Slang & typo tetap dinormalisasi: mendekatkan teks ke distribusi pralatih."""
    config = SentimentService._sanitize_config(
        {**AGGRESSIVE, 'normalize_slang': True, 'fix_typos': True}
    )

    assert config['normalize_slang'] is True
    assert config['fix_typos'] is True
    assert config['case_folding'] is True


def test_sentimen_menerima_config_kosong():
    assert SentimentService._sanitize_config(None)['stemming'] is False


def test_teks_bernegasi_utuh_lewat_jalur_sentimen():
    """Uji ujung-ke-ujung kebijakan tanpa memuat model."""
    config = SentimentService._sanitize_config(AGGRESSIVE)
    hasil = TextCleaner(config).clean_text("pelayanannya tidak bagus sama sekali")

    assert 'tidak' in hasil
    assert 'bagus' in hasil
    # tanpa stemming, 'pelayanannya' tidak berubah jadi 'layan'
    assert 'layan' not in hasil.split()


# ── Length bucketing pada inferensi sentimen ────────────────────────────────

def test_urutan_keluaran_sama_dengan_masukan(monkeypatch):
    """
    Batch diurutkan berdasarkan panjang demi efisiensi padding, tetapi hasilnya
    HARUS kembali ke urutan semula - seluruh pemetaan indeks bergantung padanya.
    """
    svc = SentimentService()

    # Palsukan model agar test tidak memuat bobot: label ditentukan panjang teks.
    class FakeProbs:
        def __init__(self, n): self.n = n
        def __getitem__(self, j): return self
        def item(self): return 0.9

    texts = ['a' * 300, 'b', 'c' * 50, 'd' * 5, 'e' * 120]

    hasil = svc._predict_rule_based(texts)

    assert [h['text'] for h in hasil] == texts


def test_semua_posisi_terisi():
    """Tidak boleh ada lubang None pada keluaran."""
    svc = SentimentService()
    texts = ['bagus sekali', '', 'buruk sekali']

    hasil = svc._predict_rule_based(texts)

    assert len(hasil) == len(texts)
    assert all(h is not None and 'sentiment' in h for h in hasil)
