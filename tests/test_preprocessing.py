"""
Test TextCleaner — pipeline preprocessing adalah satu-satunya sumber kebenaran
untuk pembersihan teks, dipakai seluruh service.
"""

import pytest

from app.preprocessing.text_cleaner import TextCleaner
from app.schemas.request_schemas import PreprocessingConfig


FULL_CONFIG = {
    'case_folding': True,
    'remove_punctuation': True,
    'remove_numbers': False,
    'remove_stopwords': True,
    'stemming': True,
}


# ── Normalisasi config ──────────────────────────────────────────────────────

def test_menerima_dict():
    cleaner = TextCleaner(FULL_CONFIG)

    assert cleaner.config['stemming'] is True


def test_menerima_model_pydantic():
    """
    /api/preprocess meneruskan objek PreprocessingConfig, bukan dict.
    Sebelum diperbaiki, endpoint itu selalu membalas
    "'PreprocessingConfig' object has no attribute 'get'".
    """
    cleaner = TextCleaner(PreprocessingConfig(stemming=False, remove_stopwords=False))

    assert cleaner.stemming is False
    assert cleaner.remove_stopwords is False


def test_menerima_none():
    cleaner = TextCleaner(None)

    assert cleaner.config == {}
    # default tetap berlaku
    assert cleaner.case_folding is True


# ── Perilaku pembersihan ────────────────────────────────────────────────────

def test_case_folding_dan_tanda_baca():
    cleaner = TextCleaner({'case_folding': True, 'remove_punctuation': True,
                           'remove_stopwords': False, 'stemming': False})

    hasil = cleaner.clean_text("HARGA Mahal!!!")

    assert hasil == hasil.lower()
    assert '!' not in hasil


def test_karakter_berulang_dinormalisasi():
    cleaner = TextCleaner({'remove_stopwords': False, 'stemming': False})

    assert 'bagusss' not in cleaner.clean_text("bagusss")


def test_slang_dinormalisasi():
    cleaner = TextCleaner({'remove_stopwords': False, 'stemming': False})

    assert 'banget' in cleaner.clean_text("bagus bgt")


def test_stemming_menghapus_imbuhan():
    cleaner = TextCleaner({'stemming': True, 'remove_stopwords': False})

    assert 'layan' in cleaner.clean_text("pelayanan")


def test_clean_texts_mempertahankan_jumlah_dan_urutan():
    """
    AssociationService bergantung pada kesejajaran indeks antara
    document_aspects dan document_topics — jumlah teks tidak boleh berubah.
    """
    cleaner = TextCleaner(FULL_CONFIG)
    texts = ["harga mahal", "pelayanan bagus", "pengiriman lambat"]

    hasil = cleaner.clean_texts(texts)

    assert len(hasil) == len(texts)


def test_teks_kosong_tidak_error():
    cleaner = TextCleaner(FULL_CONFIG)

    assert cleaner.clean_texts(["", "   "]) == ["", ""] or len(cleaner.clean_texts(["", "   "])) == 2


def test_emoticon_dibuang():
    cleaner = TextCleaner({'remove_stopwords': False, 'stemming': False})

    hasil = cleaner.clean_text("bagus 😀😀")

    assert '😀' not in hasil


# ── Regresi kamus slang ─────────────────────────────────────────────────────

def _cleaner_transformer():
    return TextCleaner.for_task('transformer', {
        'case_folding': True, 'remove_punctuation': True, 'remove_numbers': False,
    })


def test_angka_telanjang_tidak_diterjemahkan():
    """
    Regresi: kamus dulu memetakan '9'->'yang', '4'->'untuk', '8'->'delapan',
    sehingga "sesuai pasal 4 ayat 9" menjadi "sesuai pasal untuk ayat yang".
    Terukur merusak 161 token di 145 dokumen pada empat korpus proyek.
    """
    hasil = _cleaner_transformer().clean_text('sesuai pasal 4 ayat 9 dan 8 orang')

    assert '4' in hasil.split()
    assert '9' in hasil.split()
    assert 'untuk' not in hasil.split()
    assert 'delapan' not in hasil.split()


def test_kata_baku_tidak_diganti_sinonimnya():
    """
    Normalisasi tidak boleh mengubah makna. 'mantap' dan 'bagus' sama-sama baku
    dengan kadar berbeda; menggantinya membuat model melihat token yang bukan
    ditulis penulisnya.
    """
    c = _cleaner_transformer()

    assert 'mantap' in c.clean_text('pelayanannya mantap sekali')
    assert 'keren' in c.clean_text('filmnya keren banget')
    assert 'jelek' in c.clean_text('barangnya jelek sekali')


def test_bentuk_dasar_tidak_diubah_jadi_berimbuhan():
    """Regresi: 'coba lihat dulu' -> 'mencoba melihat dulu' (arah terbalik)."""
    hasil = _cleaner_transformer().clean_text('coba lihat dulu')

    assert 'coba' in hasil.split()
    assert 'lihat' in hasil.split()


def test_kata_ambigu_tidak_dipaksa_satu_makna():
    """'lagi' bisa berarti "sedang" maupun "kembali"; 'mana' bukan 'dimana'."""
    c = _cleaner_transformer()

    assert 'sedang' not in c.clean_text('dia datang lagi kemarin')
    assert 'dimana' not in c.clean_text('mana yang benar')
    assert 'aku' in c.clean_text('aku suka produk ini')


def test_kamus_slang_tidak_memuat_entri_identitas():
    """Entri k==v tidak melakukan apa pun selain memperbesar kamus."""
    c = TextCleaner({})
    identitas = {k for k, v in c.slang_dict.items() if k == v}

    assert not identitas, f'entri identitas tersisa: {identitas}'


def test_kamus_slang_tidak_menghapus_kata_lewat_string_kosong():
    """
    Memetakan kata ke '' menghapusnya walaupun remove_stopwords dimatikan -
    padahal jalur sentimen sengaja mempertahankan kata fungsi. Penghapusan
    harus lewat lapisan stopword agar tunduk pada kebijakannya.
    """
    c = TextCleaner({})
    kosong = {k for k, v in c.slang_dict.items() if v == ''}

    assert not kosong, f'entri ke string kosong tersisa: {kosong}'


def test_partikel_bertahan_saat_stopword_dimatikan():
    hasil = _cleaner_transformer().clean_text('bagus dong pelayanannya')

    assert 'dong' in hasil.split()


def test_partikel_tetap_dibuang_pada_jalur_bag_of_words():
    hasil = TextCleaner.for_task('bag_of_words', {}).clean_text('bagus dong pelayanannya')

    assert 'dong' not in hasil.split()


def test_typo_frasa_benar_benar_diperbaiki():
    """
    Kunci berisi spasi dulu tidak pernah cocok karena pencocokan hanya per
    token hasil split().
    """
    hasil = _cleaner_transformer().clean_text('terima kasi banyak')

    assert 'terima kasih' in hasil


# ── Opsi konfigurasi yang dulu mati ────────────────────────────────────────

def test_remove_punctuation_benar_benar_berpengaruh():
    """
    Opsi ini dulu dibaca di __init__ tetapi tidak pernah dipakai; tanda baca
    selalu dibuang. Penting bagi jalur transformer: IndoBERT dilatih pada teks
    bertanda baca, dan tanda seru membawa sinyal intensitas.

    Diuji pada profil transformer karena stemming Sastrawi memang melucuti
    karakter non-huruf, sehingga opsi ini hanya bermakna saat stemming mati.
    """
    teks = 'Bagus sekali!!! Tapi mahal, ya?'
    dasar = {'stemming': False, 'remove_stopwords': False}

    dengan = TextCleaner({**dasar, 'remove_punctuation': True}).clean_text(teks)
    tanpa = TextCleaner({**dasar, 'remove_punctuation': False}).clean_text(teks)

    assert '!' not in dengan
    assert '!' in tanpa
    assert ',' in tanpa


def test_lemmatization_memperingatkan_bahwa_tidak_diimplementasikan(caplog):
    """
    Tidak ada lemmatizer bahasa Indonesia yang mapan; Sastrawi adalah stemmer.
    Dulu opsi ini diabaikan diam-diam sehingga pemanggil mengira ia berjalan.
    """
    import logging

    with caplog.at_level(logging.WARNING):
        TextCleaner({'lemmatization': True})

    assert any('lemmatization' in r.message for r in caplog.records)


# ── Profil per tugas ────────────────────────────────────────────────────────

def test_profil_transformer_menolak_stemming_dari_pengguna():
    """
    Permintaan pengguna tidak boleh merusak modul yang tidak menoleransinya.
    Terukur: 5 dari 6 kalimat bernegasi berbalik polaritas bila stemming aktif.
    """
    c = TextCleaner.for_task('transformer', {'stemming': True, 'remove_stopwords': True})

    assert c.stemming is False
    assert c.remove_stopwords is False
    assert 'tidak' in c.clean_text('pelayanannya tidak bagus')


def test_profil_bag_of_words_memaksa_pembersihan_agresif():
    c = TextCleaner.for_task('bag_of_words', {'stemming': False, 'remove_stopwords': False})

    assert c.stemming is True
    assert c.remove_stopwords is True


def test_profil_span_tidak_membersihkan_apa_pun():
    """Ekstraksi aspek berbasis offset karakter; pembersihan menggesernya."""
    assert TextCleaner.for_task('span', {'stemming': True}) is None


def test_profil_tidak_dikenal_ditolak():
    with pytest.raises(ValueError):
        TextCleaner.for_task('entah-apa', {})


# ── Filter token langka (opsional, default mati) ───────────────────────────

def test_filter_token_langka_membuang_yang_muncul_sekali():
    teks = ['pajak naik pajak', 'pajak turun', 'anomali']

    hasil = TextCleaner.filter_rare_tokens(teks, min_freq=2)

    assert hasil[0] == 'pajak pajak'
    assert hasil[2] == ''


def test_filter_token_langka_menjaga_jumlah_dan_urutan():
    teks = ['a b', 'b c', 'd']

    assert len(TextCleaner.filter_rare_tokens(teks, 2)) == 3


def test_filter_token_langka_mati_secara_default():
    """
    Terukur merugikan di bawah ~900 dokumen (c_v 0,4602 -> 0,3940 pada n=120),
    jadi ia tersedia tetapi tidak aktif.
    """
    from app.config import settings

    assert settings.topic_min_token_freq == 1
    teks = ['pajak naik', 'anomali']
    assert TextCleaner.filter_rare_tokens(teks, 1) == teks


# ── Normalisasi huruf berulang ──────────────────────────────────────────────

def test_angka_tidak_dipangkas_oleh_normalisasi_huruf_berulang():
    """
    Regresi berat: pola lama `(.)\1{2,}` berlaku untuk digit juga, sehingga
    "utang 1000 triliun" menjadi "utang 10 triliun" dan "anggaran 100000"
    menjadi "anggaran 10". Fatal untuk korpus yang membahas nominal.
    """
    c = _cleaner_transformer()

    assert '1000' in c.clean_text('utang 1000 triliun')
    assert '100000' in c.clean_text('anggaran 100000 rupiah')
    assert '2000' in c.clean_text('tahun 2000 lalu')


def test_pemanjangan_gaya_medsos_dipangkas_ke_satu_huruf():
    c = _cleaner_transformer()

    assert 'bagus' in c.clean_text('bagusss sekali').split()
    assert 'lama' in c.clean_text('lamaaa banget').split()
    assert 'sangat' in c.clean_text('saaangat baik').split()


def test_kata_berhuruf_ganda_tidak_dirusak():
    """
    Memangkas ke satu huruf tanpa syarat mengubah "maaaaf" menjadi "maf".
    Bentuk dua-huruf dicoba dulu dan diterima bila dikenal kamus Sastrawi.
    """
    assert 'maaf' in _cleaner_transformer().clean_text('maaaaf ya').split()


def test_kamus_kata_dasar_sastrawi_termuat():
    """Keputusan dua-vs-satu huruf berdasar kamus, bukan tebakan."""
    from app.preprocessing.text_cleaner import ROOT_WORDS

    assert len(ROOT_WORDS) > 20000
    assert 'maaf' in ROOT_WORDS
    assert 'maf' not in ROOT_WORDS


# ── Tagar dan mention ───────────────────────────────────────────────────────

def test_kata_pada_tagar_dipertahankan():
    """
    Regresi: pola `#\w+` membuang tagar beserta katanya, sehingga
    "baca #pajak dan #korupsi" menjadi "baca dan" - justru kata paling topikal
    yang hilang, tepat pada modul yang paling membutuhkannya.
    """
    hasil = _cleaner_transformer().clean_text('baca #pajak dan #korupsi').split()

    assert 'pajak' in hasil
    assert 'korupsi' in hasil


def test_mention_tetap_dibuang():
    """Nama akun bukan isi; ia hanya menambah kosakata tanpa makna topik."""
    hasil = _cleaner_transformer().clean_text('lihat @budi bilang apa')

    assert 'budi' not in hasil
    assert 'bilang' in hasil
