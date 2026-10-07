"""
Test penggabungan aspek yang hanya berbeda bentuk infleksi.

Keluhan nyata dari studi kasus TNI: dasbor menampilkan `gaji` dan `gajinya`
sebagai dua aspek berbeda, dengan jumlah dan statistik sentimen terbelah.
Bahasa Indonesia aglutinatif dan ekstraksi mengembalikan kata seperti yang
tertulis, jadi satu aspek yang sama terpecah menjadi beberapa entri.

Dua mesin yang berbeda, sengaja dipisah:

- **Klitik posesif** (`-nya`, `-ku`, `-mu`) dipotong tanpa kamus. Infleksi
  murni, tidak pernah mengubah makna atau kelas kata. Ini yang menangani
  kosakata di luar kamus Sastrawi - dan kosakata lembaga justru banyak yang
  begitu (`alutsista`, `danramil`, `kodim`).
- **Stem Sastrawi** menangani sisanya, dengan penjaga kelas awalan supaya
  `petugas` tidak melebur dengan `tugas`.

Tidak memuat bobot model: seluruhnya menguji fungsi penggabungan langsung.
"""

import pytest

from app.services.aspect_service import AspectService


@pytest.fixture(scope='module')
def svc():
    return AspectService()


def entri(nama, count=1, confidence=0.9, n_occ=None):
    """Satu entri aspek berbentuk seperti keluaran ekstraksi."""
    n_occ = count if n_occ is None else n_occ
    return {
        'aspect': nama,
        'count': count,
        'confidence': confidence,
        'variants': [nama],
        'occurrences': [
            {'text': f'contoh {nama}', 'text_index': i, 'start': 0, 'end': len(nama)}
            for i in range(n_occ)
        ],
        'score': count * confidence,
    }


# ── Klitik posesif ──────────────────────────────────────────────────────────

@pytest.mark.parametrize('kata, harapan', [
    ('gajinya', 'gaji'),
    ('anggarannya', 'anggaran'),
    ('kondisinya', 'kondisi'),
    ('prajuritnya', 'prajurit'),
    # Di luar kamus Sastrawi - justru ini yang tidak tersentuh stemming.
    ('alutsistanya', 'alutsista'),
    ('danramilnya', 'danramil'),
])
def test_klitik_dipotong(svc, kata, harapan):
    assert svc._strip_clitic(kata) == harapan


@pytest.mark.parametrize('kata', [
    # Sisa 2 huruf - tertahan batas panjang.
    'punya', 'hanya', 'tanya', 'kamu', 'ilmu', 'nya',
    # Sisa 3+ huruf - HANYA tertahan oleh pemeriksaan kamus. Diperiksa
    # terhadap 29 933 kata dasar Sastrawi: ada 13 kata dasar berakhiran -nya,
    # dan enam di antaranya lolos batas panjang. Tanpa kamus, `makanya` akan
    # dipotong menjadi `maka` dan `bahwasanya` menjadi non-kata `bahwasa`.
    'makanya', 'empunya', 'bahwasanya', 'mangkanya', 'nyonya', 'adakalanya',
])
def test_kata_dasar_berakhiran_nya_tidak_dipotong(svc, kata):
    assert svc._strip_clitic(kata) == kata


@pytest.mark.parametrize('kata, harapan', [
    ('tninya', 'tni'), ('kpknya', 'kpk'), ('dprnya', 'dpr'),
])
def test_akronim_tiga_huruf_tetap_dipotong(svc, kata, harapan):
    """Korpus lembaga penuh akronim; batas 4 huruf melewatkan semuanya."""
    assert svc._strip_clitic(kata) == harapan


@pytest.mark.parametrize('frasa, harapan', [
    ('gaji nya', 'gaji'),
    ('tunjangan nya', 'tunjangan'),
    ('gedung nya', 'gedung'),
])
def test_klitik_yang_ditulis_terpisah_ikut_dibuang(svc, frasa, harapan):
    """Penulis komentar sering mengetik `gaji nya` dengan spasi.

    Terukur pada studi kasus enam lembaga: bentuk berspasi tidak pernah
    bergabung dengan induknya, sehingga `gaji` dan `gaji nya` tetap menjadi dua
    aspek terpisah di dasbor.
    """
    assert svc._strip_clitics(frasa) == harapan


def test_klitik_tunggal_tidak_dihabiskan(svc):
    """Membuang klitik tidak boleh menyisakan nama aspek kosong."""
    assert svc._strip_clitics('nya') == 'nya'


def test_klitik_dipotong_per_kata_pada_frasa(svc):
    assert svc._strip_clitics('pelatihan prajuritnya') == 'pelatihan prajurit'


# ── Penjaga kelas awalan ────────────────────────────────────────────────────

def test_petugas_tidak_melebur_dengan_tugas(svc):
    """Stem Sastrawi meratakan awalan pembentuk pelaku.

    `petugas` dan `tugas` sama-sama menjadi `tugas`, padahal aspeknya berbeda:
    satu orangnya, satu pekerjaannya. Tanpa penjaga ini penggabungan justru
    memperkenalkan kesalahan baru sambil memperbaiki yang lama.
    """
    hasil = svc._merge_aspect_variants([
        entri('petugas', 3), entri('petugasnya', 2), entri('tugas', 4),
    ])
    nama = {a['aspect'] for a in hasil}

    assert 'petugas' in nama
    assert 'tugas' in nama
    assert len(hasil) == 2


def test_bentuk_berklitik_tetap_bergabung_ke_induknya(svc):
    hasil = svc._merge_aspect_variants([entri('gaji', 5), entri('gajinya', 3)])

    assert len(hasil) == 1
    assert hasil[0]['aspect'] == 'gaji'
    assert hasil[0]['count'] == 8
    assert set(hasil[0]['merged_from']) == {'gaji', 'gajinya'}


def test_kata_di_luar_kamus_sastrawi_tetap_bergabung(svc):
    """Ini yang tidak bisa dikerjakan stemming sendirian."""
    hasil = svc._merge_aspect_variants([
        entri('alutsista', 2), entri('alutsistanya', 3),
    ])

    assert len(hasil) == 1
    assert hasil[0]['aspect'] == 'alutsista'
    assert hasil[0]['count'] == 5


# ── Label yang ditampilkan ──────────────────────────────────────────────────

def test_label_memakai_bentuk_permukaan_bukan_stem(svc):
    """Pengguna menulis `pelayanan`; menampilkan `layan` akan membingungkan.

    Prinsip yang sama dengan `_make_words_readable()` di modul topik: memodelkan
    pada stem itu benar, menampilkan stem tidak.
    """
    hasil = svc._merge_aspect_variants([
        entri('pelayanan', 5), entri('pelayanannya', 2),
    ])

    assert hasil[0]['aspect'] == 'pelayanan'


def test_label_dibersihkan_walau_tidak_ada_varian_lain(svc):
    """Aspek yang hanya pernah muncul sebagai `anggarannya` tetap harus terbaca."""
    hasil = svc._merge_aspect_variants([entri('anggarannya', 4)])

    assert hasil[0]['aspect'] == 'anggaran'
    assert 'anggarannya' in hasil[0]['variants']
    assert hasil[0]['count'] == 4


def test_label_dipilih_dari_varian_paling_sering(svc):
    hasil = svc._merge_aspect_variants([
        entri('pelayanannya', 9), entri('pelayanan', 2),
    ])

    # `pelayanannya` lebih sering, tetapi klitiknya tetap dipotong.
    assert hasil[0]['aspect'] == 'pelayanan'
    assert hasil[0]['count'] == 11


def test_hasil_deterministik_untuk_jumlah_yang_seri(svc):
    """Dua analisis atas korpus yang sama harus memberi label yang sama."""
    a = svc._merge_aspect_variants([entri('pelayanan', 3), entri('pelayanannya', 3)])
    b = svc._merge_aspect_variants([entri('pelayanannya', 3), entri('pelayanan', 3)])

    assert a[0]['aspect'] == b[0]['aspect']


# ── Data turunan ikut tergabung ─────────────────────────────────────────────

def test_kemunculan_digabung_bukan_dibuang(svc):
    """Sentimen per aspek dihitung dari occurrences; kehilangan satu berarti
    aspek itu dinilai dari sebagian kalimatnya saja."""
    hasil = svc._merge_aspect_variants([
        entri('gaji', 2, n_occ=2), entri('gajinya', 3, n_occ=3),
    ])

    assert len(hasil[0]['occurrences']) == 5


def test_confidence_dirata_rata_berbobot(svc):
    """Rata-rata polos membuat varian langka menarik angkanya sekuat varian
    yang muncul puluhan kali."""
    hasil = svc._merge_aspect_variants([
        entri('gaji', count=9, confidence=1.0),
        entri('gajinya', count=1, confidence=0.0),
    ])

    assert hasil[0]['confidence'] == pytest.approx(0.9)


def test_variants_menyimpan_seluruh_bentuk_asal(svc):
    """Penggabungan yang salah harus terlihat, bukan tersembunyi."""
    hasil = svc._merge_aspect_variants([
        entri('pelayanan', 2), entri('pelayanannya', 1),
    ])

    assert {'pelayanan', 'pelayanannya'} <= set(hasil[0]['variants'])


def test_daftar_kosong_aman(svc):
    assert svc._merge_aspect_variants([]) == []


def test_penggabungan_bisa_dimatikan(svc, monkeypatch):
    from app import config

    monkeypatch.setattr(config.settings, 'aspect_merge_variants', False)
    masuk = [entri('gaji', 2), entri('gajinya', 1)]

    assert len(svc._merge_aspect_variants(masuk)) == 2


def test_urutan_hasil_menurut_skor(svc):
    hasil = svc._merge_aspect_variants([
        entri('kecil', 1), entri('besar', 9), entri('sedang', 5),
    ])

    skor = [a['score'] for a in hasil]
    assert skor == sorted(skor, reverse=True)
