"""
Test normalisasi nama aspek untuk keperluan evaluasi.

Latar belakangnya sebuah artefak pengukuran, bukan cacat model. Mode
`automatic` mengembalikan bentuk permukaan seperti yang tertulis di teks
(`pelayanannya`), sedangkan anotator manusia menulis bentuk dasar
(`pelayanan`). Bahasa Indonesia aglutinatif, jadi membandingkan keduanya
sebagai string persis menghitung pasangan yang BENAR sebagai false positive
sekaligus false negative.

Terukur pada 6 kalimat / 8 aspek emas, dari keluaran `document_aspects` yang
sama: micro-F1 **0,118** dengan string persis (TP=1 FP=8 FN=7) versus
**0,941** setelah dinormalkan (TP=8 FP=1 FN=0). Angka model tidak berubah -
yang berubah hanya cara mengukurnya.

Berkas ini tidak memuat bobot model: normalisasi murni Sastrawi.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.aspect_service import AspectService


@pytest.fixture(scope='module')
def svc():
    # Konstruktor sengaja murah; bobot baru dimuat di _ensure_loaded().
    return AspectService()


@pytest.fixture(scope='module')
def client():
    with TestClient(app) as c:
        yield c


# ── Normalisasi satu nama ───────────────────────────────────────────────────

@pytest.mark.parametrize('permukaan, dasar', [
    ('pelayanannya', 'pelayanan'),
    ('pajaknya', 'pajak'),
    ('sistemnya', 'sistem'),
    ('gedungnya', 'gedung'),
    ('fasilitasnya', 'fasilitas'),
])
def test_bentuk_berimbuhan_dan_bentuk_dasar_bertemu_di_hasil_yang_sama(svc, permukaan, dasar):
    """Inti perbaikannya: kedua sisi harus mendarat di token yang sama.

    Nilai stem-nya sendiri tidak dijamin sama dengan bentuk dasar yang ditulis
    manusia (`pelayanan` -> `layan`), dan itu tidak masalah. Yang wajib benar
    adalah keduanya bertemu, karena evaluasi menormalkan KEDUA sisi.
    """
    assert svc.normalize_aspect_name(permukaan) == svc.normalize_aspect_name(dasar)


def test_normalisasi_bersifat_idempoten(svc):
    """Menormalkan hasil normalisasi tidak boleh mengubahnya lagi.

    Label emas yang sudah tersimpan dalam bentuk dasar akan ikut dinormalkan
    saat dibandingkan; kalau tidak idempoten, ia bergeser lagi dan meleset.
    """
    for kata in ['pelayanannya', 'pajak', 'anggaran', 'fasilitas umum']:
        sekali = svc.normalize_aspect_name(kata)
        assert svc.normalize_aspect_name(sekali) == sekali


def test_frasa_dinormalkan_per_kata(svc):
    assert svc.normalize_aspect_name('Pelayanan Publik') == svc.normalize_aspect_name('pelayanannya publik')


def test_tanda_baca_dan_huruf_besar_diabaikan(svc):
    assert svc.normalize_aspect_name('  Pajak.  ') == svc.normalize_aspect_name('pajak')


def test_nama_kosong_tidak_meledak(svc):
    assert svc.normalize_aspect_name('') == ''
    assert svc.normalize_aspect_name(None) == ''
    assert svc.normalize_aspect_name('   ') == ''


def test_aspek_berbeda_tetap_berbeda(svc):
    """Normalisasi tidak boleh meruntuhkan aspek yang memang tidak sama."""
    dasar = {
        svc.normalize_aspect_name(k)
        for k in ['pajak', 'pelayanan', 'gedung', 'anggaran', 'sistem']
    }
    assert len(dasar) == 5


# ── Keluaran analisis ───────────────────────────────────────────────────────

def test_document_aspects_normalized_sejajar_indeks(svc):
    """Kontrak penjajaran sama ketatnya dengan document_aspects.

    Evaluasi memasangkan baris emas dengan prediksi berdasarkan posisi; satu
    baris yang tergeser merusak seluruh sisanya.
    """
    doc = [['Pelayanannya'], [], ['Pajaknya', 'Fasilitasnya'], ['gedungnya']]
    norm = svc._build_document_aspects_normalized(doc)

    assert len(norm) == len(doc)
    assert norm[1] == []
    assert len(norm[2]) == 2
    assert norm[0] == [svc.normalize_aspect_name('pelayanan')]


def test_normalisasi_membuang_duplikat_yang_muncul_setelah_di_stem(svc):
    """`pelayanan` dan `pelayanannya` dalam satu dokumen adalah satu aspek.

    Tanpa dedup, aspek itu terhitung dua kali dan menghasilkan satu false
    positive tambahan pada tiap dokumen yang memuat dua varian.
    """
    norm = svc._build_document_aspects_normalized([['pelayanan', 'pelayanannya']])
    assert len(norm[0]) == 1


# ── Endpoint ────────────────────────────────────────────────────────────────

def test_endpoint_normalisasi_mengembalikan_peta_dan_daftar(client):
    r = client.post('/api/normalize/aspects',
                    json={'aspects': ['Pelayanan', 'pajaknya', 'Gedung']})

    assert r.status_code == 200
    hasil = r.json()['results']

    assert set(hasil['normalized'].keys()) == {'Pelayanan', 'pajaknya', 'Gedung'}
    assert len(hasil['aspects']) == 3
    # Urutan daftar mengikuti urutan permintaan.
    assert hasil['aspects'][0] == hasil['normalized']['Pelayanan']


def test_endpoint_menyamakan_label_emas_dengan_bentuk_permukaan(client):
    """Justru inilah yang dipakai sisi Laravel saat mengimpor golden dataset."""
    r = client.post('/api/normalize/aspects',
                    json={'aspects': ['pelayanan', 'pelayanannya']})

    hasil = r.json()['results']['aspects']
    assert hasil[0] == hasil[1]


def test_endpoint_menolak_daftar_kosong(client):
    assert client.post('/api/normalize/aspects', json={'aspects': []}).status_code == 422
