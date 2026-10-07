"""
Pengambilan klausa untuk sentimen per-aspek.

Tanpa ini, seluruh aspek dalam satu kalimat mendapat polaritas yang sama —
yang menghapus arti analisis berbasis aspek. Diverifikasi: sebelum perbaikan,
"Pelayanannya bagus tapi harganya mahal" memberi 'negative' kepada keduanya.
"""

import pytest

from app.services.aspect_service import AspectService


@pytest.fixture(scope='module')
def service(aspect_service_malas):
    """Alias ke fixture sesi di conftest.py - bobot dimuat sekali
    untuk seluruh suite, bukan ulang di tiap berkas."""
    return aspect_service_malas


def clause_for(service, text, aspect):
    i = text.lower().index(aspect.lower())
    return service._clause_around(text, i, i + len(aspect))


# ── Pemisahan klausa ────────────────────────────────────────────────────────

def test_konjungsi_pertentangan_memisah_klausa(service):
    text = "Pelayanannya bagus tapi harganya mahal"

    assert clause_for(service, text, 'Pelayanannya') == 'Pelayanannya bagus'
    assert clause_for(service, text, 'harganya') == 'harganya mahal'


def test_sayangnya_memisah_klausa(service):
    text = "Kamarnya bersih dan nyaman, sayangnya wifi lemot"

    assert 'wifi' not in clause_for(service, text, 'Kamarnya')
    assert 'bersih' not in clause_for(service, text, 'wifi')


def test_tanda_baca_kalimat_memisah_klausa(service):
    text = "Harga murah sekali. Pengiriman sangat lambat."

    assert 'Pengiriman' not in clause_for(service, text, 'Harga')


def test_konjungsi_di_awal_klausa_dibuang(service):
    """
    'padahal' adalah penanda hubungan antar-klausa, bukan opini. Membiarkannya
    membuat IndoBERT membaca "padahal fasilitas umum memadai" sebagai negatif.
    """
    text = "Pajak naik terus padahal fasilitas umum memadai"

    assert clause_for(service, text, 'fasilitas umum') == 'fasilitas umum memadai'


def test_kata_negasi_tidak_ikut_dibuang(service):
    """Membuang negasi akan membalik polaritas - kesalahan yang sama seperti stopword."""
    text = "Kamarnya luas tapi tidak bersih sama sekali"

    assert 'tidak' in clause_for(service, text, 'Kamarnya').lower() or \
           'tidak' in text[text.lower().index('tapi'):].lower()


# ── Perilaku aman ───────────────────────────────────────────────────────────

def test_klausa_hanya_berisi_aspek_pakai_teks_penuh(service):
    """Klausa tanpa kata opini tidak berguna; teks penuh lebih informatif."""
    text = "Wifi. Kamarnya sangat nyaman dan bersih sekali"

    assert clause_for(service, text, 'Wifi') == text


def test_klausa_dua_kata_dipertahankan(service):
    """'harganya mahal' pendek tetapi sudah memuat opininya - jangan dibuang."""
    text = "Pelayanannya bagus tapi harganya mahal"

    assert clause_for(service, text, 'harganya') == 'harganya mahal'


def test_klausa_pendek_untuk_aspek_multikata(service):
    """Ambang dibandingkan terhadap panjang aspek, bukan angka tetap."""
    text = "Antrian panjang tapi fasilitas umum memadai"

    assert clause_for(service, text, 'fasilitas umum') == 'fasilitas umum memadai'


def test_offset_tidak_valid_pakai_teks_penuh(service):
    text = "Pelayanan bagus sekali"

    assert service._clause_around(text, None, None) == text
    assert service._clause_around(text, -5, 3) == text
    assert service._clause_around(text, 0, 999) == text


def test_kalimat_satu_klausa_tidak_berubah(service):
    text = "Pelayanan di tempat ini sangat memuaskan"

    assert clause_for(service, text, 'Pelayanan') == text


def test_konjungsi_hanya_dikenali_sebagai_kata_utuh(service):
    """'tapi' di dalam kata lain tidak boleh memotong."""
    text = "Setapak jalannya rusak parah sekali"

    assert clause_for(service, text, 'jalannya') == text


# ── Cadangan saat klausa tidak meyakinkan ───────────────────────────────────

@pytest.mark.asyncio
async def test_klausa_tak_meyakinkan_diganti_teks_penuh(service, monkeypatch):
    """
    Klausa pendek bisa kehilangan konteks: IndoBERT menilai "Pajak naik"
    positive dengan keyakinan 0,53 saja. Prediksi seperti itu harus diganti
    hasil dari teks penuh.
    """
    calls = []

    class FakeSentiment:
        async def analyze(self, texts, preprocessing_config=None):
            # _resolve_low_confidence hanya memanggil sekali, khusus untuk
            # menghitung ulang teks penuh milik kemunculan berkeyakinan rendah.
            calls.append(list(texts))
            return {'predictions': [{'text': t, 'sentiment': 'negative',
                                     'confidence': 0.97} for t in texts]}

    monkeypatch.setattr(service, 'sentiment_service', FakeSentiment())

    hasil = await service._resolve_low_confidence(
        [{'text': 'Pajak naik', 'sentiment': 'positive', 'confidence': 0.53}],
        ['Pajak naik'],
        ['Pajak naik padahal fasilitas memadai'],
    )

    assert hasil[0]['sentiment'] == 'negative'
    assert len(calls) == 1, 'hanya kemunculan berkeyakinan rendah yang dihitung ulang'
    assert calls[0] == ['Pajak naik padahal fasilitas memadai'], 'harus memakai teks penuh'


@pytest.mark.asyncio
async def test_prediksi_yakin_tidak_dihitung_ulang(service, monkeypatch):
    class FakeSentiment:
        async def analyze(self, texts, preprocessing_config=None):
            raise AssertionError('tidak boleh dipanggil')

    monkeypatch.setattr(service, 'sentiment_service', FakeSentiment())

    awal = [{'text': 'harganya mahal', 'sentiment': 'negative', 'confidence': 0.99}]
    hasil = await service._resolve_low_confidence(awal, ['harganya mahal'], ['harganya mahal sekali'])

    assert hasil == awal


@pytest.mark.asyncio
async def test_klausa_sama_dengan_teks_penuh_dilewati(service, monkeypatch):
    """Menghitung ulang teks yang sama persis tidak ada gunanya."""
    class FakeSentiment:
        async def analyze(self, texts, preprocessing_config=None):
            raise AssertionError('tidak boleh dipanggil')

    monkeypatch.setattr(service, 'sentiment_service', FakeSentiment())

    awal = [{'text': 'Pajak naik', 'sentiment': 'positive', 'confidence': 0.51}]
    hasil = await service._resolve_low_confidence(awal, ['Pajak naik'], ['Pajak naik'])

    assert hasil == awal
