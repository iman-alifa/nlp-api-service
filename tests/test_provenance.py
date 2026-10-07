"""
Tes untuk catatan asal-usul checkpoint (`app/utils/training.py`).

Latar belakangnya nyata: dua pelatihan verifikasi dengan data sintetis pernah
tertulis ke `models/aspect_retrained.pt` tanpa meninggalkan jejak apa pun pada
berkas itu sendiri. Yang diuji di sini adalah perilakunya - sidik jari
membedakan kolam yang berbeda, dan kegagalan menulis tidak menggagalkan
pelatihan - bukan bentuk teksnya.
"""
import json
import os

from app.utils.training import pool_fingerprint, write_provenance


def test_sidik_jari_tidak_bergantung_urutan():
    assert pool_fingerprint(['b', 'a', 'c']) == pool_fingerprint(['a', 'c', 'b'])


def test_sidik_jari_membedakan_kolam_nyata_dari_data_karangan():
    nyata = ['gajinya kecil sekali', 'pelayanan kpk bagus']
    karangan = ['kalimat uji satu', 'kalimat uji dua']
    assert pool_fingerprint(nyata) != pool_fingerprint(karangan)


def test_sidik_jari_berubah_saat_satu_baris_ditambahkan():
    """Kolam yang bertambah harus terlihat berbeda - itu inti protokol kolam."""
    dasar = ['a', 'b']
    assert pool_fingerprint(dasar) != pool_fingerprint(dasar + ['c'])


def test_ditulis_di_dalam_direktori_untuk_checkpoint_sentimen(tmp_path):
    d = tmp_path / 'sentiment_retrained'
    d.mkdir()
    write_provenance(str(d), {'modul': 'sentiment', 'jumlah_sampel': 330})
    isi = json.loads((d / 'provenance.json').read_text(encoding='utf-8'))
    assert isi['jumlah_sampel'] == 330
    assert 'dicatat_pada' in isi


def test_ditulis_di_sebelah_berkas_untuk_checkpoint_aspek(tmp_path):
    f = tmp_path / 'aspect_retrained.pt'
    f.write_bytes(b'bobot')
    write_provenance(str(f), {'modul': 'aspect'})
    isi = json.loads((tmp_path / 'aspect_retrained.pt.provenance.json')
                     .read_text(encoding='utf-8'))
    assert isi['modul'] == 'aspect'
    assert f.read_bytes() == b'bobot', 'checkpoint tidak boleh tersentuh'


def test_kegagalan_menulis_tidak_melempar(tmp_path):
    """Pelatihan enam menit tidak boleh hilang karena satu berkas JSON."""
    mustahil = os.path.join(str(tmp_path), 'tidak', 'ada', 'checkpoint.pt')
    write_provenance(mustahil, {'modul': 'aspect'})  # tidak boleh raise


def test_nilai_tak_terserialkan_tidak_melempar(tmp_path):
    f = tmp_path / 'ckpt.pt'
    f.write_bytes(b'x')
    write_provenance(str(f), {'aneh': {1, 2, 3}})  # set bukan JSON
