"""
Identitas bobot yang hidup di /health (`_identitas_bobot`).

Path checkpoint tidak membedakan iterasi: setiap retraining menimpa berkas yang
sama. Setelah aspek dikembalikan ke iterasi 1 (27 Sep), dari luar tidak ada
yang membedakannya dari iterasi 2, dan halaman depan Laravel menampilkan angka
model yang tidak dipakai. Sidik jari kolam dari provenance.json menutup celah
itu karena sama persis dengan yang dicatat Laravel di model_trainings.
"""
import json
from unittest import mock

import pytest
from fastapi.testclient import TestClient

from app import main
from app.main import _identitas_bobot, app


def _tulis(path, isi):
    path.write_text(json.dumps(isi), encoding='utf-8')
    return str(path)


def test_membaca_sidik_jari_dan_jumlah_sampel(tmp_path):
    p = _tulis(tmp_path / 'aspect_retrained.pt.provenance.json',
               {'sidik_jari_kolam': 'e3011f86d97550a5', 'jumlah_sampel': 450})
    assert _identitas_bobot(p) == {
        'pool_fingerprint': 'e3011f86d97550a5', 'samples': 450, 'restored_from': None,
    }


def test_bobot_yang_dipulihkan_menyebut_commitnya(tmp_path):
    p = _tulis(tmp_path / 'prov.json', {
        'sidik_jari_kolam': 'e3011f86d97550a5', 'jumlah_sampel': 450,
        'dipulihkan': {'dari_commit_hf': '9116ca2fd9a0'},
    })
    assert _identitas_bobot(p)['restored_from'] == '9116ca2fd9a0'


@pytest.mark.parametrize('isi', [None, 'bukan json {'])
def test_tanpa_catatan_atau_rusak_berarti_tidak_diketahui(tmp_path, isi):
    """Bobot dasar tidak punya provenance; catatan rusak tidak boleh menjatuhkan /health."""
    if isi is None:
        jalur = str(tmp_path / 'tidak-ada.json')
    else:
        (tmp_path / 'rusak.json').write_text(isi, encoding='utf-8')
        jalur = str(tmp_path / 'rusak.json')
    assert _identitas_bobot(jalur) == {'pool_fingerprint': None, 'samples': None, 'restored_from': None}


def test_health_melaporkan_identitas_bobot_yang_hidup(tmp_path):
    ckpt = tmp_path / 'aspect_retrained.pt'
    ckpt.write_bytes(b'')
    _tulis(tmp_path / 'aspect_retrained.pt.provenance.json', {
        'sidik_jari_kolam': 'e3011f86d97550a5', 'jumlah_sampel': 450,
        'dipulihkan': {'dari_commit_hf': '9116ca2fd9a0'},
    })
    sentimen_dir = tmp_path / 'sentiment_retrained'
    sentimen_dir.mkdir()
    _tulis(sentimen_dir / 'provenance.json', {'sidik_jari_kolam': 'aaaabbbbccccdddd', 'jumlah_sampel': 591})

    with TestClient(app) as client, \
            mock.patch.object(main.aspect_service, '_model_path', str(ckpt)), \
            mock.patch.object(main.sentiment_service, 'model_name', str(sentimen_dir)), \
            mock.patch.object(main.sentiment_service, 'using_retrained', True):
        models = client.get('/health').json()['models']

    assert models['aspect_pool_fingerprint'] == 'e3011f86d97550a5'
    assert models['aspect_pool_samples'] == 450
    assert models['aspect_restored_from'] == '9116ca2fd9a0'
    assert models['sentiment_pool_fingerprint'] == 'aaaabbbbccccdddd'
    assert models['sentiment_pool_samples'] == 591


def test_sentimen_bobot_dasar_tidak_punya_sidik_jari():
    with TestClient(app) as client, \
            mock.patch.object(main.sentiment_service, 'using_retrained', False):
        models = client.get('/health').json()['models']

    assert models['sentiment_pool_fingerprint'] is None
