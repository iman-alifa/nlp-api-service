"""
Bawaan lokasi bobot dasar aspek untuk protokol kolam.

Setelan yang terlupa tidak boleh diam-diam mematikan protokol: dulu itu
berujung pelatihan yang melanjutkan checkpoint dan satu iterasi dibatalkan.
"""
import os

from app.services import aspect_service
from app.services.aspect_service import jalur_bobot_dasar_aspek


def test_setelan_kosong_jatuh_ke_model_path(monkeypatch, tmp_path):
    monkeypatch.setattr(aspect_service.settings, 'aspect_baseline_checkpoint', '')
    monkeypatch.setattr(aspect_service.settings, 'model_path', str(tmp_path))
    assert jalur_bobot_dasar_aspek() == os.path.join(str(tmp_path), 'aspect_baseline.pt')


def test_setelan_eksplisit_menang(monkeypatch, tmp_path):
    monkeypatch.setattr(aspect_service.settings, 'aspect_baseline_checkpoint', '/lain/k6.pt')
    monkeypatch.setattr(aspect_service.settings, 'model_path', str(tmp_path))
    assert jalur_bobot_dasar_aspek() == '/lain/k6.pt'
