"""
Penguncian revisi model dasar di Hugging Face Hub (`hub_kwargs`).

Tanpa revisi, `from_pretrained` di server yang cache-nya kosong menarik versi
TERBARU. Bila pemilik model memperbaruinya, titik berangkat protokol kolam ikut
berganti diam-diam dan pelatihan berikutnya tidak lagi sebanding dengan
iterasi 0-6 skripsi. Tes di bawah menguji perilaku: revisi benar-benar sampai
ke pemanggilan `from_pretrained`, dan tidak diberikan pada folder lokal.
"""
from unittest import mock

import pytest

from app import config
from app.config import HF_MODEL_REVISIONS, hub_kwargs

CRYPTER = "crypter70/IndoBERT-Sentiment-Analysis"
INDOBERT = "indobenchmark/indobert-base-p1"


def test_model_dasar_dikunci_ke_revisi_skripsi():
    assert hub_kwargs(CRYPTER) == {"revision": HF_MODEL_REVISIONS[CRYPTER]}
    assert hub_kwargs(INDOBERT) == {"revision": HF_MODEL_REVISIONS[INDOBERT]}


def test_revisi_berupa_sha_commit_lengkap():
    """Nama cabang seperti 'main' bisa bergerak; sha commit tidak."""
    for sha in HF_MODEL_REVISIONS.values():
        assert len(sha) == 40 and all(c in "0123456789abcdef" for c in sha)


@pytest.mark.parametrize("nama", [
    "./models/sentiment_retrained",
    "/opt/nlp-api-service/models/sentiment_retrained",
    "C:\\Skripsi\\nlp-api-service\\models\\sentiment_retrained",
])
def test_folder_lokal_tidak_diberi_revisi(nama):
    assert hub_kwargs(nama) == {}


def test_kandidat_lain_tidak_dikunci():
    """Hanya revisi yang benar-benar dipakai skripsi yang dikunci."""
    assert hub_kwargs("mdhugol/indonesia-bert-sentiment-classification") == {}


def test_saklar_mati_melepas_kunci(monkeypatch):
    monkeypatch.setattr(config.settings, "hf_pin_revisions", False)
    assert hub_kwargs(CRYPTER) == {}


def test_revisi_sampai_ke_from_pretrained_sentimen():
    """
    Diuji pada pemanggilan sungguhan: bobot dasar sentimen harus dimuat dengan
    revisi terkunci. Pemanggilan dicegat sebelum ada unduhan apa pun.
    """
    from app.services import sentiment_service as mod

    dicatat = []

    def cegat(nama, *args, **kwargs):
        dicatat.append((nama, kwargs.get("revision")))
        raise RuntimeError("dicegat oleh tes")

    with mock.patch.object(mod.AutoTokenizer, "from_pretrained", side_effect=cegat):
        svc = mod.SentimentService(model_path=CRYPTER)
        svc._ensure_loaded()

    assert (CRYPTER, HF_MODEL_REVISIONS[CRYPTER]) in dicatat


def test_checkpoint_lokal_sentimen_dimuat_tanpa_revisi(tmp_path):
    from app.services import sentiment_service as mod

    lokal = tmp_path / "sentiment_retrained"
    lokal.mkdir()
    (lokal / "config.json").write_text("{}", encoding="utf-8")
    dicatat = []

    def cegat(nama, *args, **kwargs):
        dicatat.append(("revision" in kwargs, str(nama)))
        raise RuntimeError("dicegat oleh tes")

    with mock.patch.object(mod.AutoTokenizer, "from_pretrained", side_effect=cegat):
        svc = mod.SentimentService(model_path=str(lokal))
        svc._ensure_loaded()

    assert (False, str(lokal)) in dicatat


def test_revisi_sampai_ke_encoder_topik():
    from app.services import topic_service as mod

    dicatat = {}

    def cegat(nama, *args, **kwargs):
        dicatat["nama"], dicatat["revision"] = nama, kwargs.get("revision")
        raise RuntimeError("dicegat oleh tes")

    with mock.patch.object(mod, "SentenceTransformer", side_effect=cegat):
        svc = mod.TopicService()
        svc._ensure_loaded()

    assert dicatat == {"nama": INDOBERT, "revision": HF_MODEL_REVISIONS[INDOBERT]}


def test_transformers_dan_hf_download_berbagi_satu_cache():
    """
    transformers harus mencari model di cache yang sama dengan `hf download`
    (huggingface_hub). Kalau tidak, model dasar yang diunduh lebih dulu dengan
    revisi terkunci tidak pernah terbaca - kegagalan yang terukur di VPS.
    """
    import os
    from huggingface_hub import constants
    from transformers.utils import hub as tf_hub

    assert os.path.normpath(tf_hub.TRANSFORMERS_CACHE) == os.path.normpath(constants.HF_HUB_CACHE)
