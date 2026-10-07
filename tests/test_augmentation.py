"""
Augmentasi substitusi istilah.

Offset harus tetap sejajar setelah teks berubah panjang; kalau meleset, seluruh
label rusak tanpa terlihat.
"""

import importlib.util
import os
import random
import sys

import pytest

_spec = importlib.util.spec_from_file_location(
    "augment_aspect_dataset",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "scripts", "augment_aspect_dataset.py"),
)
aug = importlib.util.module_from_spec(_spec)
sys.modules["augment_aspect_dataset"] = aug
_spec.loader.exec_module(aug)


def test_offset_tetap_sejajar_setelah_substitusi():
    text = "kamar mandi kotor dan wifi lemot"
    spans = [(0, 11), (22, 26)]          # 'kamar mandi', 'wifi'
    rng = random.Random(1)

    new_text, new_spans = aug.substitute(text, spans, ['resepsionis', 'ac'], rng)

    for sp in new_spans:
        potongan = new_text[sp['start']:sp['end']]
        assert potongan in ('resepsionis', 'ac'), potongan


def test_konteks_di_luar_span_dipertahankan():
    text = "kamar mandi kotor dan wifi lemot"
    rng = random.Random(1)

    new_text, _ = aug.substitute(text, [(0, 11)], ['ac'], rng)

    assert new_text.endswith('kotor dan wifi lemot')


def test_pengganti_lebih_panjang_menggeser_span_berikutnya():
    text = "ac rusak dan wifi lemot"
    spans = [(0, 2), (13, 17)]           # 'ac', 'wifi'
    rng = random.Random(0)

    new_text, new_spans = aug.substitute(text, spans, ['resepsionis'], rng)

    # span kedua harus menunjuk kata pengganti, bukan posisi lama
    assert new_text[new_spans[1]['start']:new_spans[1]['end']] == 'resepsionis'


def test_span_tidak_terurut_tetap_benar():
    text = "wifi lemot dan ac rusak"
    spans = [(15, 17), (0, 4)]           # sengaja terbalik

    new_text, new_spans = aug.substitute(text, spans, ['kolam'], random.Random(2))

    for sp in new_spans:
        assert new_text[sp['start']:sp['end']] == 'kolam'


def test_tanpa_span_teks_tidak_berubah():
    text = "tidak ada aspek di sini"
    new_text, new_spans = aug.substitute(text, [], ['ac'], random.Random(3))

    assert new_text == text
    assert new_spans == []


def test_hasil_dapat_direproduksi_dengan_seed_sama():
    text = "kamar mandi kotor"
    pool = ['ac', 'wifi', 'kolam', 'resepsionis']

    a = aug.substitute(text, [(0, 11)], pool, random.Random(7))
    b = aug.substitute(text, [(0, 11)], pool, random.Random(7))

    assert a == b
