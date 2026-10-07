"""
Uji langsung: apakah model menandai aspek berdasarkan POLA atau HAFALAN kata?

Metrik agregat (F1 lintas domain) menunjukkan gejala, tetapi tidak menunjukkan
mekanismenya. Uji ini menyerang pertanyaannya secara langsung dengan kalimat
buatan yang strukturnya identik, hanya kata bendanya diganti:

  seen    : kata yang sering muncul sebagai aspek di data latih
  unseen  : nomina Indonesia yang wajar tetapi TIDAK PERNAH menjadi aspek
            di data latih mana pun
  nonce   : kata yang tidak ada dalam bahasa Indonesia

Model yang menghafal hanya akan menandai kelompok "seen". Model yang membaca
pola akan menandai ketiganya, karena posisi sintaktisnya sama.

Pemakaian:
    python scripts/probe_pattern.py --checkpoint models_exp/k4_combined/aspect_retrained.pt
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Set, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from app.services.aspect_service import AspectService  # noqa: E402

# Pola kalimat: satu slot nomina yang jelas menjadi sasaran penilaian.
TEMPLATES = [
    "{} nya sangat mengecewakan .",
    "saya tidak puas dengan {} di tempat ini .",
    "{} tersebut perlu segera diperbaiki .",
    "menurut saya {} sudah cukup bagus .",
    "banyak keluhan mengenai {} akhir - akhir ini .",
]

SEEN = ["kamar", "wifi", "pelayanan", "harga", "lokasi"]
UNSEEN = ["jembatan", "kurikulum", "vaksin", "irigasi", "kompos"]
NONCE = ["blarum", "kentrasi", "molusa", "trapida", "senggula"]


def tag(svc: AspectService, text: str) -> List[Tuple[int, int]]:
    """Kembalikan span aspek yang diprediksi model."""
    enc = svc.tokenizer(text, return_offsets_mapping=True, truncation=True,
                        max_length=128, return_tensors='pt')
    offsets = enc.pop('offset_mapping')[0].tolist()
    with torch.no_grad():
        out = svc.model(input_ids=enc['input_ids'].to(svc.device),
                        attention_mask=enc['attention_mask'].to(svc.device))
        preds = torch.argmax(out.logits, dim=-1)[0].cpu().tolist()

    spans, cur = [], []
    for label, (s, e) in zip(preds, offsets):
        if s == 0 and e == 0:
            continue
        if label == 1:
            if cur:
                spans.append(tuple(cur))
            cur = [s, e]
        elif label == 2 and cur:
            cur[1] = e
        else:
            if cur:
                spans.append(tuple(cur))
            cur = []
    if cur:
        spans.append(tuple(cur))
    return spans


def hit_rate(svc: AspectService, words: List[str]) -> Tuple[float, List[str]]:
    """Berapa persen slot yang benar-benar ditandai model."""
    hits = 0
    total = 0
    contoh: List[str] = []
    for word in words:
        for tpl in TEMPLATES:
            text = tpl.format(word)
            slot_start = tpl.index('{}')
            slot_end = slot_start + len(word)
            spans = tag(svc, text)
            total += 1
            # dianggap kena bila ada span yang beririsan dengan slot nomina
            if any(not (e <= slot_start or s >= slot_end) for s, e in spans):
                hits += 1
            elif len(contoh) < 3:
                contoh.append(text)
    return hits / total * 100 if total else 0.0, contoh


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoint', default=None)
    ap.add_argument('--label', default=None)
    args = ap.parse_args()

    svc = AspectService(model_path=args.checkpoint) if args.checkpoint else AspectService()
    svc._ensure_loaded()
    if svc.model is None:
        raise SystemExit('model gagal dimuat')
    svc.model.eval()

    name = args.label or (args.checkpoint or 'model dasar')
    print('=' * 62)
    print(f'UJI POLA  |  {name}')
    print('=' * 62)
    print(f'  {len(TEMPLATES)} pola kalimat x 5 kata per kelompok = '
          f'{len(TEMPLATES) * 5} kalimat per kelompok')
    print()

    results: Dict[str, float] = {}
    for group, words in (('seen', SEEN), ('unseen', UNSEEN), ('nonce', NONCE)):
        rate, missed = hit_rate(svc, words)
        results[group] = rate
        print(f'  {group:8} {rate:6.1f}%  ({words[0]}, {words[1]}, ...)')
        for m in missed:
            print(f'             tidak ditandai: "{m}"')

    print()
    gap = results['seen'] - results['unseen']
    print(f'  selisih seen - unseen : {gap:+.1f} poin')

    # Selisih saja tidak cukup. Model yang hampir tidak pernah menembak akan
    # menghasilkan selisih nol pada tingkat sama-sama rendah, dan itu BUKAN
    # tanda membaca pola - ia hanya diam. Tingkat absolut harus diperiksa dulu.
    if results['seen'] < 50:
        print('  -> model jarang menandai apa pun pada pola ini '
              f"(seen hanya {results['seen']:.0f}%).")
        print('     Selisih kecil di sini TIDAK berarti membaca pola; model')
        print('     kemungkinan besar hanya mengenali kosakata domainnya sendiri.')
    elif gap <= 15 and results['nonce'] >= 50:
        print('  -> menandai kata tak dikenal DAN kata bukan-bahasa-Indonesia')
        print('     hampir sama seringnya: bukti langsung pembacaan POLA.')
    elif gap <= 15:
        print('  -> konsisten pada nomina wajar, tetapi lemah pada kata bentukan:')
        print('     sebagian pola, sebagian kemiripan bentuk kata.')
    elif gap <= 40:
        print('  -> sebagian bergantung identitas kata, sebagian pola.')
    else:
        print('  -> hanya menandai kata yang dikenalnya: perilaku MENGHAFAL.')

    print(json.dumps(results))


if __name__ == '__main__':
    main()
