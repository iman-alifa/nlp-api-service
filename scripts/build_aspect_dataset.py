"""
Bangun data latih ekstraksi aspek (BIO) dari korpus mentah + leksikon terkurasi.

Mengapa dibuat begini, bukan memakai anotasi LLM:

1. Konsistensi. Pelabelan dilakukan deterministik, sehingga setiap kemunculan
   istilah yang sama selalu ditandai. Anotasi LLM pada berkas Label Studio
   melewatkan 56% kemunculan frasa yang di kalimat lain justru ditandai —
   supervisi yang saling bertentangan seperti itu merusak token classification.
2. Ketertelusuran. Setiap span berasal dari istilah pada data/aspect_lexicon.json
   yang provenance-nya tercatat (human = hasil koreksi manusia di aplikasi).
3. Kesesuaian domain. Korpusnya komentar yang benar-benar dianalisis sistem ini.

Yang TIDAK dijamin: recall. Aspek di luar leksikon tidak akan tertandai. Karena
itu keluaran Label Studio disertakan agar sisa anotasi bisa dilengkapi manusia.

Pemakaian:
    python scripts/build_aspect_dataset.py --corpus data/corpus.json --out data/aspect_dataset
"""

import argparse
import json
import os
import sys
from collections import Counter
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.aspect_service import AspectService  # noqa: E402


def load_lexicon(path: str) -> List[Dict[str, Any]]:
    """Baca leksikon, abaikan kunci dokumentasi yang diawali underscore."""
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    return data['aspects']


def resolve_overlaps(spans: List[Tuple[int, int, str]]) -> List[Tuple[int, int, str]]:
    """
    Sisakan span terpanjang bila ada yang tumpang tindih.

    'pegawai pajak' dan 'pajak' sama-sama ada di leksikon; tanpa penyelesaian ini
    keduanya tertandai dan batas BIO-nya menjadi rancu.
    """
    ordered = sorted(spans, key=lambda s: (-(s[1] - s[0]), s[0]))
    kept: List[Tuple[int, int, str]] = []

    for start, end, term in ordered:
        if any(not (end <= k_start or start >= k_end) for k_start, k_end, _ in kept):
            continue
        kept.append((start, end, term))

    return sorted(kept, key=lambda s: s[0])


def annotate(
    svc: AspectService,
    texts: List[str],
    lexicon: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Counter]:
    """Tandai setiap teks dengan span aspek dari leksikon."""
    literal = [a['term'] for a in lexicon if a.get('literal_only')]
    stemmed = [a['term'] for a in lexicon if not a.get('literal_only')]

    records: List[Dict[str, Any]] = []
    surface_freq: Counter = Counter()

    for text in texts:
        found: List[Tuple[int, int, str]] = []

        for term in stemmed:
            for s, e in svc._find_aspect_spans(text, [term], use_stemming=True)[0]:
                found.append((s, e, term))
        for term in literal:
            for s, e in svc._find_aspect_spans(text, [term], use_stemming=False)[0]:
                found.append((s, e, term))

        spans = resolve_overlaps(found)
        for s, e, term in spans:
            surface_freq[text[s:e].lower()] += 1

        records.append({
            'text': text,
            'spans': [{'start': s, 'end': e, 'surface': text[s:e], 'term': t}
                      for s, e, t in spans],
        })

    return records, surface_freq


def to_label_studio(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Ekspor untuk ditinjau/dilengkapi manusia di Label Studio."""
    out = []
    for rec in records:
        out.append({
            'data': {'text': rec['text']},
            'predictions': [{
                'model_version': 'lexicon_v1_human_seeded',
                'result': [{
                    'from_name': 'label',
                    'to_name': 'text',
                    'type': 'labels',
                    'value': {
                        'start': sp['start'],
                        'end': sp['end'],
                        'text': sp['surface'],
                        'labels': ['ASPECT'],
                    },
                } for sp in rec['spans']],
            }],
        })
    return out


def to_retrain_payload(records: List[Dict[str, Any]], only_with_aspects: bool) -> List[Dict[str, Any]]:
    """
    Payload untuk POST /api/retrain/aspect: [{text, aspects: [...]}].

    Kalimat tanpa aspek tetap berguna sebagai contoh negatif, tapi bisa
    dikeluarkan bila proporsinya terlalu besar.
    """
    out = []
    for rec in records:
        surfaces = [sp['surface'] for sp in rec['spans']]
        if only_with_aspects and not surfaces:
            continue
        out.append({'text': rec['text'], 'aspects': surfaces})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', required=True, help='JSON berisi list teks')
    parser.add_argument('--lexicon', default='data/aspect_lexicon.json')
    parser.add_argument('--out', default='data/aspect_dataset')
    parser.add_argument('--include-empty', action='store_true',
                        help='sertakan kalimat tanpa aspek sebagai contoh negatif')
    args = parser.parse_args()

    with open(args.corpus, encoding='utf-8') as f:
        texts = json.load(f)
    texts = [t.strip() for t in texts if isinstance(t, str) and t.strip()]

    lexicon = load_lexicon(args.lexicon)
    svc = AspectService()

    records, surface_freq = annotate(svc, texts, lexicon)

    n_spans = sum(len(r['spans']) for r in records)
    covered = sum(1 for r in records if r['spans'])

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)

    ls_path = f'{args.out}_label_studio.json'
    with open(ls_path, 'w', encoding='utf-8') as f:
        json.dump(to_label_studio(records), f, ensure_ascii=False, indent=1)

    payload = to_retrain_payload(records, only_with_aspects=not args.include_empty)
    rt_path = f'{args.out}_retrain.json'
    with open(rt_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)

    human = sum(1 for a in lexicon if a.get('provenance') == 'human')

    print('=' * 64)
    print('DATA LATIH EKSTRAKSI ASPEK')
    print('=' * 64)
    print(f'  korpus                : {len(texts)} teks')
    print(f'  leksikon              : {len(lexicon)} istilah '
          f'({human} human, {len(lexicon) - human} expansion)')
    print(f'  teks dengan aspek     : {covered} ({covered / len(texts) * 100:.1f}%)')
    print(f'  total span            : {n_spans} '
          f'(rata-rata {n_spans / max(1, covered):.2f} per teks beraspek)')
    print(f'  bentuk permukaan unik : {len(surface_freq)}')
    print(f'  sampel untuk retrain  : {len(payload)}')
    print()
    print(f'  -> {ls_path}   (tinjau/lengkapi di Label Studio)')
    print(f'  -> {rt_path}   (payload POST /api/retrain/aspect)')
    print()
    print('  15 bentuk permukaan tersering:')
    for surface, count in surface_freq.most_common(15):
        print(f'      {surface:24} {count}')


if __name__ == '__main__':
    main()
