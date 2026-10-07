"""
Siapkan dataset IOB eksternal (IndoNLU TermA / KEPS) untuk fine-tuning aspek.

Berbeda dari dua builder lain di folder ini, dataset ini **berlabel manusia**,
sehingga dipakai sebagai baseline dan sebagai alat ukur generalisasi: latih pada
satu domain, uji pada domain lain. Anotasi hasil proyeksi leksikon tidak bisa
membuktikan itu karena labelnya berasal dari fungsi yang deterministik.

Format sumber: satu token per baris, `token<TAB>label`, dokumen dipisah baris
kosong (gaya CoNLL).
    TermA : O / B-ASPECT / I-ASPECT / B-SENTIMENT / I-SENTIMENT
    KEPS  : O / B / I   (keyphrase; dipetakan ke ASPECT)

Keluaran:
  *_gold.json    - teks + span aspek beserta offset karakter (lossless, dipakai
                   untuk evaluasi dan untuk jalur latih BIO langsung)
  *_retrain.json - payload {text, aspects} untuk POST /api/retrain/aspect

Laporan fidelitas menunjukkan berapa span emas yang tetap pulih ketika BIO
diturunkan ulang dari daftar string. Angka ini penting: bila rendah, latihlah
lewat jalur gold agar label manusia tidak rusak.

Pemakaian:
    python scripts/prepare_external_dataset.py --dataset terma
    python scripts/prepare_external_dataset.py --dataset keps
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATASETS = {
    'terma': {
        'dir': 'data/external/terma',
        'aspect_labels': ('B-ASPECT', 'I-ASPECT'),
        'begin': 'B-ASPECT',
    },
    'keps': {
        'dir': 'data/external/keps',
        'aspect_labels': ('B', 'I'),
        'begin': 'B',
    },
}
SPLITS = ('train', 'valid', 'test')


def read_conll(path: str) -> List[Tuple[List[str], List[str]]]:
    """Baca berkas token-per-baris menjadi daftar (tokens, labels)."""
    docs: List[Tuple[List[str], List[str]]] = []
    tokens: List[str] = []
    labels: List[str] = []

    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line.strip():
                if tokens:
                    docs.append((tokens, labels))
                    tokens, labels = [], []
                continue
            parts = line.split('\t')
            if len(parts) < 2:
                continue
            tokens.append(parts[0])
            labels.append(parts[1].strip())

    if tokens:
        docs.append((tokens, labels))
    return docs


def to_spans(
    tokens: List[str],
    labels: List[str],
    aspect_labels: Tuple[str, ...],
    begin: str,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Gabungkan token menjadi teks dan ubah label IOB menjadi span offset karakter.

    Returns:
        Tuple (text, spans) dengan span berisi start, end, dan surface.
    """
    text_parts: List[str] = []
    offsets: List[Tuple[int, int]] = []
    cursor = 0

    for tok in tokens:
        if text_parts:
            cursor += 1  # spasi pemisah
        offsets.append((cursor, cursor + len(tok)))
        cursor += len(tok)
        text_parts.append(tok)

    text = ' '.join(text_parts)

    spans: List[Dict[str, Any]] = []
    current: List[int] = []

    def flush() -> None:
        if not current:
            return
        start = offsets[current[0]][0]
        end = offsets[current[-1]][1]
        spans.append({'start': start, 'end': end, 'surface': text[start:end]})
        current.clear()

    for i, label in enumerate(labels):
        if label == begin:
            flush()
            current.append(i)
        elif label in aspect_labels and current:
            current.append(i)
        elif label in aspect_labels:
            # I-* tanpa B-* sebelumnya: perlakukan sebagai awal span baru
            flush()
            current.append(i)
        else:
            flush()
    flush()

    return text, spans


def fidelity(text: str, spans: List[Dict[str, Any]]) -> Tuple[int, int]:
    """
    Berapa span emas yang tetap dapat dipulihkan bila BIO diturunkan ulang dari
    daftar string (jalur {text, aspects}).

    Kehilangan terjadi ketika kata yang sama muncul beberapa kali tetapi hanya
    sebagian yang berlabel aspek: pencocokan string akan menandai semuanya.

    Returns:
        Tuple (jumlah_span_emas, jumlah_span_yang_akan_dihasilkan_ulang).
    """
    import re

    surfaces = [s['surface'] for s in spans]
    produced = 0
    for surf in set(surfaces):
        escaped = r'\s+'.join(re.escape(w) for w in surf.split())
        produced += len(list(re.finditer(rf'(?<!\w){escaped}(?!\w)', text, re.IGNORECASE)))
    return len(spans), produced


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', required=True, choices=sorted(DATASETS))
    ap.add_argument('--out-dir', default='data/external')
    args = ap.parse_args()

    cfg = DATASETS[args.dataset]
    out_base = os.path.join(args.out_dir, args.dataset)
    os.makedirs(out_base, exist_ok=True)

    # Split TermA yang didistribusikan IndoNLU memuat SELURUH dokumen valid di
    # dalam train (996/996). Bila dibiarkan, F1 validasi diukur pada data latih
    # sendiri sehingga tidak berarti, dan pemilihan epoch terbaik selalu jatuh
    # ke epoch terakhir. Teks valid/test karena itu dibuang dari train.
    holdout_texts = set()
    for other in ('valid', 'test'):
        other_path = os.path.join(cfg['dir'], f'{other}_preprocess.txt')
        if os.path.exists(other_path):
            for tokens, labels in read_conll(other_path):
                holdout_texts.add(to_spans(tokens, labels,
                                           cfg['aspect_labels'], cfg['begin'])[0])

    print('=' * 68)
    print(f'DATASET {args.dataset.upper()}  (IndoNLU, berlabel manusia)')
    print('=' * 68)
    print(f"{'split':8}{'dok':>7}{'span':>8}{'unik':>8}{'kosong':>9}{'fidelitas':>12}")
    print('-' * 52)

    totals = {'docs': 0, 'spans': 0}

    for split in SPLITS:
        path = os.path.join(cfg['dir'], f'{split}_preprocess.txt')
        if not os.path.exists(path):
            print(f'{split:8}  (berkas tidak ada: {path})')
            continue

        docs = read_conll(path)
        removed = 0
        gold: List[Dict[str, Any]] = []
        uniq = set()
        n_spans = 0
        empty = 0
        fid_gold = fid_prod = 0

        for tokens, labels in docs:
            text, spans = to_spans(tokens, labels, cfg['aspect_labels'], cfg['begin'])
            if split == 'train' and text in holdout_texts:
                removed += 1
                continue
            if not spans:
                empty += 1
            n_spans += len(spans)
            for s in spans:
                uniq.add(s['surface'].lower())
            g, p = fidelity(text, spans)
            fid_gold += g
            fid_prod += p
            gold.append({'text': text, 'spans': spans})

        # Lossless: offset dipertahankan
        with open(f'{out_base}/{split}_gold.json', 'w', encoding='utf-8') as f:
            json.dump(gold, f, ensure_ascii=False, indent=1)

        # Payload endpoint retraining. 'spans' disertakan agar AspectService
        # memakai offset anotator apa adanya; tanpa itu BIO diturunkan ulang
        # lewat pencocokan string dan ~5% label manusia rusak.
        payload = [{'text': g['text'],
                    'aspects': [s['surface'] for s in g['spans']],
                    'spans': [{'start': s['start'], 'end': s['end']} for s in g['spans']]}
                   for g in gold if g['spans']]
        with open(f'{out_base}/{split}_retrain.json', 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=1)

        ratio = (fid_prod / fid_gold * 100) if fid_gold else 0.0
        suffix = f'  (-{removed} tumpang tindih dibuang)' if removed else ''
        print(f'{split:8}{len(gold):>7}{n_spans:>8}{len(uniq):>8}{empty:>9}'
              f'{ratio:>11.1f}%{suffix}')

        totals['docs'] += len(gold)
        totals['spans'] += n_spans

    print('-' * 52)
    print(f"{'total':8}{totals['docs']:>7}{totals['spans']:>8}")
    print()
    print(f'  -> {out_base}/<split>_gold.json     (offset utuh, untuk evaluasi)')
    print(f'  -> {out_base}/<split>_retrain.json  (payload /api/retrain/aspect)')
    print()
    print('  fidelitas = span yang dihasilkan ulang dari daftar string dibanding')
    print('  span emas. >100% berarti pencocokan string menandai kemunculan yang')
    print('  oleh anotator manusia sengaja TIDAK dilabeli.')


if __name__ == '__main__':
    main()
