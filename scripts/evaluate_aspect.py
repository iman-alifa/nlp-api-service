"""
Evaluasi checkpoint ekstraksi aspek pada set uji berlabel emas.

Dipakai untuk menjawab pertanyaan yang tidak bisa dijawab oleh metrik retraining
sendiri: apakah model benar-benar mempelajari POLA ekstraksi aspek, atau sekadar
menghafal daftar istilah yang muncul di data latihnya?

Caranya: latih pada satu domain, evaluasi pada domain lain (mis. latih TermA
ulasan hotel -> uji KEPS tweet perbankan, atau sebaliknya). Selisih skornya
adalah bukti generalisasi yang bisa dilaporkan di skripsi.

Dua metrik dilaporkan:
  token-level : precision/recall/F1 per token aspek (longgar, sebanding dengan
                angka yang dilaporkan proses retraining)
  span-level  : span dihitung benar hanya bila batas awal & akhirnya persis sama
                dengan anotasi manusia (ketat, ini yang lazim di literatur ABSA)

Pemakaian:
    python scripts/evaluate_aspect.py --gold data/external/terma/test_gold.json
    python scripts/evaluate_aspect.py --gold data/external/keps/test_gold.json \
        --checkpoint models/aspect_retrained.pt
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Set, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from app.services.aspect_service import AspectService  # noqa: E402
from app.utils.training import token_f1  # noqa: E402


def predict_spans(svc: AspectService, text: str) -> Tuple[Set[Tuple[int, int]], List[int]]:
    """
    Jalankan model pada satu teks, kembalikan span prediksi (offset karakter)
    dan label per token.
    """
    enc = svc.tokenizer(
        text, return_offsets_mapping=True, truncation=True,
        max_length=128, return_tensors='pt'
    )
    offsets = enc.pop('offset_mapping')[0].tolist()

    with torch.no_grad():
        out = svc.model(
            input_ids=enc['input_ids'].to(svc.device),
            attention_mask=enc['attention_mask'].to(svc.device),
        )
        preds = torch.argmax(out.logits, dim=-1)[0].cpu().tolist()

    spans: Set[Tuple[int, int]] = set()
    cur: List[int] = []
    for label, (s, e) in zip(preds, offsets):
        if s == 0 and e == 0:
            continue
        if label == 1:                      # B-ASPECT
            if cur:
                spans.add((cur[0], cur[1]))
            cur = [s, e]
        elif label == 2 and cur:            # I-ASPECT
            cur[1] = e
        else:
            if cur:
                spans.add((cur[0], cur[1]))
            cur = []
    if cur:
        spans.add((cur[0], cur[1]))

    return spans, preds


def gold_token_labels(svc: AspectService, text: str, spans: List[Dict[str, Any]]) -> List[int]:
    """Label BIO acuan, diturunkan dari offset anotator (bukan pencocokan string)."""
    return svc._generate_bio_tags(
        text, [], char_spans=[(s['start'], s['end']) for s in spans]
    )


def prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = (2 * p * r / (p + r)) if (p + r) else 0.0
    return {'precision': round(p, 4), 'recall': round(r, 4), 'f1': round(f, 4)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--gold', required=True, help='berkas *_gold.json')
    ap.add_argument('--checkpoint', default=None,
                    help='path checkpoint .pt; default = yang aktif di settings')
    ap.add_argument('--limit', type=int, default=0, help='batasi jumlah dokumen (0 = semua)')
    args = ap.parse_args()

    with open(args.gold, encoding='utf-8') as f:
        gold_docs = json.load(f)
    if args.limit:
        gold_docs = gold_docs[:args.limit]

    svc = AspectService(model_path=args.checkpoint) if args.checkpoint else AspectService()
    svc._ensure_loaded()

    if svc.model is None:
        raise SystemExit('model tidak berhasil dimuat')

    svc.model.eval()

    tp = fp = fn = 0
    y_true: List[List[int]] = []
    y_pred: List[List[int]] = []

    for doc in gold_docs:
        text = doc['text']
        gold = {(s['start'], s['end']) for s in doc['spans']}

        pred, pred_labels = predict_spans(svc, text)
        true_labels = gold_token_labels(svc, text, doc['spans'])

        n = min(len(true_labels), len(pred_labels))
        y_true.append(true_labels[:n])
        y_pred.append(pred_labels[:n])

        tp += len(gold & pred)
        fp += len(pred - gold)
        fn += len(gold - pred)

    span_metrics = prf(tp, fp, fn)
    tok_metrics = token_f1(y_true, y_pred, ignore_index=-100, positive_labels=(1, 2))

    checkpoint = args.checkpoint or svc._model_path or '(model dasar, belum dilatih)'

    print('=' * 62)
    print('EVALUASI EKSTRAKSI ASPEK')
    print('=' * 62)
    print(f'  set uji     : {args.gold}')
    print(f'  dokumen     : {len(gold_docs)}')
    print(f'  checkpoint  : {checkpoint}')
    print(f'  model_trained: {svc.model_trained}')
    print()
    print(f'  span-level (batas persis)')
    print(f'      precision {span_metrics["precision"]:.4f}  '
          f'recall {span_metrics["recall"]:.4f}  F1 {span_metrics["f1"]:.4f}')
    print(f'      TP {tp}   FP {fp}   FN {fn}')
    print()
    print(f'  token-level')
    print(f'      precision {tok_metrics["precision"]:.4f}  '
          f'recall {tok_metrics["recall"]:.4f}  F1 {tok_metrics["f1"]:.4f}')


if __name__ == '__main__':
    main()
