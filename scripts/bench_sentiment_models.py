"""
Bandingkan model sentimen bahasa Indonesia yang tersedia publik.

Modul ini memakai `mdhugol/indonesia-bert-sentiment-classification` sejak awal
tanpa pernah dibandingkan dengan alternatifnya. Pilihan model adalah keputusan
terbesar yang masih bisa diubah tanpa data berlabel baru, jadi ia layak diukur
alih-alih diasumsikan.

Dua hal diukur, dan yang kedua justru lebih penting untuk klaim generalitas:

1. **Mutu in-domain** pada SmSA test. Sebagian besar kandidat di-fine-tune pada
   SmSA train, sehingga angka ini adalah batas atas untuk semuanya secara adil -
   tidak ada yang diuntungkan.

2. **Kesepakatan antar-model di domain produksi** (komentar YouTube), yang TIDAK
   punya label. Bila beberapa model yang dilatih terpisah sepakat pada sebuah
   dokumen, prediksi itu jauh lebih mungkin benar; tempat mereka berbeda pendapat
   menandai wilayah yang rapuh. Ini cara mengukur ketahanan lintas domain tanpa
   anotasi manusia - bukan pengganti data berlabel, tetapi jauh lebih baik
   daripada tidak mengukur apa pun.

PENTING - urutan label berbeda-beda tiap model. `taufiqdp` memakai
0=negatif sementara `mdhugol` memakai 0=positive. Skrip ini membaca `id2label`
dari config tiap model, bukan mengasumsikan urutan apa pun; salah di sini
membalik seluruh hasil tanpa memunculkan galat.

Contoh:
    python scripts/bench_sentiment_models.py
    python scripts/bench_sentiment_models.py --limit 200
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from typing import Any, Dict, List, Optional

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings  # noqa: E402
from app.preprocessing.text_cleaner import TextCleaner  # noqa: E402
from app.services.sentiment_service import (  # noqa: E402
    LABEL_MAP,
    LABEL_TO_ID,
    SentimentService,
)
from app.utils.training import (  # noqa: E402
    classification_metrics,
    expected_calibration_error,
)

CANDIDATES = [
    'mdhugol/indonesia-bert-sentiment-classification',
    'w11wo/indonesian-roberta-base-sentiment-classifier',
    'ayameRushia/bert-base-indonesian-1.5G-sentiment-analysis-smsa',
    'crypter70/IndoBERT-Sentiment-Analysis',
    'taufiqdp/indonesian-sentiment',
]

LARAVEL_CONFIG = {
    'case_folding': True, 'remove_punctuation': True, 'remove_numbers': False,
    'remove_stopwords': True, 'stemming': True, 'lemmatization': False,
    'custom_stopwords': [],
}

# Bentuk label yang dipakai berbagai model -> label kanonik proyek ini.
_ALIASES = {
    'positive': 'positive', 'positif': 'positive', 'pos': 'positive',
    'neutral': 'neutral', 'netral': 'neutral', 'neu': 'neutral',
    'negative': 'negative', 'negatif': 'negative', 'neg': 'negative',
}


def resolve_label_map(model, model_name: str) -> Dict[int, str]:
    """
    Turunkan peta id -> label kanonik dari config model itu sendiri.

    Model yang config-nya masih `LABEL_0/1/2` (tidak informatif) memakai peta
    milik proyek, yang hanya benar untuk `mdhugol`. Model lain wajib menyebut
    labelnya secara eksplisit, atau dilewati - menebak urutannya berisiko
    membalik seluruh hasil secara diam-diam.
    """
    raw = getattr(model.config, 'id2label', None) or {}
    resolved: Dict[int, str] = {}

    for key, value in raw.items():
        name = _ALIASES.get(str(value).strip().lower())
        if name:
            resolved[int(key)] = name

    if len(resolved) == 3:
        return resolved

    if model_name == settings.sentiment_base_model:
        # Satu-satunya model yang urutan labelnya sudah diverifikasi manual
        # lewat kartu modelnya (LABEL_0=positive, LABEL_1=neutral, LABEL_2=negative).
        return dict(LABEL_MAP)

    raise ValueError(f'{model_name}: id2label tidak informatif ({raw})')


def predict(model_name: str, texts: List[str], batch_size: int = 64):
    """Muat model, prediksi seluruh teks, lalu lepaskan dari memori."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()

    id2label = resolve_label_map(model, model_name)

    labels: List[str] = []
    confidences: List[float] = []
    started = time.time()

    # Diurutkan panjang, sama seperti jalur produksi, lalu dikembalikan ke
    # urutan semula - padding dinamis jauh lebih murah begitu.
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    hasil: List[Optional[tuple]] = [None] * len(texts)

    for i in range(0, len(order), batch_size):
        slots = order[i:i + batch_size]
        batch = tokenizer([texts[k] for k in slots], return_tensors='pt',
                          truncation=True, max_length=512, padding=True)
        with torch.no_grad():
            probs = torch.softmax(model(**batch).logits, dim=-1)
        for j, slot in enumerate(slots):
            idx = int(torch.argmax(probs[j]).item())
            hasil[slot] = (id2label.get(idx, 'neutral'), float(probs[j][idx].item()))

    for label, confidence in hasil:  # type: ignore[misc]
        labels.append(label)
        confidences.append(confidence)

    elapsed = time.time() - started
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return labels, confidences, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gold', default='data/external/smsa/test_gold.json')
    parser.add_argument('--domain', default='data/external/youtube/test_gold.json',
                        help='korpus domain produksi TANPA label, untuk uji kesepakatan')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--domain-limit', type=int, default=300)
    parser.add_argument('--out', default='data/experiments/sentiment_models.json')
    args = parser.parse_args()

    gold = [
        r for r in json.load(open(args.gold, encoding='utf-8'))
        if r.get('text', '').strip() and r.get('label') in LABEL_TO_ID
    ]
    if args.limit:
        gold = gold[:args.limit]

    domain_texts = [
        (r['text'] if isinstance(r, dict) else str(r))
        for r in json.load(open(args.domain, encoding='utf-8'))
    ]
    domain_texts = [t for t in domain_texts if t and t.strip()][:args.domain_limit]

    # Pembersihan identik dengan jalur produksi, supaya perbandingannya adil.
    cleaner = TextCleaner(SentimentService._sanitize_config(LARAVEL_CONFIG))
    gold_texts = cleaner.clean_texts([r['text'] for r in gold])
    domain_clean = cleaner.clean_texts(domain_texts)

    y_true = [LABEL_TO_ID[r['label']] for r in gold]

    print(f'SmSA test: {len(gold)} dokumen | domain produksi: {len(domain_clean)} dokumen\n')
    header = (f"{'model':52} {'acc':>7} {'macroF1':>8} {'ECE':>7} "
              f"{'detik':>6}  distribusi domain (P/N/Neg)")
    print(header)
    print('-' * len(header))

    rows: List[Dict[str, Any]] = []
    domain_preds: Dict[str, List[str]] = {}

    for name in CANDIDATES:
        try:
            labels, confidences, elapsed = predict(name, gold_texts)
            y_pred = [LABEL_TO_ID[label] for label in labels]
            metrics = classification_metrics(y_true, y_pred, LABEL_MAP)
            ece = expected_calibration_error(
                confidences, [p == t for p, t in zip(y_pred, y_true)]
            )

            dom_labels, _, _ = predict(name, domain_clean)
            domain_preds[name] = dom_labels
            counts = Counter(dom_labels)
            total = max(1, len(dom_labels))
            share = '/'.join(
                f'{counts.get(k, 0) / total:.0%}'
                for k in ('positive', 'neutral', 'negative')
            )

            rows.append({
                'model': name, 'accuracy': metrics['accuracy'],
                'macro_f1': metrics['macro_f1'], 'weighted_f1': metrics['weighted_f1'],
                'ece': ece, 'seconds': round(elapsed, 1),
                'per_class': metrics['per_class'],
                'domain_distribution': {k: counts.get(k, 0) for k in
                                        ('positive', 'neutral', 'negative')},
            })
            print(f"{name:52} {metrics['accuracy']:>7.4f} {metrics['macro_f1']:>8.4f} "
                  f"{ece:>7.4f} {elapsed:>6.1f}  {share}", flush=True)

        except Exception as exc:  # noqa: BLE001
            print(f'{name:52} GAGAL: {type(exc).__name__}: {str(exc)[:60]}', flush=True)

    # ── Recall netral: kelemahan utama modul, dibandingkan antar-model ──────
    if rows:
        print(f"\n{'model':52} {'recall netral':>14} {'recall neg':>11} {'recall pos':>11}")
        for row in rows:
            per = row['per_class']
            print(f"{row['model']:52} {per['neutral']['recall']:>14.4f} "
                  f"{per['negative']['recall']:>11.4f} {per['positive']['recall']:>11.4f}")

    # ── Kesepakatan di domain produksi (tanpa label) ────────────────────────
    if len(domain_preds) > 1:
        names = list(domain_preds)
        n = len(domain_clean)
        print(f'\nKesepakatan berpasangan di domain produksi (n={n}):')
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                sama = sum(1 for x, y in zip(domain_preds[a], domain_preds[b]) if x == y)
                print(f'  {sama / n:.3f}  {a.split("/")[-1][:28]:30} vs {b.split("/")[-1][:28]}')

        bulat = sum(
            1 for i in range(n)
            if len({domain_preds[m][i] for m in names}) == 1
        )
        print(f'\n  {bulat}/{n} ({bulat / n:.1%}) dokumen disepakati SELURUH model.')
        print('  Sisanya wilayah rapuh: tempat pilihan model benar-benar berpengaruh.')

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as handle:
        json.dump({'in_domain': rows, 'domain_corpus': args.domain,
                   'domain_n': len(domain_clean)}, handle, ensure_ascii=False, indent=2)
    print(f'\ntersimpan ke {args.out}')


if __name__ == '__main__':
    main()
