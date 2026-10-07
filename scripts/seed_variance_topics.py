"""
Ukur varians hasil topic modeling lintas seed acak.

Seluruh angka pada dokumen metodologi berasal dari satu kali jalan dengan seed
tetap 42. Penguji yang teliti akan menanyakan apakah selisih antar-konfigurasi
bermakna atau sekadar derau. Skrip ini menjalankan pipeline yang sama dengan
beberapa seed dan melaporkan rata-rata serta simpangan bakunya.

UMAP bersifat stokastik; dengan `random_state` tetap hasilnya dapat direproduksi,
tetapi nilai seed yang berbeda menghasilkan proyeksi berbeda - dan itulah sumber
varians yang perlu dilaporkan.

Contoh:
    python scripts/seed_variance_topics.py --seeds 42,7,2024
"""

import argparse
import asyncio
import json
import os
import statistics
import sys
from typing import Any, Dict, List

import torch  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings  # noqa: E402
from app.services.topic_service import TopicService  # noqa: E402

LARAVEL_CONFIG = {
    'case_folding': True, 'remove_punctuation': True, 'remove_numbers': False,
    'remove_stopwords': True, 'stemming': True, 'lemmatization': False,
    'custom_stopwords': [],
}


def load_texts(path: str, limit: int) -> List[str]:
    with open(path, encoding='utf-8') as handle:
        raw = json.load(handle)
    if isinstance(raw, dict):
        raw = raw.get('data') or list(raw.values())[0]
    texts = [i.get('text') if isinstance(i, dict) else str(i) for i in raw]
    texts = [t for t in texts if t and str(t).strip()]
    return texts[:limit] if limit else texts


async def main_async(args: argparse.Namespace) -> None:
    texts = load_texts(args.corpus, args.limit)
    seeds = [int(s) for s in args.seeds.split(',')]
    print(f'{len(texts)} dokumen | seed: {seeds} | num_topics={args.num_topics}\n')

    service = TopicService()
    rows: List[Dict[str, Any]] = []

    header = f"{'seed':>6} {'topik':>6} {'c_v':>8} {'c_npmi':>8} {'divers':>8} {'outlier':>8}"
    print(header)
    print('-' * len(header))

    for seed in seeds:
        # Seed dibaca ulang oleh UMAP/HDBSCAN/KMeans pada tiap pemanggilan.
        settings.topic_seed = seed
        result = await service.analyze(
            texts=texts, preprocessing_config=LARAVEL_CONFIG,
            num_topics=args.num_topics,
        )
        quality = result.get('quality', {})
        row = {
            'seed': seed,
            'num_topics': result.get('num_topics', 0),
            'c_v': quality.get('c_v', 0.0),
            'c_npmi': quality.get('c_npmi', 0.0),
            'diversity': quality.get('diversity', 0.0),
            'outlier_rate': quality.get('outlier_rate', 0.0),
        }
        rows.append(row)
        print(f"{seed:>6} {row['num_topics']:>6} {row['c_v']:>8.4f} "
              f"{row['c_npmi']:>8.4f} {row['diversity']:>8.4f} "
              f"{row['outlier_rate']:>8.4f}", flush=True)

    print('\nrata-rata ± simpangan baku:')
    ringkasan: Dict[str, Any] = {}
    for key in ('num_topics', 'c_v', 'c_npmi', 'diversity', 'outlier_rate'):
        values = [r[key] for r in rows]
        mean = statistics.mean(values)
        stdev = statistics.stdev(values) if len(values) > 1 else 0.0
        ringkasan[key] = {'mean': round(mean, 4), 'stdev': round(stdev, 4)}
        print(f"  {key:14} {mean:8.4f} ± {stdev:.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as handle:
        json.dump({'runs': rows, 'summary': ringkasan}, handle,
                  ensure_ascii=False, indent=2)
    print(f'\ntersimpan ke {args.out}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', default='data/external/youtube/test_gold.json')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--seeds', default='42,7,2024,123,999')
    parser.add_argument('--num-topics', type=int, default=0)
    parser.add_argument('--out', default='data/experiments/topic_seed_variance.json')
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == '__main__':
    main()
