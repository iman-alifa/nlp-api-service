"""
Uji ketahanan modul topic modeling lintas DOMAIN dan lintas UKURAN korpus.

Parameter modul disetel pada satu korpus (885 komentar YouTube) dan satu ukuran.
Klaim bahwa modul "bekerja untuk kasus apa pun" tidak bisa didasarkan pada itu.
Skrip ini menjalankan pipeline produksi yang sama pada beberapa domain dan
beberapa ukuran, lalu melaporkan metrik yang sama untuk tiap kombinasi.

Yang diperiksa bukan hanya coherence, tetapi juga tanda-tanda degenerasi:
- topik terbesar menelan berapa persen korpus (satu topik = seluruh korpus),
- berapa dokumen yang tidak masuk topik mana pun,
- apakah jumlah topik masuk akal untuk ukuran korpusnya.

Contoh:
    python scripts/robustness_topics.py
    python scripts/robustness_topics.py --sizes 60,200 --domains youtube,berita
"""

import argparse
import asyncio
import json
import os
import sys
import time
from typing import Any, Dict, List

import torch  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.topic_service import TopicService  # noqa: E402

LARAVEL_CONFIG = {
    'case_folding': True, 'remove_punctuation': True, 'remove_numbers': False,
    'remove_stopwords': True, 'stemming': True, 'lemmatization': False,
    'custom_stopwords': [],
}

# Tiga domain yang benar-benar berbeda ragam bahasanya: komentar media sosial
# (pendek, tidak baku), artikel berita (panjang, baku), dan ulasan produk/hotel
# (menengah, banyak istilah aspek).
DOMAINS = {
    'youtube': 'data/external/youtube/test_gold.json',
    'berita': 'data/aspect_news_retrain.json',
    'ulasan': 'data/external/terma/train_gold.json',
}


def load_texts(path: str) -> List[str]:
    with open(path, encoding='utf-8') as handle:
        raw = json.load(handle)
    if isinstance(raw, dict):
        raw = raw.get('data') or list(raw.values())[0]
    texts = [i.get('text') if isinstance(i, dict) else str(i) for i in raw]
    return [t for t in texts if t and str(t).strip()]


async def evaluate(service: TopicService, texts: List[str],
                   num_topics: int) -> Dict[str, Any]:
    started = time.time()
    result = await service.analyze(
        texts=texts, preprocessing_config=LARAVEL_CONFIG, num_topics=num_topics
    )
    elapsed = time.time() - started

    topics = result.get('topics', [])
    quality = result.get('quality', {})
    largest = max((t.get('proportion', 0) for t in topics), default=1.0)

    return {
        'num_topics': len(topics),
        'c_v': quality.get('c_v', 0.0),
        'diversity': quality.get('diversity', 0.0),
        'outlier_rate': quality.get('outlier_rate', 1.0),
        'largest_share': round(largest, 4),
        'seconds': round(elapsed, 1),
        'aligned': len(result.get('document_topics', [])) == len(texts),
        'top_words': ', '.join(topics[0].get('words', [])[:6]) if topics else '',
    }


def verdict(row: Dict[str, Any]) -> str:
    """Tandai hasil yang degenerate, bukan sekadar melaporkan angkanya."""
    if not row['aligned']:
        return 'PENJAJARAN RUSAK'
    if row['num_topics'] < 2:
        return 'GAGAL: <2 topik'
    if row['largest_share'] > 0.50:
        return f"TIMPANG: 1 topik {row['largest_share']:.0%}"
    if row['outlier_rate'] > 0.25:
        return f"CAKUPAN: outlier {row['outlier_rate']:.0%}"
    return 'ok'


async def main_async(args: argparse.Namespace) -> None:
    sizes = [int(s) for s in args.sizes.split(',')]
    domains = args.domains.split(',')

    service = TopicService()
    results: List[Dict[str, Any]] = []

    header = (f"{'domain':9} {'n':>5} {'topik':>5} {'c_v':>7} {'divers':>7} "
              f"{'outlier':>7} {'terbesar':>8} {'detik':>6}  keterangan")
    print(header)
    print('-' * len(header))

    for name in domains:
        path = DOMAINS.get(name)
        if not path or not os.path.isfile(path):
            print(f'{name}: korpus tidak ditemukan ({path})')
            continue
        corpus = load_texts(path)

        for n in sizes:
            if n > len(corpus):
                continue
            try:
                row = await evaluate(service, corpus[:n], args.num_topics)
            except Exception as exc:                    # noqa: BLE001
                print(f"{name:9} {n:>5}  gagal: {type(exc).__name__}: {str(exc)[:45]}")
                continue
            row['domain'] = name
            row['n'] = n
            row['verdict'] = verdict(row)
            results.append(row)
            print(f"{name:9} {n:>5} {row['num_topics']:>5} {row['c_v']:>7.4f} "
                  f"{row['diversity']:>7.3f} {row['outlier_rate']:>7.3f} "
                  f"{row['largest_share']:>8.3f} {row['seconds']:>6.0f}  "
                  f"{row['verdict']}", flush=True)

    gagal = [r for r in results if r['verdict'] != 'ok']
    print(f"\n{len(results) - len(gagal)}/{len(results)} kombinasi lolos")
    for r in gagal:
        print(f"  {r['domain']} n={r['n']}: {r['verdict']}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    print(f'\ntersimpan ke {args.out}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes', default='60,120,250,500,900')
    parser.add_argument('--domains', default='youtube,berita,ulasan')
    parser.add_argument('--num-topics', type=int, default=0,
                        help='0 = pilih otomatis berdasarkan coherence')
    parser.add_argument('--out', default='data/experiments/topic_robustness.json')
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == '__main__':
    main()
