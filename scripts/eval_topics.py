"""
Ukur mutu topic modeling pada korpus nyata.

Dipakai untuk membandingkan konfigurasi SEBELUM dan SESUDAH perubahan, memakai
korpus dan seed yang sama, sehingga setiap klaim perbaikan bisa diperiksa ulang.

Contoh:
    python scripts/eval_topics.py --corpus data/external/youtube/test_gold.json
    python scripts/eval_topics.py --limit 200 --num-topics 5 --label baseline

Hasil ditambahkan ke data/experiments/topic_results.json, satu entri per label,
supaya riwayat percobaan tidak tertimpa.
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List

# Impor torch lebih dulu: pada Windows, memuat nltk sebelum torch memicu
# OSError WinError 1114 saat memuat c10.dll. text_cleaner menarik nltk, jadi
# urutan ini yang menjaga skrip tetap bisa dijalankan.
import torch  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.preprocessing.text_cleaner import TextCleaner  # noqa: E402
from app.services.topic_service import TopicService  # noqa: E402
from app.utils.topic_metrics import evaluate  # noqa: E402

RESULTS_PATH = os.path.join('data', 'experiments', 'topic_results.json')

# Konfigurasi yang benar-benar dikirim Laravel (lihat
# ProcessTextAnalysis::getPreprocessingConfig). Mengukur dengan konfigurasi
# lain akan memberi angka yang tidak mewakili produksi.
LARAVEL_CONFIG = {
    'case_folding': True,
    'remove_punctuation': True,
    'remove_numbers': False,
    'remove_stopwords': True,
    'stemming': True,
    'lemmatization': False,
    'custom_stopwords': [],
}


def load_corpus(path: str, limit: int = 0) -> List[str]:
    """Baca korpus dari JSON berisi list objek ber-key `text`, atau list string."""
    with open(path, encoding='utf-8') as handle:
        raw = json.load(handle)

    texts = [
        item['text'] if isinstance(item, dict) else str(item)
        for item in raw
    ]
    texts = [t for t in texts if t and t.strip()]
    return texts[:limit] if limit else texts


def reference_tokens(texts: List[str]) -> List[List[str]]:
    """
    Bangun korpus rujukan untuk coherence.

    Coherence harus dihitung atas teks yang SAMA bentuknya dengan kata-kata
    topik. Kata topik berasal dari teks yang sudah dibersihkan, jadi korpus
    rujukan pun dibersihkan dengan konfigurasi yang sama - kalau tidak, kata
    hasil stemming tidak akan pernah ditemukan di rujukan dan seluruh skor
    jatuh ke nol tanpa alasan yang terlihat.
    """
    cleaner = TextCleaner(LARAVEL_CONFIG)
    return [t.split() for t in cleaner.clean_texts(texts) if t.strip()]


async def run(args: argparse.Namespace) -> Dict[str, Any]:
    texts = load_corpus(args.corpus, args.limit)
    print(f'korpus: {len(texts)} teks dari {args.corpus}')

    service = TopicService()
    started = datetime.now()
    result = await service.analyze(
        texts=texts,
        preprocessing_config=LARAVEL_CONFIG,
        num_topics=args.num_topics,
    )
    elapsed = (datetime.now() - started).total_seconds()

    topics = result.get('topics', [])
    topic_words = [t.get('words_stemmed') or t.get('words', []) for t in topics]
    metrics = evaluate(
        topic_words=topic_words,
        tokenized_docs=reference_tokens(texts),
        document_topics=result.get('document_topics', []),
        top_n=args.top_n,
    )
    metrics['seconds'] = round(elapsed, 1)
    metrics['requested_topics'] = args.num_topics
    metrics['num_texts'] = len(texts)
    metrics['method'] = topics[0].get('method') if topics else None

    print(f"\n=== {args.label} ===")
    for key, value in metrics.items():
        print(f'  {key:18} {value}')
    print('\n  topik:')
    for topic in topics:
        words = ', '.join(topic.get('words', [])[:9])
        print(f"    T{topic['topic_id']} p={topic.get('proportion', 0):.3f} "
              f"n={topic.get('size', 0):<4} {words}")

    return {
        'label': args.label,
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'corpus': args.corpus,
        'metrics': metrics,
        'topics': [
            {
                'topic_id': t.get('topic_id'),
                'words': t.get('words', [])[:10],
                'proportion': t.get('proportion'),
                'size': t.get('size'),
            }
            for t in topics
        ],
    }


def save(entry: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)

    history: Dict[str, Any] = {}
    if os.path.isfile(RESULTS_PATH):
        try:
            with open(RESULTS_PATH, encoding='utf-8') as handle:
                history = json.load(handle)
        except json.JSONDecodeError:
            print('peringatan: topic_results.json rusak, dimulai ulang')

    history[entry['label']] = entry
    with open(RESULTS_PATH, 'w', encoding='utf-8') as handle:
        json.dump(history, handle, ensure_ascii=False, indent=2)
    print(f'\ntersimpan ke {RESULTS_PATH} dengan label "{entry["label"]}"')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', default='data/external/youtube/test_gold.json')
    parser.add_argument('--limit', type=int, default=0, help='0 = seluruh korpus')
    parser.add_argument('--num-topics', type=int, default=5)
    parser.add_argument('--top-n', type=int, default=10,
                        help='jumlah kata teratas yang dinilai coherence')
    parser.add_argument('--label', default='baseline')
    args = parser.parse_args()

    entry = asyncio.run(run(args))
    save(entry)


if __name__ == '__main__':
    main()
