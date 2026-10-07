"""
Pilih jumlah topik berdasarkan coherence, bukan berdasarkan tebakan.

Menentukan jumlah topik dengan memaksimalkan coherence adalah prosedur baku pada
literatur topic modeling; angka yang dilaporkan menjadi hasil pencarian yang
dapat diperiksa ulang, bukan nilai yang kebetulan dipilih.

Model di-fit SEKALI, lalu dipangkas bertahap turun (`reduce_topics` hanya bisa
mengurangi) dan dinilai pada tiap langkah. Embedding di-cache, sehingga seluruh
sapuan hanya membayar satu kali proses encoding.

Contoh:
    python scripts/sweep_topic_count.py
    python scripts/sweep_topic_count.py --encoder intfloat/multilingual-e5-small --prefix "query: "
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List

import torch  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings  # noqa: E402
from app.preprocessing.text_cleaner import TextCleaner  # noqa: E402
from app.services.topic_service import TOPIC_KEYWORD_STOPWORDS  # noqa: E402
from app.utils.topic_metrics import evaluate  # noqa: E402

LARAVEL_CONFIG = {
    'case_folding': True, 'remove_punctuation': True, 'remove_numbers': False,
    'remove_stopwords': True, 'stemming': True, 'lemmatization': False,
    'custom_stopwords': [],
}


def load_texts(path: str, limit: int) -> List[str]:
    with open(path, encoding='utf-8') as handle:
        raw = json.load(handle)
    texts = [i['text'] if isinstance(i, dict) else str(i) for i in raw]
    texts = [t for t in texts if t and t.strip()]
    return texts[:limit] if limit else texts


def build_model(n: int, embedding_model: Any):
    from bertopic import BERTopic
    from bertopic.representation import MaximalMarginalRelevance
    from bertopic.vectorizers import ClassTfidfTransformer
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    return BERTopic(
        embedding_model=embedding_model,
        umap_model=UMAP(
            n_neighbors=max(10, min(30, n // 59)), n_components=5,
            min_dist=0.0, metric='cosine', random_state=settings.topic_seed),
        hdbscan_model=HDBSCAN(
            min_cluster_size=max(5, min(25, n // 59)), min_samples=2,
            metric='euclidean', cluster_selection_method='eom',
            prediction_data=True),
        vectorizer_model=CountVectorizer(
            stop_words=sorted(TOPIC_KEYWORD_STOPWORDS),
            min_df=2 if n >= 50 else 1, max_df=0.85 if n >= 50 else 1.0,
            ngram_range=(1, 1), token_pattern=r'\b[a-z][a-z]+\b'),
        ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=True),
        representation_model=(
            MaximalMarginalRelevance(diversity=settings.topic_mmr_diversity)
            if settings.topic_mmr_diversity > 0 else None
        ),
        top_n_words=settings.topic_top_n_words,
        calculate_probabilities=False,
        verbose=False,
    )


def score(model, documents, reference, assigned) -> Dict[str, Any]:
    if -1 in set(assigned):
        try:
            assigned = model.reduce_outliers(
                documents, list(assigned), strategy='c-tf-idf',
                threshold=settings.topic_outlier_threshold)
        except Exception:                               # noqa: BLE001
            pass

    ids = sorted(t for t in set(assigned) if t != -1)
    topic_words = [[w for w, _ in model.get_topic(t)] for t in ids if model.get_topic(t)]
    if not topic_words:
        return {}

    metrics = evaluate(topic_words, reference, list(assigned))
    sizes = [sum(1 for t in assigned if t == tid) for tid in ids]
    metrics['largest_share'] = round(max(sizes) / len(assigned), 4) if sizes else 1.0
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', default='data/external/youtube/test_gold.json')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--encoder', default='indobenchmark/indobert-base-p1')
    parser.add_argument('--prefix', default='')
    parser.add_argument('--out', default='data/experiments/topic_count_sweep.json')
    args = parser.parse_args()

    texts = load_texts(args.corpus, args.limit)
    documents = [t for t in TextCleaner(LARAVEL_CONFIG).clean_texts(texts) if t.strip()]
    reference = [d.split() for d in documents]
    n = len(documents)
    print(f'{n} dokumen | encoder {args.encoder}\n')

    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(args.encoder)
    payload = [args.prefix + d for d in documents] if args.prefix else documents
    embeddings = encoder.encode(payload, show_progress_bar=False)

    model = build_model(n, encoder)
    assigned, _ = model.fit_transform(documents, embeddings=embeddings)
    natural = len(set(assigned) - {-1})
    print(f'klaster alami HDBSCAN: {natural} topik\n')

    header = f"{'k':>4} {'c_v':>8} {'c_npmi':>8} {'divers':>8} {'outlier':>8} {'terbesar':>9}"
    print(header)
    print('-' * len(header))

    results: List[Dict[str, Any]] = []

    row = score(model, documents, reference, assigned)
    if row:
        row['k'] = natural
        results.append(row)
        print(f"{natural:>4} {row['c_v']:>8.4f} {row['c_npmi']:>8.4f} "
              f"{row['diversity']:>8.4f} {row['outlier_rate']:>8.4f} "
              f"{row['largest_share']:>9.4f}")

    # reduce_topics hanya bisa MENGURANGI, jadi sapuan berjalan menurun.
    for k in range(natural - 1, 1, -1):
        try:
            model.reduce_topics(documents, nr_topics=k)
        except Exception as exc:                        # noqa: BLE001
            print(f"{k:>4} gagal: {type(exc).__name__}: {str(exc)[:45]}")
            break
        row = score(model, documents, reference, model.topics_)
        if not row:
            continue
        row['k'] = k
        results.append(row)
        print(f"{k:>4} {row['c_v']:>8.4f} {row['c_npmi']:>8.4f} "
              f"{row['diversity']:>8.4f} {row['outlier_rate']:>8.4f} "
              f"{row['largest_share']:>9.4f}")

    if results:
        best = max(results, key=lambda r: r['c_v'])
        print(f"\nC_v tertinggi pada k={best['k']}: {best['c_v']:.4f} "
              f"(diversity {best['diversity']:.3f}, outlier {best['outlier_rate']:.3f})")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    print(f'tersimpan ke {args.out}')


if __name__ == '__main__':
    main()
