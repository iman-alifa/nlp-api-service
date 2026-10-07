"""
Sapuan parameter UMAP/HDBSCAN untuk topic modeling, dipandu metrik.

Parameter BERTopic pada layanan ini semula ditetapkan dengan rumus tebakan
(`min_cluster_size = n // 20`). Pada 885 komentar rumus itu memberi
min_cluster_size 44 dan min_samples 17 - terlalu kasar, sehingga hanya 3 topik
terbentuk, satu di antaranya menyerap 53% dokumen, dan 34% dokumen tidak masuk
topik mana pun. Skrip ini mengukur alternatifnya alih-alih menebak lagi.

Embedding dihitung SEKALI lalu dipakai ulang untuk seluruh kombinasi; tanpa itu
setiap kombinasi menanggung ~25 detik encoding dan sapuan menjadi tidak praktis.

Contoh:
    python scripts/sweep_topic_params.py --limit 885
    python scripts/sweep_topic_params.py --limit 300 --out data/experiments/sweep_300.json
"""

import argparse
import json
import os
import sys
from itertools import product
from typing import Any, Dict, List

# torch lebih dulu; lihat catatan urutan impor di scripts/eval_topics.py.
import torch  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from app.config import settings  # noqa: E402
from app.preprocessing.text_cleaner import TextCleaner  # noqa: E402
from app.services.topic_service import TOPIC_KEYWORD_STOPWORDS  # noqa: E402
from app.utils.topic_metrics import evaluate  # noqa: E402

LARAVEL_CONFIG = {
    'case_folding': True,
    'remove_punctuation': True,
    'remove_numbers': False,
    'remove_stopwords': True,
    'stemming': True,
    'lemmatization': False,
    'custom_stopwords': [],
}

# Rentang yang diuji. Nilai produksi lama (44/17) sengaja disertakan sebagai
# pembanding, dan nilai kecil disertakan karena panduan BERTopic menganjurkan
# min_cluster_size jauh lebih kecil untuk korpus beberapa ratus dokumen.
GRID = {
    'min_cluster_size': [6, 8, 10, 12, 15, 20],
    'min_samples': [2, 3, 5],
    'n_neighbors': [10, 15, 25],
    'n_components': [5],
}


def load_texts(path: str, limit: int) -> List[str]:
    with open(path, encoding='utf-8') as handle:
        raw = json.load(handle)
    texts = [i['text'] if isinstance(i, dict) else str(i) for i in raw]
    texts = [t for t in texts if t and t.strip()]
    return texts[:limit] if limit else texts


def run_one(
    embeddings: np.ndarray,
    documents: List[str],
    reference: List[List[str]],
    params: Dict[str, int],
) -> Dict[str, Any]:
    """Klasterisasi memakai embedding yang sudah ada, lalu ukur mutunya."""
    from bertopic import BERTopic
    from bertopic.representation import MaximalMarginalRelevance
    from bertopic.vectorizers import ClassTfidfTransformer
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    umap_model = UMAP(
        n_neighbors=params['n_neighbors'],
        n_components=params['n_components'],
        min_dist=0.0,
        metric='cosine',
        random_state=settings.topic_seed,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=params['min_cluster_size'],
        min_samples=params['min_samples'],
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=True,
    )
    vectorizer = CountVectorizer(
        stop_words=sorted(TOPIC_KEYWORD_STOPWORDS),
        min_df=2,
        max_df=0.85,
        ngram_range=(1, 1),
        token_pattern=r'\b[a-z][a-z]+\b',
    )

    model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=True),
        representation_model=(
            MaximalMarginalRelevance(diversity=settings.topic_mmr_diversity)
            if settings.topic_mmr_diversity > 0 else None
        ),
        top_n_words=settings.topic_top_n_words,
        calculate_probabilities=False,
        verbose=False,
    )

    assigned, _ = model.fit_transform(documents, embeddings=embeddings)

    topic_words = [
        [w for w, _ in model.get_topic(tid)]
        for tid in sorted(set(assigned))
        if tid != -1 and model.get_topic(tid)
    ]
    if not topic_words:
        return {**params, 'num_topics': 0, 'c_v': 0.0, 'c_npmi': 0.0,
                'diversity': 0.0, 'outlier_rate': 1.0, 'largest_share': 1.0}

    metrics = evaluate(topic_words, reference, list(assigned))

    sizes = [sum(1 for t in assigned if t == tid) for tid in sorted(set(assigned)) if tid != -1]
    metrics['largest_share'] = round(max(sizes) / len(assigned), 4) if sizes else 1.0

    return {**params, **metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', default='data/external/youtube/test_gold.json')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--out', default='data/experiments/topic_sweep.json')
    args = parser.parse_args()

    texts = load_texts(args.corpus, args.limit)
    cleaner = TextCleaner(LARAVEL_CONFIG)
    cleaned = cleaner.clean_texts(texts)

    keep = [i for i, t in enumerate(cleaned) if t.strip()]
    documents = [cleaned[i] for i in keep]
    reference = [d.split() for d in documents]
    print(f'{len(documents)} dokumen dimodelkan (dari {len(texts)} teks)')

    from sentence_transformers import SentenceTransformer
    print('menghitung embedding sekali untuk seluruh kombinasi...')
    encoder = SentenceTransformer(settings.aspect_base_model)
    embeddings = encoder.encode(documents, show_progress_bar=False)

    combos = [
        dict(zip(GRID.keys(), values))
        for values in product(*GRID.values())
        # min_samples > min_cluster_size tidak bermakna bagi HDBSCAN.
        if dict(zip(GRID.keys(), values))['min_samples']
        <= dict(zip(GRID.keys(), values))['min_cluster_size']
    ]
    print(f'{len(combos)} kombinasi\n')

    results: List[Dict[str, Any]] = []
    header = (f"{'mcs':>4} {'ms':>4} {'nn':>4} | {'topik':>5} {'c_v':>7} "
              f"{'c_npmi':>7} {'divers':>7} {'outlier':>7} {'terbesar':>8}")
    print(header)
    print('-' * len(header))

    for combo in combos:
        try:
            row = run_one(embeddings, documents, reference, combo)
        except Exception as exc:                      # noqa: BLE001
            print(f"  {combo} gagal: {type(exc).__name__}: {str(exc)[:60]}")
            continue
        results.append(row)
        print(f"{row['min_cluster_size']:>4} {row['min_samples']:>4} "
              f"{row['n_neighbors']:>4} | {row['num_topics']:>5} "
              f"{row['c_v']:>7.4f} {row['c_npmi']:>7.4f} {row['diversity']:>7.4f} "
              f"{row['outlier_rate']:>7.4f} {row['largest_share']:>8.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    print(f'\ntersimpan ke {args.out}')


if __name__ == '__main__':
    main()
