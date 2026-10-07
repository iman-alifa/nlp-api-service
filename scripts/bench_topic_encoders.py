"""
Bandingkan model embedding untuk BERTopic, diukur dengan coherence.

Layanan ini semula memakai `SentenceTransformer('indobenchmark/indobert-base-p1')`.
IndoBERT adalah model masked-language-modeling, BUKAN sentence encoder, sehingga
sentence-transformers membungkusnya dengan mean pooling mentah dan mencetak:

    No sentence-transformers model found with name indobenchmark/indobert-base-p1.
    Creating a new one with mean pooling.

Embedding BERT mentah bersifat anisotropik - seluruh vektor menumpuk pada satu
kerucut sempit - sehingga jarak antar-dokumen kurang informatif. Karena BERTopic
mengelompokkan dokumen persis berdasarkan jarak itu, pilihan encoder adalah
pengungkit terbesar bagi mutu topik.

Perbandingan dibuat ADIL: parameter klasterisasi, vectorizer, seed, dan korpus
identik untuk semua encoder; hanya embedding yang berbeda.

Contoh:
    python scripts/bench_topic_encoders.py
    python scripts/bench_topic_encoders.py --limit 300 --num-topics 15
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List

# torch lebih dulu; lihat catatan urutan impor di app/services/topic_service.py.
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

# (nama, awalan). Model keluarga E5 dilatih dengan awalan dan menurun tajam
# mutunya bila awalannya dihilangkan.
ENCODERS = [
    ('indobenchmark/indobert-base-p1', ''),                 # dipakai sekarang
    ('firqaaa/indo-sentence-bert-base', ''),                 # sentence-BERT Indonesia
    ('LazarusNLP/all-indo-e5-small-v4', 'query: '),          # E5 Indonesia
    ('intfloat/multilingual-e5-small', 'query: '),           # E5 multibahasa
    ('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', ''),
]


def load_texts(path: str, limit: int) -> List[str]:
    with open(path, encoding='utf-8') as handle:
        raw = json.load(handle)
    texts = [i['text'] if isinstance(i, dict) else str(i) for i in raw]
    texts = [t for t in texts if t and t.strip()]
    return texts[:limit] if limit else texts


def cluster_and_score(
    embeddings: np.ndarray,
    documents: List[str],
    reference: List[List[str]],
    num_topics: int,
) -> Dict[str, Any]:
    """Klasterisasi dengan parameter produksi, lalu ukur mutunya."""
    from bertopic import BERTopic
    from bertopic.representation import MaximalMarginalRelevance
    from bertopic.vectorizers import ClassTfidfTransformer
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    n = len(documents)
    umap_model = UMAP(
        n_neighbors=max(10, min(30, n // 60)),
        n_components=5,
        min_dist=0.0,
        metric='cosine',
        random_state=settings.topic_seed,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=max(5, min(25, n // 45)),
        min_samples=3,
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=True,
    )
    vectorizer = CountVectorizer(
        stop_words=sorted(TOPIC_KEYWORD_STOPWORDS),
        min_df=2 if n >= 50 else 1,
        max_df=0.85 if n >= 50 else 1.0,
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

    if len(set(assigned) - {-1}) > num_topics:
        model.reduce_topics(documents, nr_topics=num_topics)
        assigned = model.topics_

    if -1 in set(assigned):
        try:
            assigned = model.reduce_outliers(
                documents, list(assigned), strategy='c-tf-idf',
                threshold=settings.topic_outlier_threshold,
            )
        except Exception:                               # noqa: BLE001
            pass

    ids = sorted(t for t in set(assigned) if t != -1)
    topic_words = [[w for w, _ in model.get_topic(t)] for t in ids if model.get_topic(t)]
    if not topic_words:
        return {'num_topics': 0, 'c_v': 0.0, 'c_npmi': 0.0,
                'diversity': 0.0, 'outlier_rate': 1.0, 'largest_share': 1.0}

    metrics = evaluate(topic_words, reference, list(assigned))
    sizes = [sum(1 for t in assigned if t == tid) for tid in ids]
    metrics['largest_share'] = round(max(sizes) / len(assigned), 4) if sizes else 1.0
    metrics['top_words'] = ', '.join(topic_words[0][:6])
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', default='data/external/youtube/test_gold.json')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--num-topics', type=int, default=15)
    parser.add_argument('--out', default='data/experiments/topic_encoders.json')
    args = parser.parse_args()

    texts = load_texts(args.corpus, args.limit)
    cleaned = TextCleaner(LARAVEL_CONFIG).clean_texts(texts)
    documents = [t for t in cleaned if t.strip()]
    reference = [d.split() for d in documents]
    print(f'{len(documents)} dokumen, num_topics={args.num_topics}\n')

    from sentence_transformers import SentenceTransformer

    header = (f"{'encoder':46} {'topik':>5} {'c_v':>7} {'c_npmi':>7} "
              f"{'divers':>7} {'outlier':>7}")
    print(header)
    print('-' * len(header))

    results: List[Dict[str, Any]] = []
    for name, prefix in ENCODERS:
        try:
            encoder = SentenceTransformer(name)
            payload = [prefix + d for d in documents] if prefix else documents
            embeddings = encoder.encode(payload, show_progress_bar=False)
            row = cluster_and_score(embeddings, documents, reference, args.num_topics)
        except Exception as exc:                        # noqa: BLE001
            print(f"{name[:46]:46} gagal: {type(exc).__name__}: {str(exc)[:40]}")
            continue
        finally:
            encoder = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        row['encoder'] = name
        results.append(row)
        print(f"{name[:46]:46} {row['num_topics']:>5} {row['c_v']:>7.4f} "
              f"{row['c_npmi']:>7.4f} {row['diversity']:>7.4f} "
              f"{row['outlier_rate']:>7.4f}")
        print(f"{'':46} kata topik teratas: {row.get('top_words', '')}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    print(f'\ntersimpan ke {args.out}')


if __name__ == '__main__':
    main()
