"""
Bandingkan model representasi topik BERTopic, diukur dengan coherence.

Klasterisasi menentukan dokumen mana masuk topik mana; *representasi* menentukan
KATA apa yang mewakili tiap topik. Keduanya terpisah, dan coherence dihitung
justru atas kata-kata itu - jadi representasi berpengaruh langsung pada angka
yang dinilai penguji, tanpa mengubah penugasan dokumen sama sekali.

Yang dibandingkan:
- c-TF-IDF polos: bobot bawaan BERTopic.
- reduce_frequent_words: menekan kata yang umum di seluruh korpus.
- BM25: pembobotan alternatif yang lebih tahan terhadap dokumen panjang.
- MaximalMarginalRelevance: memilih kata yang saling melengkapi, bukan sinonim
  yang berulang.
- KeyBERTInspired: menimbang ulang kandidat berdasarkan kemiripan embedding
  dengan dokumen topik.

Embedding dihitung sekali dan dipakai ulang untuk seluruh varian.

Contoh:
    python scripts/bench_topic_representation.py --encoder LazarusNLP/all-indo-e5-small-v4
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List

import torch  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

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


def build_variants(embedding_model: Any) -> Dict[str, Dict[str, Any]]:
    """Bangun kombinasi ctfidf_model / representation_model yang diuji."""
    from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
    from bertopic.vectorizers import ClassTfidfTransformer

    return {
        'ctfidf-polos': {
            'ctfidf_model': ClassTfidfTransformer(),
            'representation_model': None,
        },
        'ctfidf-reduce': {
            'ctfidf_model': ClassTfidfTransformer(reduce_frequent_words=True),
            'representation_model': None,
        },
        'ctfidf-bm25': {
            'ctfidf_model': ClassTfidfTransformer(bm25_weighting=True),
            'representation_model': None,
        },
        'ctfidf-bm25+reduce': {
            'ctfidf_model': ClassTfidfTransformer(
                bm25_weighting=True, reduce_frequent_words=True),
            'representation_model': None,
        },
        'mmr-0.3': {
            'ctfidf_model': ClassTfidfTransformer(reduce_frequent_words=True),
            'representation_model': MaximalMarginalRelevance(diversity=0.3),
        },
        'mmr-0.5': {
            'ctfidf_model': ClassTfidfTransformer(reduce_frequent_words=True),
            'representation_model': MaximalMarginalRelevance(diversity=0.5),
        },
        'keybert': {
            'ctfidf_model': ClassTfidfTransformer(reduce_frequent_words=True),
            'representation_model': KeyBERTInspired(),
        },
        'keybert+mmr': {
            'ctfidf_model': ClassTfidfTransformer(reduce_frequent_words=True),
            'representation_model': [KeyBERTInspired(),
                                     MaximalMarginalRelevance(diversity=0.3)],
        },
    }


def run_variant(
    name: str,
    variant: Dict[str, Any],
    embeddings: np.ndarray,
    documents: List[str],
    reference: List[List[str]],
    embedding_model: Any,
    num_topics: int,
) -> Dict[str, Any]:
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    n = len(documents)
    model = BERTopic(
        embedding_model=embedding_model,
        umap_model=UMAP(
            n_neighbors=max(10, min(30, n // 60)), n_components=5,
            min_dist=0.0, metric='cosine', random_state=settings.topic_seed),
        hdbscan_model=HDBSCAN(
            min_cluster_size=max(5, min(25, n // 45)), min_samples=3,
            metric='euclidean', cluster_selection_method='eom',
            prediction_data=True),
        vectorizer_model=CountVectorizer(
            stop_words=sorted(TOPIC_KEYWORD_STOPWORDS),
            min_df=2 if n >= 50 else 1, max_df=0.85 if n >= 50 else 1.0,
            ngram_range=(1, 1), token_pattern=r'\b[a-z][a-z]+\b'),
        top_n_words=settings.topic_top_n_words,
        calculate_probabilities=False,
        verbose=False,
        **{k: v for k, v in variant.items() if v is not None},
    )

    assigned, _ = model.fit_transform(documents, embeddings=embeddings)

    if len(set(assigned) - {-1}) > num_topics:
        model.reduce_topics(documents, nr_topics=num_topics)
        assigned = model.topics_
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
        return {'variant': name, 'num_topics': 0, 'c_v': 0.0, 'c_npmi': 0.0,
                'diversity': 0.0, 'outlier_rate': 1.0}

    metrics = evaluate(topic_words, reference, list(assigned))
    metrics['variant'] = name
    metrics['top_words'] = ', '.join(topic_words[0][:6])
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', default='data/external/youtube/test_gold.json')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--num-topics', type=int, default=15)
    parser.add_argument('--encoder', default='indobenchmark/indobert-base-p1')
    parser.add_argument('--prefix', default='', help="awalan E5, mis. 'query: '")
    parser.add_argument('--out', default='data/experiments/topic_representation.json')
    args = parser.parse_args()

    texts = load_texts(args.corpus, args.limit)
    documents = [t for t in TextCleaner(LARAVEL_CONFIG).clean_texts(texts) if t.strip()]
    reference = [d.split() for d in documents]
    print(f'{len(documents)} dokumen | encoder {args.encoder}\n')

    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(args.encoder)
    payload = [args.prefix + d for d in documents] if args.prefix else documents
    embeddings = encoder.encode(payload, show_progress_bar=False)

    header = f"{'varian':20} {'topik':>5} {'c_v':>7} {'c_npmi':>7} {'divers':>7} {'outlier':>7}"
    print(header)
    print('-' * len(header))

    results: List[Dict[str, Any]] = []
    for name, variant in build_variants(encoder).items():
        try:
            row = run_variant(name, variant, embeddings, documents, reference,
                              encoder, args.num_topics)
        except Exception as exc:                        # noqa: BLE001
            print(f"{name:20} gagal: {type(exc).__name__}: {str(exc)[:45]}")
            continue
        results.append(row)
        print(f"{row['variant']:20} {row['num_topics']:>5} {row['c_v']:>7.4f} "
              f"{row['c_npmi']:>7.4f} {row['diversity']:>7.4f} {row['outlier_rate']:>7.4f}")
        print(f"{'':20} {row.get('top_words', '')}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    print(f'\ntersimpan ke {args.out}')


if __name__ == '__main__':
    main()
