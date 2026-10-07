# System Architecture and Context Document
## NLP-API-Service: Indonesian Text Analysis Platform

**Document Version:** 1.0.0  
**Last Updated:** May 19, 2026  
**Target Audience:** AI Systems & Development Teams

---

## 1. System Overview

### 1.1 Core Objective

This project is an **NLP-based REST API service** designed to perform advanced text analysis on Indonesian social media data and general Indonesian text. The system provides three primary NLP capabilities:

- **Sentiment Analysis:** Classify text emotions (positive, negative, neutral)
- **Aspect Extraction:** Identify opinion targets and aspects mentioned in text
- **Topic Modeling:** Discover latent topics and themes within text corpora

The system is specifically engineered to handle:
- Indonesian language with code-mixing and slang (e.g., "bagus banget" → "very good")
- Social media comments with informal grammar, abbreviations, and colloquialisms
- Text preprocessing challenges including typos, repeated characters, and non-standard orthography

### 1.2 Overarching Methodology

The project follows the **CRISP-ML(Q) framework** (Cross-Industry Standard Process for Machine Learning and Quality):

```
Data Understanding & Preparation
    ↓
Feature Engineering & Preprocessing
    ↓
Model Selection & Training (Supervised Learning)
    ↓
Evaluation (Accuracy, F1-Score, Confidence Metrics)
    ↓
Deployment (REST API via FastAPI)
    ↓
Feedback Loop (Active Learning Potential)
```

**Key Principle:** Each NLP module (Sentiment, Aspect, Topic) operates independently but can consume the same preprocessed text pipeline, enabling reusability and consistency.

---

## 2. Tech Stack & Integrations

### 2.1 Core Technology Stack

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| **Framework** | FastAPI | 0.109.0 | Async REST API server |
| **Server** | Uvicorn | 0.27.0 | ASGI application server |
| **Language** | Python | 3.8+ | Primary development language |
| **Data Validation** | Pydantic | 2.5.3 | Request/response schema validation |

### 2.2 Machine Learning & NLP Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| **Transformers** | 4.36.2 | Pre-trained BERT models (HuggingFace) |
| **Torch** | 2.1.2 | Deep learning framework for transformer inference |
| **Scikit-learn** | 1.3.2 | Classical ML models (Logistic Regression, Random Forest, etc.) |
| **NLTK** | 3.8.1 | Natural Language Toolkit for text processing |
| **Sastrawi** | 1.0.1 | Indonesian stemmer and stopword remover |
| **BERTopic** | 0.16.0 | Neural topic modeling with embedding clustering |
| **Gensim** | 4.3.2 | Alternative topic modeling (LDA backend) |
| **Pandas** | 2.1.4 | Data manipulation and analysis |
| **Numpy** | 1.26.2 | Numerical computing |

### 2.3 External Model Dependencies

The system leverages pre-trained models from **HuggingFace Model Hub**:

```
sentiment_service:
  └─ mdhugol/indonesia-bert-sentiment-classification
     (3-class classification: positive, neutral, negative)

aspect_service:
  └─ indobenchmark/indobert-base-p1 (IndoBERT)
     (Token classification for Named Entity Recognition)

topic_service:
  └─ indobenchmark/indobert-base-p1 (via SentenceTransformer)
     (Embedding generation for semantic clustering)
```

### 2.4 Text Processing Tools

- **Sastrawi Stemmer:** Indonesian morphological analysis
- **NLTK Stopwords:** Multilingual stopword removal
- **Regex Patterns:** Custom slang normalization & typo correction
- **Levenshtein Distance:** String similarity matching for typo detection

### 2.5 Integration Points

**Current Integration:** None (standalone API)

**Potential Future Integrations:**
- **Label Studio:** Human-in-the-loop annotation for active learning
- **PostgreSQL/MongoDB:** Persistent storage for predictions and feedback
- **Apache Kafka:** Event streaming for batch processing
- **MLflow:** Model versioning and experiment tracking

---

## 3. Machine Learning Pipeline & Data Flow

### 3.1 Data Ingestion & Preprocessing Pipeline

```
Raw Text Input (Social Media Comments, Reviews, etc.)
        ↓
    TextCleaner (app/preprocessing/text_cleaner.py)
        ↓
┌───────────────────────────────────────────────┐
│  Preprocessing Configuration (Configurable)    │
├───────────────────────────────────────────────┤
│ 1. Case Folding (lowercase conversion)         │
│ 2. Repeated Character Normalization            │
│    (e.g., "bagussss" → "bagus")               │
│ 3. Slang Normalization                         │
│    (e.g., "gak" → "tidak", "yg" → "yang")    │
│ 4. Typo Correction                             │
│    (Levenshtein distance-based matching)      │
│ 5. Punctuation Removal (optional)              │
│ 6. Number Removal (optional)                   │
│ 7. Stopword Removal (Indonesian + NLTK)       │
│ 8. Stemming (Sastrawi for Indonesian)         │
│ 9. Lemmatization (optional, not currently used)│
└───────────────────────────────────────────────┘
        ↓
    Cleaned Text Output
```

**Key Components:**

- **`TextCleaner` Class:** 
  - Accepts dictionary configuration for enabling/disabling each step
  - Maintains comprehensive Indonesian slang dictionary (100+ mappings)
  - Uses Sastrawi stemmer for morphological reduction
  - Combines multiple stopword sources (Sastrawi + NLTK)

- **Configuration Example:**
  ```python
  PreprocessingConfig(
      case_folding=True,
      remove_punctuation=True,
      remove_numbers=False,
      remove_stopwords=True,
      stemming=True,
      normalize_slang=True,
      fix_typos=True,
      custom_stopwords=[]  # User-defined stopwords
  )
  ```

### 3.2 Modeling Layer

#### 3.2.1 Sentiment Analysis Service

**Architecture:**
```
Input Texts → Preprocessing → Tokenization (IndoBERT)
    ↓
    Model: mdhugol/indonesia-bert-sentiment-classification
    (3-class classification)
    ↓
Inference:
  - Max token length: 512
  - Device: GPU (if available) or CPU
  - Output: 3 logits → softmax → class probability
    ↓
├─ Predicted Class: argmax(logits)
├─ Confidence Score: softmax(logits)[predicted_class]
└─ Full Score Distribution: {positive, neutral, negative}
```

**Fallback Mechanism:**
- If model loading fails, system falls back to **rule-based sentiment analysis**
- Uses predefined positive/negative word dictionaries
- Counts word overlap for sentiment determination
- Minimum confidence: 0.5 (neutral), Maximum: 0.95

**Output Schema:**
```json
{
  "predictions": [
    {
      "text": "Produk ini bagus banget!",
      "sentiment": "positive",
      "confidence": 0.95,
      "scores": {
        "positive": 0.95,
        "neutral": 0.04,
        "negative": 0.01
      }
    }
  ],
  "distribution": {
    "positive": 75.5,
    "neutral": 20.0,
    "negative": 4.5
  },
  "metrics": {
    "avg_confidence": 0.87,
    "min_confidence": 0.42,
    "max_confidence": 0.99,
    "total_analyzed": 100
  },
  "summary": "Dari 100 teks yang dianalisis, 75.5% menunjukkan sentimen positif..."
}
```

#### 3.2.2 Aspect Extraction Service

**Architecture (Hybrid Approach):**

The service implements three extraction modes:

1. **Trained Model Mode** (when checkpoint available):
   ```
   Texts → Tokenize (IndoBERT) → Single Forward Pass
      ↓
   Token-level Predictions (BIO tagging)
      ↓
   Decode Tokens → Extract Aspect Spans
      ↓
   Post-Process (remove stopwords, filter <3 chars)
      ↓
   Aggregate & Score Aspects
   ```

2. **Zero-Shot Mode** (default, no trained checkpoint):
   ```
   Texts → Tokenize (IndoBERT) → Single Forward Pass
      ↓
   Low-confidence Model Predictions (untrained)
      ↓
   Hybrid Strategy:
      ├─ Keyword Matching (high confidence)
      ├─ Common Aspect Dictionary Lookup
      └─ Noun/Adjective Heuristics
      ↓
   Score = Frequency × Confidence
   ```

3. **Rule-Based Mode** (predefined aspects):
   ```
   Predefined Aspects → Pattern Matching in Texts
      ↓
   Extract Context Windows
      ↓
   Calculate Position & Frequency
   ```

**Label Scheme (BIO Tagging):**
- `O` (Outside): Not an aspect
- `B-ASPECT` (Begin): Start of aspect phrase
- `I-ASPECT` (Inside): Continuation of aspect phrase

**Output Schema:**
```json
{
  "aspects": [
    {
      "aspect": "pelayanan",
      "count": 12,
      "score": 9.84,
      "confidence": 0.82,
      "variants": ["pelayanan", "layanan", "service"],
      "occurrences": [
        {
          "text": "Pelayanannya sangat memuaskan",
          "context": "Pelayanannya sangat memuaskan",
          "text_index": 0
        }
      ]
    }
  ],
  "aspect_sentiments": {
    "pelayanan": {"positive": 8, "negative": 2, "neutral": 2},
    "harga": {"positive": 3, "negative": 7, "neutral": 0}
  },
  "statistics": {
    "total_aspects_found": 8,
    "total_mentions": 45,
    "aspect_sentiment_pairs": 23
  },
  "summary": "Ditemukan 8 aspek unik dari 45 mention..."
}
```

#### 3.2.3 Topic Modeling Service

**Architecture (Dual Approach):**

1. **BERTopic with IndoBERT** (primary):
   ```
   Texts → SentenceTransformer Embeddings (IndoBERT)
      ↓
   Dimensionality Reduction (UMAP)
      ├─ Adaptive n_neighbors: min(50, max(15, n_texts//20))
      └─ Adaptive n_components: min(5, max(2, n_texts//100))
      ↓
   Clustering:
      ├─ HDBSCAN (min_cluster_size adaptive to dataset)
      └─ Fallback: KMeans (if HDBSCAN unavailable)
      ↓
   Topic Representation (TF-IDF + diversity)
      ↓
   Output: Topic labels + keywords per topic
   ```

2. **LDA (Fallback):**
   ```
   TF-IDF Vectorization → Latent Dirichlet Allocation
      ↓
   Extract topic distributions per document
   ```

**Output Schema:**
```json
{
  "topics": [
    {
      "topic_id": 0,
      "keywords": ["harga", "murah", "diskon", "cashback", "mahal"],
      "topic_label": "Price & Cost",
      "document_count": 34,
      "coherence": 0.62
    },
    {
      "topic_id": 1,
      "keywords": ["pengiriman", "cepat", "lambat", "sampai", "ongkir"],
      "topic_label": "Delivery & Shipping",
      "document_count": 28,
      "coherence": 0.58
    }
  ],
  "word_frequencies": {
    "harga": 156,
    "bagus": 134,
    "pengiriman": 98,
    ...
  },
  "num_texts": 100,
  "num_topics": 5,
  "summary": "Analisis menemukan 5 topik utama dari 100 teks..."
}
```

### 3.3 Model Evaluation Metrics

The system calculates and reports the following metrics:

#### Sentiment Analysis:
- **Accuracy:** Correctly predicted sentiments / total predictions
- **Confidence Distribution:** Mean, Min, Max confidence scores
- **Class Distribution:** % breakdown per sentiment class

#### Aspect Extraction:
- **Precision:** Relevant aspects / extracted aspects
- **Recall:** Found relevant aspects / total relevant aspects
- **F1-Score:** Harmonic mean of precision and recall
- **Coverage:** Unique aspects found / expected aspects

#### Topic Modeling:
- **Coherence Score:** Topic keyword semantic coherence (0-1 scale)
- **Perplexity:** Model's ability to generalize to unseen data
- **Silhouette Score:** Clustering quality metric

### 3.4 Active Learning Loop (Future Implementation)

**Proposed Architecture:**

```
1. Model Predictions (Current)
        ↓
2. Confidence Scoring
        ↓
3. Uncertainty Sampling
   ├─ Select predictions with confidence 0.3-0.7
   └─ Export to Label Studio (if integrated)
        ↓
4. Human Annotation
        ↓
5. Dataset Enrichment
        ↓
6. Model Retraining
        ↓
7. Performance Evaluation
        ↓
[Return to Step 1]
```

**Key Metrics for Uncertainty:**
- Entropy of probability distribution
- Margin between top-2 predicted classes
- Distance to decision boundary

---

## 4. Architecture & Directory Structure

### 4.1 Directory Tree with Descriptions

```
nlp-api-service/
│
├── run.py                           # Application entry point
│                                    # Usage: python run.py
│
├── requirements.txt                 # Python dependencies
│
├── .env                             # Environment variables (not in repo)
│
├── logs/                            # Runtime logs
│   └── api.log                      # Application log file
│
├── data/                            # Static data assets
│   └── stopwords_id.txt             # Indonesian stopwords reference
│
├── models/                          # ML model checkpoints (empty in base)
│   └── [trained_models_go_here]    # Aspect extraction checkpoint, etc.
│
├── tests/                           # Unit and integration tests (future)
│
└── app/                             # Main application package
    │
    ├── __init__.py                  # Package initialization
    │
    ├── main.py                      # FastAPI app definition & routes
    │   ├─ GET  /                    # Health check
    │   ├─ GET  /health              # Detailed health status
    │   ├─ POST /api/preprocess      # Text preprocessing
    │   ├─ POST /api/analyze/sentiment     # Sentiment analysis
    │   ├─ POST /api/analyze/aspect        # Aspect extraction
    │   └─ POST /api/analyze/topic         # Topic modeling
    │
    ├── config.py                    # Application configuration
    │   └─ Settings class (pydantic)
    │       ├─ app_name, version, env
    │       ├─ host, port, debug
    │       ├─ model paths & GPU config
    │       ├─ max_text_length, max_batch_size
    │       └─ log level & file paths
    │
    ├── preprocessing/               # Text preprocessing module
    │   ├── __init__.py
    │   ├── text_cleaner.py          # TextCleaner class (433 lines)
    │   │   ├─ Case folding
    │   │   ├─ Slang normalization (100+ mappings)
    │   │   ├─ Typo correction (Levenshtein)
    │   │   ├─ Stopword removal (Sastrawi + NLTK)
    │   │   ├─ Stemming (Indonesian morphology)
    │   │   └─ Clean batch processing
    │   │
    │   └── indonesian_stopwords.py  # Custom stopword definitions
    │
    ├── schemas/                     # Pydantic data models
    │   ├── __init__.py
    │   └── request_schemas.py       # Request/response schemas
    │       ├─ PreprocessingConfig
    │       ├─ PreprocessRequest
    │       ├─ TextAnalysisRequest
    │       └─ HealthResponse
    │
    ├── services/                    # ML services (core logic)
    │   ├── __init__.py
    │   │
    │   ├── sentiment_service.py     # SentimentService (224 lines)
    │   │   ├─ Model loading (mdhugol/indonesia-bert)
    │   │   ├─ Async analyze()
    │   │   ├─ Predict with BERT or rule-based
    │   │   ├─ Distribution calculation
    │   │   ├─ Metrics generation
    │   │   └─ Summary generation
    │   │
    │   ├── aspect_service.py        # AspectService (763 lines)
    │   │   ├─ Model initialization (indobert-base-p1)
    │   │   ├─ BIO tagging scheme
    │   │   ├─ Three extraction modes:
    │   │   │   ├─ Trained model (checkpoint-based)
    │   │   │   ├─ Zero-shot (hybrid approach)
    │   │   │   └─ Rule-based (predefined aspects)
    │   │   ├─ Single forward pass for batch
    │   │   ├─ Token decoding & aspect aggregation
    │   │   ├─ Aspect-sentiment correlation
    │   │   └─ Statistics computation
    │   │
    │   └── topic_service.py         # TopicService (376 lines)
    │       ├─ BERTopic initialization
    │       ├─ SentenceTransformer embedding
    │       ├─ UMAP dimensionality reduction (adaptive)
    │       ├─ HDBSCAN/KMeans clustering (adaptive)
    │       ├─ LDA fallback mechanism
    │       ├─ Word frequency extraction
    │       └─ Coherence score calculation
    │
    ├── utils/                       # Utility functions
    │   ├── __init__.py
    │   └── logger.py                # Logging configuration
    │       ├─ setup_logger()
    │       ├─ Console + file handlers
    │       └─ Formatted output with timestamps
    │
    └── models/                      # Model definitions (empty)
        └── [custom_model_classes_go_here]
```

### 4.2 Component Interaction Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                         FastAPI Server                            │
│                    (Uvicorn ASGI Runner)                         │
└──────────────────────────────────────────────────────────────────┘
                              ↑
                    HTTP Requests/Responses
                              ↑
┌──────────────────────────────────────────────────────────────────┐
│                    Route Handlers (main.py)                      │
│  ├─ /api/preprocess                                             │
│  ├─ /api/analyze/sentiment  → SentimentService.analyze()        │
│  ├─ /api/analyze/aspect     → AspectService.analyze()           │
│  └─ /api/analyze/topic      → TopicService.analyze()            │
└──────────────────────────────────────────────────────────────────┘
         ↑                           ↑                    ↑
         │ (called by)               │                    │
         │                           │                    │
┌────────┴──────────────────┐  ┌─────┴─────────┐  ┌─────┴──────────┐
│  TextCleaner              │  │ Aspect        │  │ Topic          │
│  (preprocessing/          │  │ Service       │  │ Service        │
│   text_cleaner.py)        │  │ (services/)   │  │ (services/)    │
│                           │  │               │  │                │
│ • Case folding            │  │ Depends on:   │  │ Depends on:    │
│ • Slang normalization     │  │ • TextCleaner │  │ • TextCleaner  │
│ • Typo correction         │  │ • IndoBERT    │  │ • IndoBERT     │
│ • Stopword removal        │  │ • Sentiment   │  │ • BERTopic     │
│ • Stemming                │  │   Service     │  │ • LDA (fallback)
└────────┬──────────────────┘  └─────┬─────────┘  └─────┬──────────┘
         │                           │                    │
         └──────────────┬────────────┴─────────────────────┘
                        │
                        ↓
        ┌───────────────────────────────────────┐
        │  External NLP Libraries               │
        ├───────────────────────────────────────┤
        │ • Transformers (HuggingFace)          │
        │ • PyTorch (GPU/CPU inference)         │
        │ • Sastrawi (Indonesian stemmer)       │
        │ • NLTK (text processing)              │
        │ • BERTopic (neural topic modeling)    │
        │ • Scikit-learn (classical ML)         │
        └───────────────────────────────────────┘
                        ↓
        ┌───────────────────────────────────────┐
        │  Pre-trained Models (HuggingFace)     │
        ├───────────────────────────────────────┤
        │ 1. mdhugol/indonesia-bert-sentiment   │
        │ 2. indobenchmark/indobert-base-p1     │
        │ 3. SentenceTransformer (embeddings)   │
        └───────────────────────────────────────┘
```

### 4.3 Data Flow: Request to Response

```
HTTP POST /api/analyze/sentiment
    ↓
1. Request Validation (Pydantic)
   ├─ TextAnalysisRequest schema check
   ├─ Validate texts (non-empty, length limits)
   └─ Extract preprocessing_config
    ↓
2. Instantiate TextCleaner
   └─ Load configuration dict
    ↓
3. Preprocessing (Optional)
   ├─ clean_texts() batch processing
   └─ Return cleaned text list
    ↓
4. Model Inference
   ├─ Tokenize cleaned texts (max_length=512)
   ├─ Single forward pass through BERT
   ├─ Softmax probability distribution
   └─ argmax for predicted class
    ↓
5. Post-Processing
   ├─ Calculate sentiment distribution
   ├─ Aggregate confidence metrics
   └─ Generate human-readable summary
    ↓
6. Return JSON Response
   ├─ predictions: [list of predictions]
   ├─ distribution: {positive, neutral, negative}
   ├─ metrics: {avg_confidence, min, max, total}
   └─ summary: "Dari X teks..."
    ↓
HTTP 200 OK (JSON Response)
```

### 4.4 Database Integration (Future)

**Planned Schema for Persistence:**

```sql
-- Predictions Table
CREATE TABLE predictions (
    id UUID PRIMARY KEY,
    created_at TIMESTAMP,
    analysis_type VARCHAR (sentiment, aspect, topic),
    raw_text TEXT,
    preprocessed_text TEXT,
    model_name VARCHAR,
    confidence FLOAT,
    result JSONB,
    user_feedback VARCHAR (correct, incorrect, irrelevant),
    feedback_timestamp TIMESTAMP
);

-- Active Learning Queue
CREATE TABLE uncertainty_samples (
    id UUID PRIMARY KEY,
    prediction_id UUID FOREIGN KEY,
    entropy FLOAT,
    confidence_margin FLOAT,
    annotation_status VARCHAR (pending, annotated, rejected),
    human_label VARCHAR,
    created_at TIMESTAMP
);
```

---

## 5. Strict Rules for AI Interaction

### Rule 1: Code Formatting & Style Standards

**Requirement:** All code changes MUST adhere to the following standards:

- **Language:** Python 3.8+ with type hints throughout
- **Format:** PEP 8 with max line length of 100 characters
- **Imports:** Organized in 3 sections (stdlib, third-party, local) with blank lines
- **Docstrings:** All public functions/classes must have Google-style docstrings
- **Comments:** Explain "why", not "what"; use `# ──` for section dividers
- **Naming:** 
  - Classes: PascalCase
  - Functions/variables: snake_case
  - Constants: UPPER_SNAKE_CASE
  - Async functions: prefix with `async` keyword

**Enforcement:**
```python
# ✅ GOOD
async def analyze_sentiment(
    texts: List[str],
    preprocessing_config: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Analyze sentiment of texts using pre-trained BERT model.
    
    Args:
        texts: List of text strings to analyze
        preprocessing_config: Optional configuration for text cleaning
    
    Returns:
        Dictionary containing predictions, distribution, and metrics
    """
    # ── Preprocessing ────────────────────────────────────
    if preprocessing_config:
        cleaner = TextCleaner(preprocessing_config)
        preprocessed_texts = cleaner.clean_texts(texts)
    else:
        preprocessed_texts = texts
    
    # ── Model inference ──────────────────────────────────
    predictions = self._predict_with_model(preprocessed_texts)
    
    return predictions

# ❌ WRONG
async def analyze_sentiment(texts, config=None):
    if config:
        cleaned = cleaner.clean(texts)
    preds = self._predict(cleaned) # What does this do?
    return preds
```

---

### Rule 2: CRISP-ML(Q) Methodology Alignment

**Requirement:** Any model modifications or data pipeline changes MUST follow the CRISP-ML(Q) lifecycle:

1. **Data Understanding:** Validate text distribution and quality before changes
2. **Data Preparation:** Document preprocessing changes in code comments
3. **Modeling:** If changing models:
   - Compare with baseline (existing model)
   - Report at least 3 evaluation metrics
   - Include confidence intervals in results
4. **Evaluation:** Before deploying model changes:
   - Test on hold-out dataset (minimum 20% of data)
   - Verify metrics: Accuracy, F1-Score, AUC-ROC
   - Check for class imbalance and handle if needed
5. **Deployment:** 
   - Use version control for model checkpoints
   - Update `config.py` with new model paths
   - Test health endpoint thoroughly
6. **Monitoring:** 
   - Log all predictions with timestamps
   - Track confidence distribution over time
   - Flag performance degradation (>5% accuracy drop)

**Implementation:**
```python
# ✅ GOOD: Model change with CRISP-ML tracking
# app/services/sentiment_service.py
# NEW MODEL: mdhugol/indonesia-bert-sentiment-classification
# REASON: 5.2% accuracy improvement on hold-out set (95.8% vs 90.6%)
# METRICS:
#   - Accuracy: 95.8%
#   - F1-Score (weighted): 0.956
#   - AUC-ROC: 0.979
# TESTED ON: 500 Indonesian social media comments (hold-out)
# DEPLOYMENT: May 19, 2026

model_name = "mdhugol/indonesia-bert-sentiment-classification"
```

---

### Rule 3: Preprocessing Pipeline Consistency

**Requirement:** Text preprocessing MUST be consistent across all services:

- **Single Source of Truth:** `TextCleaner` in `app/preprocessing/text_cleaner.py`
- **Configuration Propagation:** `PreprocessingConfig` must be passed through all layers
- **No Service-Specific Preprocessing:** Each service should NOT reimplement cleaning logic
- **Defaults:** If no config provided, use sensible defaults (stemming=True, remove_stopwords=True)

**Violation Check:**
```python
# ❌ WRONG: Service-specific preprocessing
class SentimentService:
    def analyze(self, texts):
        # Don't do this:
        texts = [t.lower() for t in texts]  # Duplicates TextCleaner logic
        texts = [self.remove_stopwords(t) for t in texts]
        return self._predict(texts)

# ✅ GOOD: Use TextCleaner consistently
class SentimentService:
    async def analyze(self, texts, preprocessing_config=None):
        if preprocessing_config:
            cleaner = TextCleaner(preprocessing_config)
            preprocessed = cleaner.clean_texts(texts)
        else:
            preprocessed = texts
        return self._predict(preprocessed)
```

---

### Rule 4: Model Error Handling & Fallback Mechanisms

**Requirement:** Every service MUST implement graceful fallback mechanisms:

1. **Model Loading Failures:**
   - Log error with clear context
   - Activate fallback mode
   - Return results with "method" field indicating fallback used
   - Never leave service in broken state

2. **Inference Failures:**
   - Catch exceptions at token level (not just batch level)
   - Skip problematic texts with error logging
   - Continue processing remaining texts
   - Return partial results with error indicators

3. **GPU/Device Issues:**
   - Auto-detect GPU availability
   - Fall back to CPU if GPU unavailable
   - Log device selection

**Implementation Checklist:**
```python
# ✅ GOOD: Comprehensive error handling
class SentimentService:
    def __init__(self):
        try:
            # Attempt primary model
            self.model = AutoModel.from_pretrained(MODEL_NAME)
            logger.info(f"✅ Model loaded from {MODEL_NAME}")
        except Exception as e:
            logger.error(f"❌ Failed to load model: {str(e)}")
            self.model = None
            logger.warning("⚠️ Using rule-based fallback")
    
    def _predict_with_model(self, texts: List[str]) -> List[Dict]:
        predictions = []
        for text in texts:
            try:
                # Per-text try-catch
                result = self._single_prediction(text)
                predictions.append(result)
            except Exception as e:
                logger.error(f"Error on text: {text[:50]}... Error: {e}")
                predictions.append({
                    'text': text,
                    'sentiment': 'neutral',
                    'error': str(e),
                    'method': 'fallback'
                })
        return predictions
```

---

### Rule 5: API Contract Stability & Backward Compatibility

**Requirement:** All API endpoint changes MUST maintain backward compatibility:

1. **Response Schema Stability:**
   - Never remove fields from JSON responses
   - New fields must be optional or have defaults
   - Deprecated fields should coexist with replacements for 2+ releases

2. **Endpoint Versioning:**
   - Current: `/api/v1/analyze/sentiment`
   - Future versions: `/api/v2/analyze/sentiment`
   - Old versions maintained for minimum 3 months

3. **Request Validation:**
   - Use Pydantic validators for all inputs
   - Provide clear error messages (400 Bad Request)
   - Reject before processing (fail fast)

4. **HTTP Status Codes:**
   - 200: Success
   - 400: Invalid request
   - 503: Service unavailable (model not loaded)
   - 500: Server error

**Contract Example:**
```python
# ✅ GOOD: Maintain backward compatibility
class TextAnalysisRequest(BaseModel):
    texts: List[str]  # Required (existing)
    preprocessing_config: Optional[PreprocessingConfig] = None  # Existing
    
    # NEW FIELDS (backward compatible)
    analysis_id: Optional[str] = None  # For tracking
    priority: Optional[int] = 1  # Future load balancing
    include_explanation: Optional[bool] = False  # New feature

# Response always includes these fields
response = {
    "status": "success",  # Always present
    "analysis_type": "sentiment",  # Always present
    "results": {...},  # Always present
    
    # New fields (optional)
    "analysis_id": "abc123",
    "explanation": {...},
    "version": "1.0"
}
```

---

## 6. Service Dependencies & Version Compatibility

### 6.1 Critical Dependency Matrix

| Component | Min Version | Max Version | Reason for Constraint |
|-----------|------------|------------|----------------------|
| Python | 3.8 | 3.11 | Type hint syntax support |
| PyTorch | 2.0 | 2.2 | CUDA compatibility |
| Transformers | 4.30 | 4.40 | Model API stability |
| FastAPI | 0.100 | 0.110 | Async context manager (lifespan) |
| Pydantic | 2.0 | 2.6 | V2 API breaking changes |

### 6.2 Optional Dependencies

- **BERTopic:** Required for modern topic modeling; skipped gracefully if missing
- **HDBSCAN:** Preferred clustering; falls back to KMeans

---

## 7. Performance & Scalability Considerations

### 7.1 Inference Optimization

```
Current Approach:
├─ Batch Processing: All texts → single forward pass
├─ GPU Acceleration: Auto-detect and utilize
├─ Model Caching: Models loaded once at startup
└─ Async/Await: Non-blocking I/O

Bottleneck: Model inference (transformer forward pass)
  - BERT embedding: O(n) where n = sequence length
  - Max sequence: 512 tokens

Optimization Strategy:
├─ Implement token batching (group by input length)
├─ Use quantization (INT8) for 4x speedup
└─ Add request caching for identical inputs (TTL=5min)
```

### 7.2 Scalability Path

```
Current (Single Server):
  └─ Single uvicorn process on 0.0.0.0:8001

Recommended (Production):
  ├─ Load balancer (nginx)
  │  └─ Round-robin to 4x uvicorn workers
  │
  ├─ Shared model cache (Redis)
  │
  ├─ Async task queue (Celery + Redis)
  │  └─ Offload long-running topics analysis
  │
  └─ Monitoring (Prometheus + Grafana)
     ├─ Request latency p50/p95/p99
     ├─ Model inference time
     └─ Error rates by service
```

---

## 8. Development Workflow Guidelines

### 8.1 Adding New NLP Service

**Template to follow:**

```python
# app/services/new_service.py

from typing import List, Dict, Any, Optional
from app.preprocessing.text_cleaner import TextCleaner
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class NewService:
    """Description of service purpose"""
    
    def __init__(self):
        logger.info("Initializing New Service...")
        try:
            # Load models, set up resources
            logger.info("✅ New Service initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize: {str(e)}")
            raise
    
    async def analyze(
        self,
        texts: List[str],
        preprocessing_config: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Analyze texts.
        
        Args:
            texts: Input texts
            preprocessing_config: Optional preprocessing configuration
        
        Returns:
            Analysis results
        """
        try:
            # Preprocessing (consistent)
            if preprocessing_config:
                cleaner = TextCleaner(preprocessing_config)
                preprocessed = cleaner.clean_texts(texts)
            else:
                preprocessed = texts
            
            # Analysis logic
            results = self._core_analysis(preprocessed)
            
            return {
                'results': results,
                'summary': self._generate_summary(results),
                'metrics': self._calculate_metrics(results)
            }
            
        except Exception as e:
            logger.error(f"Analysis error: {str(e)}")
            raise
```

### 8.2 Test Structure

```
tests/
├── test_preprocessing.py
│   ├─ test_case_folding()
│   ├─ test_slang_normalization()
│   └─ test_stopword_removal()
│
├── test_sentiment_service.py
│   ├─ test_model_loading()
│   ├─ test_inference()
│   └─ test_fallback_mode()
│
├── test_aspect_service.py
│   ├─ test_extraction_modes()
│   └─ test_aspect_sentiment_correlation()
│
└── test_api_endpoints.py
    ├─ test_health_check()
    ├─ test_preprocess_endpoint()
    └─ test_analyze_endpoints()
```

---

## 9. Security & Privacy Considerations

### 9.1 Input Validation

- All text inputs validated for length (max 10,000 characters)
- No SQL injection possible (no database queries)
- No code injection (no `eval()` or `exec()`)

### 9.2 Model Security

- Models downloaded from official HuggingFace Model Hub
- No untrusted model files accepted
- Model checksums verified (when available)

### 9.3 Data Privacy

- No user data logged in production
- Predictions can be logged for evaluation only with consent
- Sensitive text fields not persisted without explicit opt-in

---

## 10. Future Enhancement Roadmap

1. **Active Learning Integration:** Label Studio integration for continuous model improvement
2. **Model Quantization:** INT8/FP16 for 2-4x inference speedup
3. **Multilingual Support:** Extend beyond Indonesian (Malay, Tagalog, Thai)
4. **Custom Model Training:** Fine-tuning endpoints for domain-specific models
5. **Batch Processing API:** Async endpoint for large-scale analysis jobs
6. **Model Distillation:** Smaller, faster models for edge deployment
7. **Explainability:** SHAP/LIME integration for prediction explanations

---

## 11. Contact & References

- **Base Model References:**
  - IndoBERT: https://github.com/indobenchmark/indobert
  - Indonesia BERT Sentiment: https://huggingface.co/mdhugol/indonesia-bert-sentiment-classification
  - BERTopic: https://maartengr.github.io/BERTopic/

- **Framework Documentation:**
  - FastAPI: https://fastapi.tiangolo.com/
  - PyTorch: https://pytorch.org/
  - Transformers: https://huggingface.co/transformers/

- **Standards:**
  - CRISP-ML(Q): https://github.com/EvalAI/crisp-dm-ml
  - PEP 8: https://www.python.org/dev/peps/pep-0008/

---

**Document Integrity:** This document represents the system state as of May 19, 2026. For latest updates, refer to the codebase repository.

