# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

FastAPI service for Indonesian-language text analysis (sentiment, aspect extraction, topic modeling,
aspect–topic association). It is the NLP backend for the Laravel app in `../text-analysis-web`, which
calls it over HTTP (`NLP_API_URL`, default `http://localhost:8001`).

Note: the git repository root is `C:\Skripsi`, one level **above** this directory — it also contains
`text-analysis-web`, `Scraper`, and `Referensi`. `git status` here shows changes across all of them.

## Commands

```powershell
# Setup (Python 3.11; venv/ already exists locally and is gitignored)
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Run (either form; both read host/port/reload from app/config.py -> .env)
python run.py
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload

# Smoke test
curl http://localhost:8001/health
# Interactive docs: http://localhost:8001/docs
```

```powershell
pip install -r requirements-dev.txt   # pytest + httpx (dipisah agar image Railway tetap ramping)
python -m pytest tests/ -q            # 75 tes, ~15 detik, tidak memuat bobot model
python -m pytest tests/test_training_utils.py -q   # satu file
```

Tests deliberately avoid loading model weights, so they run without network. `httpx` must stay
`<0.28` — starlette bundled with FastAPI 0.109 calls `TestClient(app=...)`, removed in 0.28.

There is no linter or formatter configured; style is enforced by convention only (see below).

## Architecture

### Request path

`app/main.py` is the single router — no `routers/` package. On startup, the `lifespan` handler
instantiates four services into **module-level globals** (`sentiment_service`, `aspect_service`,
`topic_service`, `association_service`). Endpoints read those globals and return 503 if one is `None`.

Endpoints: `POST /api/preprocess`, `/api/analyze/{sentiment,aspect,topic,combined}`,
`/api/retrain/{aspect,sentiment}`, plus `GET /` and `/health`.

`/api/analyze/combined` **must** forward `predefined_aspects` and `mode` to `aspect_service.analyze()`.
Both live in `TextAnalysisRequest` but were silently dropped there, so every combined analysis ran in
`automatic` mode and the user's declared categories were discarded — with no error, and invisible in
the UI, which does offer the aspect-mode radio for `combined`. Laravel dropped them too, in *both*
`executeCombinedAnalysis` and the >50-text batched path (which hardcoded `'automatic'`).
(Ignore the `/api/v1/...` paths in `SYSTEM_ARCHITECTURE_SERVICE.md` §5 Rule 5 — they do not exist.)

### Lazy model loading — the central design constraint

Constructors are cheap; every service defers HuggingFace downloads/loading to `_ensure_loaded()`,
called at the top of `analyze()`. This keeps startup RAM low enough for Railway's free tier, so the
**first** request of each type pays the model-download cost (minutes). `_loaded` is set to `True` even
when loading *fails*, so a broken model degrades to the fallback path instead of retrying every call.

Because model state (including retrained weights) lives in per-process globals, the service **must run
with `--workers 1`** — this is why `Procfile`, `nixpacks.toml`, and `railway.json` all pin it.

### Fallback chains

Every service degrades rather than erroring, and tags the result with a `method`/`mode` field:

- **Sentiment** (`crypter70/IndoBERT-Sentiment-Analysis`): BERT → Indonesian
  positive/negative wordlists (`_predict_rule_based`).

  **The checkpoint is chosen, not inherited — `app/config.py: SENTIMENT_MODELS` holds four
  measured candidates** with their own temperature and review threshold, so swapping is a config
  change. Architecture was verified from each config's `_name_or_path`, never from its name:
  `ayameRushia/bert-base-indonesian-…` is *not* IndoBERT (it is `cahya/bert-base-indonesian-1.5G`)
  and `w11wo/indonesian-roberta-…` is RoBERTa. **Only `crypter70` and `mdhugol` are genuine
  `indobenchmark/indobert-base-p1`**, which the thesis proposal committed to.

  | checkpoint | base | test acc / macroF1 | ECE | behavioural |
  |---|---|---|---|---|
  | **`crypter70`** | **IndoBERT p1** | **0.9120 / 0.8855** | **0.0240** | **0.952** |
  | `mdhugol` (was default) | IndoBERT p1 | 0.9080 / 0.8762 | 0.0367 | 0.924 |
  | `w11wo` | RoBERTa | 0.9160 / 0.8853 | 0.0209 | 0.972 |
  | `ayameRushia` | BERT Cahya | 0.9180 / 0.9010 | — | 0.956 |

  **Selection used validation + behavioural tests, never the test set** — the ranking actually
  flips between splits (`ayameRushia` wins on test, `w11wo` on valid) and the spread
  (0.9020–0.9180) sits inside the ±1.3-point standard error at n=500.

  The selection evidence is `data/experiments/sentiment_models_valid.json` (SmSA **valid**,
  n=1260), where `crypter70` (0.9357 / 0.9103) beats `mdhugol` (0.9310 / 0.9030) on both
  aggregates; `sentiment_models.json` is the test-set run and is **confirmation after the fact**,
  not the basis.

  **`crypter70` does not win on every metric, and an earlier version of this file claimed it
  did.** On SmSA test `mdhugol` is better on **raw ECE** (0.0489 vs 0.0827), **positive recall**
  (0.9423 vs 0.9279) and **negative F1** (0.9526 vs 0.9504), and they tie on negative recall.
  `crypter70` wins accuracy, macro F1, weighted F1, positive F1, and both neutral figures —
  and, after temperature scaling, ECE too (0.0240 vs 0.0367), which is the number that actually
  ships. The honest statement is: **two aggregate metrics, the neutral class, calibrated ECE, and
  the behavioural suite**, against a gap of about 2 documents in 500 that is not significant. The
  decisive evidence is behavioural, not aggregate: **contrastive "X buruk tapi Y bagus" 0.417 →
  0.917** — real comments are full of that construction. Full write-up in
  `docs/METODOLOGI_ANALISIS_SENTIMEN.md`.

  **Two headline numbers exist for this model and both are correct.** 0.9120 / 0.8855 is the run
  *before* the slang-dictionary audit (`sentiment_models.json`); 0.9160 / 0.8888 is *after*
  (`sentiment_eval.json`). Quote **`sentiment_eval.json`** — it is the current pipeline. The
  candidate comparison has not been re-run since the audit, which is fine because selection was
  made on validation, but do not mix numbers from the two files in one table.

  All candidates are fine-tuned on SmSA *train*, so SmSA test is **in-domain** for all of them:
  an upper bound, not evidence of generalisation — the same caveat as the aspect module's 0.97
  in-domain vs 0.029 cross-domain. **Swapping the model does not fix the domain limitation.**

  **Label order is read from the model, never assumed** (`_resolve_label_map`). It differs across
  public checkpoints with no convention: `crypter70`/`mdhugol` use `0=positive` but
  `taufiqdp/indonesian-sentiment` uses `0=negatif` — the exact reverse. A wrong map inverts every
  result **without raising anything**. `LABEL_MAP` remains only as the fallback for configs still
  carrying `LABEL_0/1/2` (which is `mdhugol`). The retrain path uses `self.label_to_id` derived
  from the same map — training through the wrong one swaps the labels, and the loss still falls.

  **Only half that map used to be applied.** `sentiment` read `label_map`, but the `scores` dict
  was built from hardcoded indices `probs[0]/[1]/[2]`. On a checkpoint with a different order
  (`taufiqdp` is the exact reverse) the **label would be right while the probability vector was
  scrambled** — argmax of `scores` pointing at a different class than `sentiment`, with nothing
  raised. Latent rather than live, because `crypter70` and `mdhugol` happen to use 0/1/2, but
  `SENTIMENT_MODELS` exists precisely so checkpoints can be swapped. `_scores_dict()` now builds
  it through the resolved map and always emits all three keys in a fixed order (missing classes
  as 0.0), because Laravel reads all three directly.

  **Neutral recall is 0.7045 and cannot be fixed by calibration or by swapping models.** All four
  candidates land in 0.5795–0.7727 because they share SmSA's training data, where neutral is only
  10.4% of 11 000 docs — the cause is the data, not the architecture. Per-class weights were swept on SmSA *valid*: the optimum is exactly 1.0 (no
  adjustment) and every upweighting of neutral *lowers* macro F1, because the model is
  confidently wrong — "di indonesia lagi banyak bencana" scores negative at 0.9897. Errors are
  not near the decision boundary, so no decision rule separates them. Fixing this needs labelled
  data, which is what the active-learning loop is for. Do not "tune" it.

  **Empty text is never sent to the model.** The alignment contract forbids dropping blank rows,
  but IndoBERT classifies a bare `[CLS] [SEP]` as `positive` at **0.9429** — consistently, and
  meaninglessly. Every blank CSV row therefore used to inflate the positive percentage, which is
  the headline number Laravel displays. Blanks now return neutral/0.0 tagged `method: 'empty'`
  and are excluded from the distribution denominator (`metrics.total_empty` reports how many).
  The check runs **after** cleaning, because cleaning can *produce* an empty string from
  non-empty input (`"a"`, `"😀"`).

  Every prediction carries **`method`** (`indobert` / `rule-based` / `empty` / `error`) — without
  it, an IndoBERT prediction and a wordlist guess are indistinguishable to the caller, the same
  silent-degradation class that hid a total BERTopic failure in the topic module. `text` is the
  caller's **original** string and `processed_text` the cleaned one (it used to return only the
  cleaned text, so `"wkwkwk anjay bang"` came back as `"tertawa bang"`).

  **`_predict_rule_based` handles negation.** Measured on SmSA test: 0.6120 → **0.6720** accuracy
  (macro F1 0.5974 → 0.6538). The old list counted `tidak`/`kurang` as *negative words*, so
  "tidak jelek" scored two negatives and "tidak bagus" tied and came back neutral. Negators now
  flip polarity for at most `NEGATION_SCOPE = 2` following words. The ~25-point gap to IndoBERT
  is the cost of the fallback, now a known number rather than an assumption.

  **Confidence is temperature-scaled** (`settings.sentiment_temperature = 2.7748`, Guo et al.
  2017). Raw, this model is badly overconfident — measured ECE 0.0827 on test with mean
  confidence 0.9926 against 0.9120 accuracy. T is fitted by LBFGS on SmSA **valid** and reported
  on **test**: ECE 0.0827 → **0.0240** (−71%) with accuracy, macro F1 and the confusion matrix
  **identical to four decimals**. That identity is guaranteed, not lucky: dividing logits by a
  positive scalar cannot reorder them, so argmax never moves. Reproduce with `--fit-temperature`.

  **T belongs to the weights, not to the module** — 1.3057 for `mdhugol`, 1.6132 for `w11wo`,
  2.7748 for `crypter70`. It is stored per candidate in `SENTIMENT_MODELS` and applied when the
  model changes; borrowing another model's T silently miscalibrates the review queue.

  **`fit_temperature` refuses to calibrate when calibration is undefined** — found on the real
  retrain path, not anticipated. A 19-sample validation split with **zero errors** yielded
  **T = 0.308**: with nothing wrong to calibrate against, NLL keeps falling as confidence rises,
  so T slides toward zero and the model becomes maximally *overconfident* — which would silently
  empty the review queue that depends on it. It now returns 1.0 (no calibration) when the
  validation set is <50 samples, has no errors, has no correct predictions, or yields T outside
  [0.5, 5.0]. The real 1 260-sample fit still gives 1.3057, so the guards block only the
  pathological case.

  **`review_queue` is the bridge to retraining, and its threshold is chosen on validation, then
  confirmed on test.** It used to be read off the *test* table and then quoted from that same
  table — selection-on-test, so the quoted yield was an in-sample estimate with no confirmation.
  It is now derived on SmSA **valid** (n=1260, `sentiment_eval_valid.json`):

  | threshold | reviewed | share | errors caught | error recall | lift |
  |---|---|---|---|---|---|
  | 0.90 | 71 | 5.6% | 26 | 33.8% | 5.99 |
  | 0.92 | 100 | 7.9% | 38 | 49.3% | **6.22** |
  | **0.94** | **153** | **12.1%** | **51** | **66.2%** | 5.45 |
  | 0.95 | 248 | 19.7% | 64 | 83.1% | 4.22 |
  | 0.96 | 675 | 53.6% | 74 | 96.1% | 1.79 |

  `settings.sentiment_review_threshold = 0.94` — the last point before lift collapses. Held-out
  **confirmation on SmSA test** (n=500, `sentiment_eval.json`, never used to pick it): 17.0%
  reviewed → **69.1%** of errors caught, lift 4.06. Valid and test agree within 3 points, which
  is what makes the operating point credible rather than fitted.

  **The threshold is per-model and must not be copied**: 0.90 is the equivalent operating point
  on `mdhugol` (18.0% → 67.4%) but on `crypter70` it catches only 50%, because this model is more
  confident so fewer rows fall below it. The block returns indices sorted least-confident-first, aligned to the
  caller's `texts`; `empty` rows are excluded. **Calibration is per-weights**, so a retrain
  refits T on its own validation split and stores it beside the checkpoint as `calibration.json`,
  reloaded on startup.

  **Truncation and chunk failures are reported, not silent.** Text filling the full 512-token
  window is tagged `truncated: true` and counted in `metrics.total_truncated` — no project corpus
  reaches it (max measured 253 tokens on YouTube) but the API accepts 10 000 characters, and on
  reviews it is the tail that carries the verdict. A failing inference chunk used to drop **all
  256** of its texts to neutral/0.0; `_score_chunk` now halves and retries down to size 1, so only
  genuinely broken texts get `method: 'error'` (counted in `metrics.total_failed`).
- **Aspect** (`indobenchmark/indobert-base-p1` for token classification, BIO tags `O/B-ASPECT/I-ASPECT`):
  trained checkpoint (`_extract_with_model`) → zero-shot (untrained model logits merged with the
  `self.common_aspects` keyword set, keywords winning at confidence 1.0) → `rule-based` mode when the
  caller passes `predefined_aspects`. `AspectService` owns its own `SentimentService` instance to score
  per-aspect polarity.

  **`rule-based` mode is two-tier, and the second tier is what makes it aspect-category detection
  rather than string matching.** Tier 1 is literal/stem span matching over the declared aspect names.
  Tier 2 (`_semantic_matches`) runs the trained extractor to find candidate aspect *terms*, then maps
  each to the nearest declared *category* by cosine similarity against template-built prototypes.
  Measured on 6 sentences / 3 categories: mapping recall 2/6 → 6/6, with all foreign sentences still
  rejected. Turn it off with `aspect_semantic_mapping=false` to get the old literal-only behaviour.

  Three non-obvious decisions, each measured (the numbers live in `config.py` and the method
  docstrings, and re-deriving them costs a GPU-less hour):

  - **Embeddings use the *base* IndoBERT encoder (`_ensure_embedder`), not the fine-tuned K6 one**,
    even though K6 is already in RAM. BIO fine-tuning reorganises the space around "is this token an
    aspect" and destroys topical structure: 9/10 vs 7/10 accuracy, and the separation gap between
    in-category and foreign clauses collapses from +0.138 to +0.001 — i.e. no threshold works at all.
    The extra encoder is lazy-loaded, so `automatic` mode never pays for it.
  - **The candidate is represented by its *span in context* (`_embed_spans`), not by the clause.**
    Embedding the clause makes assignment topic-driven: on a corpus where every comment is about tax,
    `DPR` mapped to `pajak` (0.52) instead of `pemerintah`. Span pooling fixed exactly that case.
    Measured across two domains: span 15/17, clause 14/17, bare term 14/17.
  - **Tier-1 filtering is per-*span*, not per-document.** Skipping whole documents that already
    matched literally loses the other aspects in them: in "Pelayanannya bagus tapi antriannya lama",
    `antriannya` was never scored, so `pelayanan` came back 100% positive instead of 50/50.

  **Known limitation, and it is not a bug to be tuned away.** Precision depends on how distinct the
  declared categories are. With broad overlapping ones (`pajak`/`korupsi`/`pemerintah`/`pelayanan
  publik`) every extracted term scores in a narrow band — measured 0.254–0.428 — and `rakyat`→
  `pemerintah` (0.381) outranks the correct `dpr`→`pemerintah` (0.378). **No threshold separates them**,
  so raising `aspect_semantic_min_score` removes correct mappings before wrong ones. Distinguishing
  "DPR is part of government" from "rakyat is not" needs category-membership knowledge, not
  distributional similarity; this is why mainstream ABSA does category detection *supervised*. Every
  occurrence therefore carries `match_type` (`lexical`/`semantic`) and `similarity`, and each aspect
  carries `lexical_matches`/`semantic_matches`, so a lopsided split is visible rather than silent.
  The reliable path to precision here stays the existing active-learning loop.

  `_find_aspect_spans` matches literally first, then falls back to Sastrawi stem matching so
  `pejabat` finds `pejabatnya` (Indonesian is agglutinative; literal matching missed 4.3% of real
  corrections). `_is_verbal_derivation` rejects matches whose surface form carries a verbal prefix
  (me-, di-, ter-, ber-) the aspect lacks — otherwise `harga` would match `menghargai`, teaching the
  model to tag verbs. Stemming is used **only** here, never on the sentiment path.

  **Inflected variants are merged into one aspect before anything downstream runs**
  (`_merge_aspect_variants`, `settings.aspect_merge_variants`, default on). Extraction returns the
  word as written, so one aspect used to split across entries — `gaji` and `gajinya`,
  `pelayanan` and `pelayanannya` — giving the dashboard duplicate rows with **the count and the
  per-aspect sentiment split between them**. Found in the TNI case study, not in testing. Merging
  runs immediately after extraction so sentiment, statistics, and `document_aspects` all inherit
  one list; merging afterwards would leave the per-aspect statistics split.

  Two mechanisms, deliberately separate:

  - **Possessive clitics (`-nya`/`-ku`/`-mu`) are stripped without a dictionary.** Pure
    inflection — never changes meaning or word class. This is what handles vocabulary Sastrawi
    does not know, and institutional corpora are full of it: measured, `alutsistanya` and
    `danramilnya` come back from the stemmer **unchanged**, because Sastrawi only strips an affix
    when the result is one of its 29 933 roots. **The guard is the dictionary, with length only as
    a backstop**: if the whole word is itself a root, the trailing `-nya` is not a clitic. A
    length rule alone is provably wrong — of the 13 Sastrawi roots ending in `-nya`, six clear any
    sane length bar (`makanya`→`maka`, `empunya`→`empu`, `bahwasanya`→`bahwasa`, plus
    `mangkanya`, `adakalanya`, `segianya`). With the dictionary first, the remainder bar can drop
    to 3 characters, which is what lets the acronyms that fill institutional corpora work —
    `tninya`→`tni`, `kpknya`→`kpk`. Clitics written as a separate word are dropped too
    (`gaji nya` → `gaji`): found in the six-institution case study, where the spaced form never
    merged with its parent and left `gaji` and `gaji nya` as two dashboard rows.
  - **Sastrawi stems handle the rest, guarded by prefix class.** The stem alone flattens
    agent-forming prefixes: `petugas` and `tugas` both stem to `tugas`, but they are different
    aspects — the officer versus the duty. Comparing prefix class first keeps
    `petugas`+`petugasnya` together while `tugasnya` stands alone. Without the guard, merging
    would introduce a new error class while fixing the old one.

  **The displayed label is the most frequent surface form with clitics stripped, never the stem** —
  same principle as `_make_words_readable()` in the topic module: modelling on stems is right,
  showing users `layan` when they wrote `pelayanan` is not. Ties break by shortest then
  alphabetical, so repeated analyses of one corpus produce the same label. `variants` and
  `merged_from` list what was folded in, so a wrong merge is visible rather than silent.

  **Rule-based mode is never merged** — those categories are the user's own declaration.

  Known residual, not removed by the prefix guard: forms differing only by a derivational
  *suffix* still merge, e.g. `pendidikan` (the process) with `pendidik` (the person), both stem
  `didik` with prefix `pen`. Separating those needs word-class knowledge, not surface morphology.
  Phrases are also not folded into their head noun — `anggaran` and `anggaran pertahanan` stay
  distinct, which is deliberate: collapsing them would discard real information.

  **`document_aspects` carries surface forms, so comparing it to human labels needs
  `document_aspects_normalized`.** In `automatic` mode the extractor returns the word as written
  (`pelayanannya`, `pajaknya`) while an annotator writes the base form (`pelayanan`, `pajak`).
  Indonesian is agglutinative, so exact-string scoring counts a **correct** pair as a false positive
  *and* a false negative. Measured on 6 sentences / 8 gold aspects, same `document_aspects` output:
  micro-F1 **0.118** (TP=1 FP=8 FN=7) exact-string vs **0.941** (TP=8 FP=1 FN=0) normalised — the
  model finds the right aspects either way; the gap is entirely a measurement artefact. This was
  live in Laravel's `ClassificationMetrics::normalizeSet()`, which would have reported the aspect
  module as barely working. `normalize_aspect_name()` reuses the **same cached Sastrawi stemmer as
  `_find_aspect_spans`**, so evaluation matches by the same definition extraction does.
  `POST /api/normalize/aspects` normalises the other side (gold labels) — **both sides must be
  normalised**, since the stem need not equal the human base form (`pelayanan` → `layan`); what
  matters is that they land on the same token. It is idempotent, so normalising an already-base
  label is safe. The surface form stays in `document_aspects` because that is the word the user
  actually wrote, and it is what the UI displays.

  **Per-aspect polarity is computed on the clause, not the sentence.** `_clause_around()` splits on
  sentence punctuation *and* contrastive connectives (`tapi`, `namun`, `padahal`, `sayangnya`, …),
  then strips a leading connective — `padahal fasilitas umum memadai` reads negative to IndoBERT
  while `fasilitas umum memadai` reads positive at 0.99. Negation words are deliberately **not**
  stripped. Without this the whole point of ABSA is lost: measured before the fix, every aspect in
  "Pelayanannya bagus tapi harganya mahal" came back `negative` because the sentence-level sentiment
  was copied onto each aspect. The clause falls back to the full text when it contains nothing beyond
  the aspect itself (compared against the aspect's own word count, not a fixed threshold).

  Very short clauses can lose the context the model needs, so
  `_resolve_low_confidence` re-scores any occurrence whose clause prediction fell below
  `settings.aspect_clause_min_confidence` (0.6) using the full text — measured: `"Pajak naik"` reads
  positive at 0.53, `"Pajak naik terus"` negative at 0.73. Only the low-confidence subset is
  recomputed, so the extra pass is cheap.

  All clauses across all aspects go through **one** sentiment call, and
  `SentimentService._predict_with_model` sorts by length before batching (dynamic padding pads a batch
  to its longest member, so one long comment used to inflate 255 short ones). Measured on 331 clauses:
  19.6s → 11.3s; end-to-end for 80 comments 24.9s → 18.5s. Output order is restored to input order —
  every positional contract downstream depends on it.
- **Topic**: BERTopic + IndoBERT embeddings (UMAP + HDBSCAN; KMeans if hdbscan is missing) → sklearn
  LDA over **word counts** → raw word-frequency buckets. BERTopic also falls back to LDA when every
  document lands in the `-1` outlier cluster.

  **Import order is load-bearing here too.** `topic_service.py` imports `torch` before
  `text_cleaner`, because bertopic pulls torch via sentence-transformers and text_cleaner pulls nltk —
  nltk-before-torch is the WinError 1114 crash. The module used to import text_cleaner first and
  survived only because `main.py` happened to load the torch-importing services earlier; that broke the
  moment a test imported `topic_service` on its own.

  **UMAP/HDBSCAN parameters are measured, not derived from a formula.** The old
  `min_cluster_size = n // 20`, `min_samples = n // 50` gave 44/17 on 885 comments — far too coarse.
  `scripts/sweep_topic_params.py` (embeddings computed once, reused across ~100 combinations) settled
  it; `scripts/eval_topics.py` measures the end-to-end pipeline and appends to
  `data/experiments/topic_results.json`:

  | config | topics | c_v | diversity | outliers | largest topic |
  |---|---|---|---|---|---|
  | 44/17/44 (old) | 3 | 0.390 | 0.90 | 34% | **53%** |
  | 20/3/15 (now) | 15 | **0.505** | 0.98 | 18% | 14% |

  `min_samples` is pinned at **3**: low values consistently cut outliers without costing coherence,
  while 17 pushed outliers past 50% in the sweep.

  **Order of the two reductions matters.** `reduce_topics()` rewrites `topic_model.topics_`, so
  outlier reduction must run *after* it — doing it first silently discards the result. With the right
  order, outliers drop 32.3% → 9.6% on 885 comments, and near 0% on smaller corpora. Outliers are not
  cosmetic: every `-1` document is also dropped from the aspect–topic PMI.

  **The `bag_of_words` cleaning policy used to be skipped entirely when the caller sent no
  `preprocessing_config`.** The whole cleaning block sat inside `if config_dict:`, so topic
  modelling ran on **raw text** — no stopwords, no stemming, no slang normalisation — despite this
  file documenting the profile as mandatory for this path. Laravel's `PreprocessingConfigResolver`
  always returns a config (it has a `FALLBACK`), so **production was never affected**; every other
  caller was — `/api/analyze/topic` without a config, the evaluation scripts, and any measurement
  driven from Python. That last one is the dangerous part: **numbers measured through that path do
  not describe what users see**, and it silently invalidated a threshold sweep run in this repo
  before it was caught. Cleaning is now unconditional (`config_dict or {}`).

  **Topic *labels* are drawn from a frequency-restricted vocabulary** (`_label_vocabulary`,
  `settings.topic_label_min_freq = 3`). Found by inspecting real output, not by testing: of 18
  topics on the production corpus, **six were labelled by words occurring exactly once** —
  `chuaaakksss`, `preetttt`, `membaaaanguuuuun`, `indonesiiiiaaaa`, `mahpuuudd`, `gatel`. Those
  are typo and character-elongation clusters, not topics.

  The cause is structural, not bad luck: **c-TF-IDF gives its highest weight to rare words**, so a
  hapax wins its own topic outright, and MMR reinforces it because a rare word always looks
  "complementary". Adding stopwords cannot fix this — misspellings cannot be enumerated.

  Measured on the production path (1 139 comments, cleaning applied, auto-k). "Junk" counts a
  topic whose top-8 labels are at least half words occurring ≤2 times in the corpus:

  | | filter off | `min_freq = 3` |
  |---|---|---|
  | c_v | 0.5003 | 0.5101 |
  | diversity | 0.975 | 0.806 |
  | topics | 20 | 17 |
  | **junk topics** | **14 of 20** | **2 of 17** |
  | rare share of label words | 65% | 14% |

  **The headline finding is that c_v did not move** (+0.0098, inside the ±0.0221 seed variance)
  while 14 of 20 topics were unreadable — labels like `nagara, ndhak, pongrekun, planga, mencle`
  and `kuruptor, lillahi, ngeo, ugalan`. **Coherence is blind to this failure by construction**:
  c_v rewards rare co-occurring words, which is exactly what a typo cluster is. Diversity is worse
  than blind — it was *inflated* to 0.975 because every misspelling is unique. So both quality
  numbers looked healthy while most topics were noise. This is the same silent-degradation class
  as the BERTopic total failure that only `method` revealed: **judge topic labels by reading them,
  never by the quality block alone.**

  `min_freq = 3` is chosen where junk topics collapse, not where c_v peaks. On a separate sweep,
  5 scored higher c_v (0.5106 vs 0.4892) but cost 0.105 diversity for +0.021 coherence — buying
  the number by repeating the same words across topics.

  **Where the filter is applied is load-bearing, and the obvious place is wrong.** The first
  version passed the filtered vocabulary as `vocabulary=` on the c-TF-IDF `CountVectorizer`,
  reasoning that c-TF-IDF only produces labels. **That reasoning is false**: BERTopic also uses
  c-TF-IDF to *merge similar topics* and to *reduce outliers*, so a narrowed vocabulary makes
  topics look more alike and more of them get merged. Measured on news, n=250: **9 topics
  (largest 22.4%) collapsed to 2 (largest 59.6%)**, and `scripts/robustness_topics.py` fell from
  14/14 to 13/14. Isolated by ablation — disabling the filter restored 9 topics; disabling the
  degeneracy floor changed nothing.

  The filter therefore runs in `_filter_label_words()`, over the word list BERTopic *returns*,
  after the model is finished. There it genuinely cannot move a document or merge a topic.
  Moving it restored 14/14 with news n=250 back to 9 topics / c_v 0.4478 / largest 22.4%.
  If filtering would empty a topic's label entirely, the unfiltered words are kept — a rare
  label beats no label.

  Two lessons worth keeping. First, `min_df` cannot express this at all: BERTopic fits the
  c-TF-IDF vectorizer on the **per-topic joined documents**, so "df" there means "how many
  topics", not "how often in the corpus". Second, the tests that guarded the first version
  asserted on *source text* (`assert 'if str(w).strip()' in sumber`) and passed happily through a
  design error — they could not have caught it. `_filter_label_words` is now tested behaviourally
  instead. Contrast `filter_rare_tokens()`, which strips tokens from documents *before*
  modelling and was measured harmful (c_v 0.4602 → 0.3940 at n=120) — see `topic_min_token_freq`.

  Two guards, for the same reason `min_df` is conditional: off below
  `topic_label_min_docs = 50` (on a small corpus nearly all vocabulary is hapax, and any threshold
  removes the most important word), and cancelled if fewer than `topic_label_min_vocab = 50` terms
  survive (labels from a too-thin vocabulary are worse than unfiltered ones).

  `TOPIC_KEYWORD_STOPWORDS` also gained **chat abbreviations** (`dgn`, `utk`, `tsb`, `sdh`, `klo`,
  `bgt`, `lg`, `smpe`, …). These are function words in disguise: they explain no topic, but slip
  into labels precisely because the abbreviated spelling is rare. One whole production topic was
  labelled `tsb, dgn, sdh, utk`.

  **Two stopword layers, deliberately different.** `TextCleaner` keeps negations (removing them flips
  polarity downstream), but `TOPIC_KEYWORD_STOPWORDS` strips them plus generic evaluatives from the
  **c-TF-IDF vectorizer only** — that changes topic *labels*, never document assignment. Without it
  the top words were `tidak`, `susah`, `hahaha`, `ngapain`. `ngram_range` is `(1,1)`: bigrams on short
  comments produced `negara negara` and both `utang diskon` and `diskon utang` in one topic, and cost
  0.14 c_v. `max_df` is only applied at n≥50 — as a fraction it is unstable on small corpora and drops
  the corpus's most important word.

  **Topic words are displayed unstemmed.** Modelling on stems is right (it merges
  bayar/membayar/dibayar), but the labels users saw were `jabat`, `anggar`, `tri`.
  `_make_words_readable()` maps each stem back to its most frequent surface form (`pejabat`,
  `anggaran`, `mentri`) and keeps the stems in `words_stemmed` — coherence **must** be scored on the
  same vocabulary as the reference corpus, so anything measuring quality reads `words_stemmed`.
  Candidate tokens are pre-filtered by substring (Indonesian affixation is concatenative) plus a
  `w[1:]` variant for nasal mutation, without which `pimpin` never reaches `pemimpin`.

  **Quality metrics ship in the response** under `quality` (`c_v`, `c_npmi`, `diversity`,
  `outlier_rate`), computed by `app/utils/topic_metrics.py` — implemented locally because
  `gensim==4.3.2` **cannot be imported** on this environment (`cannot import name 'triu' from
  'scipy.linalg'`; incompatible with scipy ≥1.13) and pinning scipy down would drag scikit-learn and
  BERTopic with it. Read c_v as the primary number (Röder et al. 2015 validated it against human
  judgement); c_npmi rewards *frequent* words, so it moves against c_v whenever generic high-frequency
  terms are filtered out of the labels — do not tune on it alone.

  **Granularity is where coherence lives, and `num_topics` caps it.** The Laravel form's hint
  "Rekomendasi: 3-7 topik untuk hasil optimal" is empirically wrong for this corpus — more topics
  score better on both coherence and diversity.

  **`num_topics=0` selects k automatically by maximising c_v under a balance constraint**
  (`_select_num_topics`). Maximising coherence alone is unsafe: measured on 885 comments the *highest*
  c_v is at k=8 (0.5182) where one topic swallows 58% of the corpus — coherent because it represents
  everything. The constraint is **relative to k** (largest topic ≤ 3.5x the balanced 1/k share), not an
  absolute threshold: an absolute 0.35 is unsatisfiable at k=2, where the smallest possible share is
  0.5. Auto mode picks k=18 and lands at c_v 0.5123, largest topic 15.5%.

  Because `reduce_topics()` is destructive and only ever reduces, the search leaves the model at the
  lowest k tried, so the model is refitted from **cached embeddings** afterwards. `topic_model.nr_topics`
  **must** be cleared before that refit — BERTopic stores it and re-applies it on `fit_transform`,
  which silently pinned the result to the search's last k (3 topics, one at 78%).

  **Levers, measured, in order of size** (`scripts/bench_topic_*.py`, `sweep_topic_*.py`):
  representation model is the big one — `reduce_frequent_words` alone is +0.140 c_v and MMR(0.3) adds
  +0.032; KeyBERTInspired *lowers* both c_v and diversity. Encoder choice is nearly irrelevant: at
  equal k, multilingual-e5-small scores 0.4765 vs IndoBERT's 0.4658, so IndoBERT stays (already cached
  for the other modules, no extra Railway download).

  **Higher coherence is not automatically better topics.** Slang normalisation *lowered* c_v
  (0.5477 → 0.5135) and diversity, yet was kept: without it one "topic" was
  `nagih, klo, bnyak, krna, bkin, jngn` — a **register cluster**, comments grouped by spelling rather
  than content, which c_v rewards precisely because non-standard spellings are rare. Any coherence
  number has to be checked against the words it scored.

  **Two vectorizer configurations, not one — `_build_vectorizer(..., for_ctfidf=)`.** BERTopic fits the
  CountVectorizer on the **per-topic joined documents**, so the document count it sees equals the number
  of *topics*, not texts. With `min_df=2, max_df=0.85`, a corpus that yields 2 topics gives max_df=1
  document against min_df=2 and sklearn raises `max_df corresponds to < documents than min_df`. The whole
  BERTopic path then failed, LDA failed too (`num_topics=0` reached `n_components`), and the result fell
  through to the crude frequency fallback — **with nothing in the API output to say so**. Only the
  per-topic `method` field revealed it. c-TF-IDF therefore uses `min_df=1, max_df=1.0` and leaves
  frequent-word suppression to `ClassTfidfTransformer`; the LDA path keeps the real thresholds.

  **Degeneracy is checked on the FINAL assignment, then KMeans is tried.** HDBSCAN is density-based and
  collapses on small or poorly-separated corpora, returning few clusters with one swallowing the corpus.
  `_is_degenerate()` flags "<3 clusters" or "one cluster >50%", and `_cluster_with_kmeans()` refits — the
  result is kept only if it is *not* degenerate, so one problem is never traded for another. The check
  runs after `reduce_topics` + `reduce_outliers`, because most of the imbalance appears only there:
  outliers are absorbed into the nearest topic and the biggest topic absorbs the most. Checking the raw
  clusters let three cases through.

  **Robustness is measured, not assumed** (`scripts/robustness_topics.py`): 3 domains (social comments,
  news articles, hotel/product reviews) x 5 corpus sizes. Was 9/14 passing, now **14/14**; no topic
  exceeds half the corpus and worst-case outliers fell 29% → 14%. Average c_v is unchanged (0.4473 vs
  0.4496) — these fixes removed silent failures rather than raising a number. Re-run it after any change
  to clustering, preprocessing, or the vectorizer.

  **c_v is strongly corpus-size dependent, which changes how every c_v number must be read.**
  Same corpus, same config, only n differs: **n=300 → 0.3597**, **n=1139 → 0.4498**. The gap
  (0.09) is four times the seed variance below, and it crosses this project's own
  "weak"/"good" interpretation bands. So a c_v from one corpus size is not comparable to a c_v
  from another, and coherence must never be measured on a *sample* of a corpus when the claim is
  about the corpus — this is why the Laravel hold-out evaluation must compute it on the full
  case-study corpus and not on the 300-row golden dataset
  (`docs/TINJAUAN_EVALUASI_HOLDOUT.md` §2.2).

  **All 18 configuration variants were tuned on the same 885-doc corpus, so that number is
  in-sample — but the tuning was checked and it transfers.** Measured at **matched n=177**
  (`data/experiments/topic_heldout.json`), because comparing the 177-doc held-out set against the
  885-doc tuning number would confound overfitting with the size effect above:

  | corpus | c_v |
  |---|---|
  | `heldout_gold.json` (never used for tuning) | **0.3714** |
  | tuning-corpus subsamples, 3 seeds | 0.3180 / 0.3986 / 0.3859 → mean **0.3675** |

  The held-out corpus scores **+0.0039** *above* the corpus the config was tuned on, and that gap
  is an order of magnitude smaller than the spread across tuning subsamples (0.318–0.399). The
  configuration is not fitted to the tuning corpus. Report the 885-doc number as the headline with
  this as the generalisation check — not the bare held-out number, which is low only because
  n=177.

  **Seed variance is measured** (`scripts/seed_variance_topics.py`, 5 seeds): c_v **0.5023 ± 0.0221**
  and diversity 0.9498 ± 0.0314 are stable — far tighter than the deltas the design decisions rest on
  (+0.140 reduce_frequent_words, +0.032 MMR, +0.086 baseline→final), so those are not seed noise. The
  **topic count is not stable**: 6–19 topics, 13.6 ± 5.0, because the c_v-vs-k surface is flat and a
  slightly different UMAP projection moves the peak. Pin `num_topics` when analyses must be comparable
  over time; auto mode is for exploration.

  **Re-tune clustering after changing preprocessing.** Normalisation merged word variants, densifying
  the embedding space; the previously-optimal 20/3/15 then produced 9 topics with one at 52%. The
  re-swept optimum is 15/2/15.

### Preprocessing — the policy differs per task, deliberately

`app/preprocessing/text_cleaner.py` (`TextCleaner`) is the **single source of truth** — Sastrawi
stemmer + Sastrawi/NLTK stopwords, plus large inline slang and typo dictionaries, emoticon removal,
and repeated-char normalization. It accepts a dict *or* a pydantic model. Services must not
reimplement cleaning.

**The per-task policy lives in `PROFILES`, not in the services.** Use
`TextCleaner.for_task('transformer' | 'bag_of_words' | 'span', user_config)`. Profile keys always
win over the caller's config, and the override is logged — that is what stops one module's needs
from silently changing another's behaviour. `'span'` returns **`None`**: aspect extraction is
offset-based, so any cleaning would shift every character offset, and the entry records that
decision rather than leaving it implicit.

**`text_cleaner.py` imports torch before nltk itself.** It used to rely on its caller for that, and
survived only because `main.py` happened to load a torch-importing service first — the moment a test
imported it alone, the whole collection died with `WinError 1114`. Same fix as `topic_service.py`.

What each service may do with it is **not** uniform, and the differences are load-bearing:

- **Sentiment** runs `_sanitize_config()` first, which force-disables `stemming`, `remove_stopwords`,
  and `lemmatization` no matter what the caller sends. IndoBERT tokenises to subwords and was
  pretrained on natural text, so bag-of-words cleaning removes the signal it uses. Measured: 5 of 6
  negated sentences flipped `negative → positive` at ~0.99 confidence
  ("pelayanannya tidak bagus sama sekali" → "layan bagus"). Slang/typo/repeat normalisation is kept —
  it moves social-media text *toward* the pretraining distribution.

  **The guard is verified end to end, not just asserted.** In the SmSA ablation
  (`evaluate_sentiment.py --ablation`) the Laravel config — which *does* request stemming and
  stopword removal — scores **identically to four decimals** (0.9120 / 0.8855) to a config that
  doesn't. That equality is only possible if `_sanitize_config` really is stripping them, and it
  holds on both models tested.

  **The other cleaning steps move accuracy by less than the noise, in model-dependent
  directions** — so do not tune them. On `mdhugol` any cleaning cost 0.9180 → 0.9040; on
  `crypter70` it *helps*, 0.9060 → 0.9140. Slang normalisation raises `mdhugol` (+0.4) and
  slightly lowers `crypter70` (−0.2). Every one of these deltas is 1–5 documents out of 500,
  inside the ±1.3-point standard error. What is load-bearing is the negation guard (a
  correctness issue, 5 of 6 sentences flip), not these second-order choices.

  Normalisation is kept on **domain** grounds, not SmSA numbers: it touches **5.0%** of SmSA
  tokens but **11.2%** of YouTube tokens — 2.2x more. Tuning a policy on a corpus where the
  effect is half-sized is exactly the mistake documented in the topic module, where normalisation
  *lowered* c_v yet removed a register cluster.
- **Aspect** does not clean at all. Extraction is span-based, so character offsets must address the
  original text. `analyze()` still accepts `preprocessing_config` for API compatibility and ignores it.
- **Topic** uses `for_task('bag_of_words')`, which forces stemming and stopword removal **on** —
  the exact opposite of the sentiment path. Aggressive cleaning genuinely helps bag-of-words models.

**Rare-token filtering was tried, measured, and rejected as a default.** The problem is real: on 885
comments **63.4% of the vocabulary occurs exactly once** (1 523 of 2 403 types), and since c-TF-IDF
rewards rare words, misspellings can float up into topic labels — one run produced a topic reading
`mahakuasah, majak, kendaran, rakya`, a typo cluster rather than a meaning cluster.
`TextCleaner.filter_rare_tokens()` implements the fix, but it costs more than it buys:

| n docs | c_v without | c_v with | tokens dropped |
|---|---|---|---|
| 60 | **0.4884** | 0.3228 | 30.9% |
| 120 | **0.4602** | 0.3940 | 27.3% |
| 250 | **0.4972** | 0.4244 | 21.7% |
| 885 | 0.5139 | 0.5205 | 15.6% |

It helps only at 885 docs, by +0.0066 — well inside the ±0.0221 seed variance, so not a real gain —
while below that it is clearly harmful, and on 20 documents it strips **half the tokens**. Diversity
falls at every size (0.930 → 0.686 at n=120). `settings.topic_min_token_freq` therefore defaults to
**1 (off)**; the function stays for corpora far larger than this project's.

`NEGATION_WORDS` (tidak/bukan/jangan/gak/…) is subtracted from the stopword set whenever
`protect_negation` is on (the default). Sastrawi and NLTK both classify negators as stopwords, which
silently inverts polarity for any consumer that removes them.

**The slang dictionary was audited and cut from 313 to 265 entries.** It had been doing four things
that are not normalisation, each measured across the project's four corpora:

- **Bare digits were mapped to words** — `'9'→'yang'`, `'4'→'untuk'`, `'8'→'delapan'`. So
  `"sesuai pasal 4 ayat 9"` became `"sesuai pasal untuk ayat yang"`. **161 tokens in 145 documents.**
  Only leet patterns containing letters survive (`j4d1`, `4ku`); those cannot be numbers.
- **Standard words were swapped for synonyms** — `mantap`/`keren`→`bagus`, `jelek`→`buruk`. That is
  meaning substitution, and on the transformer path the model stops seeing the token the author
  actually wrote (`mantap` is a much stronger positive marker than `bagus`). **178 tokens / 163 docs.**
  Rule now: **the key must be a non-standard form.**
- **Base forms were mapped to affixed forms** — `cari→mencari`, `lihat→melihat`, `coba→mencoba`;
  backwards, and then stemming strips them back again. **207 tokens / 187 docs.**
- **Ambiguous words were forced to one sense** — `lagi→sedang` (it also means "again"), `mana→dimana`,
  `aku→saya` (`aku` is standard). `lagi` alone hit **604 tokens in 514 documents**, the largest of all.

Plus 11 identity entries (`'beli'→'beli'`) and 14 mappings to `''`. The empty mappings were a real
policy leak: they deleted words **even when `remove_stopwords=False`**, which is exactly the sentiment
path. Those particles now live in `_get_social_media_stopwords()` so their removal obeys the policy.

Net effect on sentiment, measured on SmSA test: accuracy **0.9120 → 0.9160**, macro F1
**0.8855 → 0.8888**, ECE **0.0240 → 0.0209**. Behavioural suite unchanged at 237/249.

**Repeated-character normalisation was eating numbers.** The pattern `(.){2,}` matched any
character, digits included: `"utang 1000 triliun"` became `"utang 10 triliun"` and
`"anggaran 100000"` became `"anggaran 10"` — on a corpus that is *about* tax and debt figures. It is
now `([a-zA-Z]){2,}`. Collapsing to a single letter also broke genuine double letters
(`"maaaaf"` → `"maf"`), so the choice is now **dictionary-driven**: the two-letter form wins if the
resulting word is in Sastrawi's 29 933 root words, otherwise it collapses to one. Measured:
`bagusss`→`bagus`, `maaaaf`→`maaf`, `saaangat`→`sangat`, `1000`→`1000`.

**Hashtags were deleted whole.** `#\w+` removed the tag *and its word*, so
`"baca #pajak dan #korupsi"` became `"baca dan"` — the most topical words lost, in the module that
needs them most. Only the `#` is stripped now. Mentions (`@budi`) are still removed entirely.

**Two config options were dead and are now real.** `remove_punctuation` was read in `__init__` and
never applied — punctuation was always stripped, which matters because IndoBERT was pretrained on
punctuated text. `lemmatization` is *still* not implemented (there is no established Indonesian
lemmatizer; Sastrawi is a stemmer) but now **logs a warning** instead of being silently ignored.
`fix_typos` also gained phrase matching: multi-word keys like `'terima kasi'` could never match a
`split()`-based lookup, so those entries were dead.

**Import-order hazard:** on Windows, importing `nltk` before `torch` raises
`OSError WinError 1114` loading `c10.dll`. Both services import torch/transformers before
`text_cleaner`, which is what keeps startup working — do not import `text_cleaner` first in new modules.

### Association (combined endpoint only)

`AssociationService` computes PMI between aspects and topics. It is wired only inside
`/api/analyze/combined`, and depends on a positional contract: `aspect_result["document_aspects"]` and
`topic_result["document_topics"]` must both be **index-aligned to the input `texts` list**, same length,
with `-1` marking topic outliers.

**The contract is now enforced end to end: N texts in, N results out.** Two places used to silently
drop rows — `TextAnalysisRequest.validate_texts` filtered blanks out of the list, and
`TopicService.analyze` filtered texts that cleaned to empty before modelling. Either one shifts every
index after a blank row, so Laravel stored the wrong `original_text` against each prediction (its
`saveResults` maps positionally onto `raw_data`). Blanks are now kept in place and come back as `[]`
and `-1`; `TopicService` models only the non-empty subset and scatters the topic ids back to their
original positions. Keep this property when touching any service: **never filter the `texts` list.** Any change to filtering or ordering in `AspectService` or
`TopicService` must preserve that alignment or association returns `{"error": "Length mismatch"}`.

`AssociationService._aspect_name` accepts **both** shapes (plain strings, which is what
`_build_document_aspects` actually returns, and `{'aspect': ...}` dicts). This was a real outage: the
service used to call `.get()` on strings, so every combined request returned
`association: {"error": ...}`, invisible because Laravel never read the block.
`POST /api/analyze/association` exposes the same computation standalone, for Laravel's batched
(>50 texts) path where sentiment/aspect/topic are called separately.

### Memory safety

Batched inference is chunked, never done in one pass: sentiment uses 256 texts/chunk at `max_length=512`;
aspect uses `_chunked_forward_pass` at 512 texts/chunk, `max_length=128` (also the training length —
BIO label padding assumes 128). Tensors are explicitly `del`'d and CUDA cache emptied per chunk.

**Retraining must hand its memory back to the OS — `lepas_memori()` in both `retrain()` wrappers.**
glibc keeps freed pages, so after a retrain the process stayed at its training peak. Measured on
the production VPS (separate process, 12 IndoBERT training steps): peak 3 879 MB, after
`gc.collect()` still **2 890 MB**, after `malloc_trim(0)` **331 MB**. Before the fix, two
back-to-back retrains left the service at 2.4 GB RAM + 4.1 GB swap; systemd `MemoryHigh=5G`
throttled it 4.5 million times, and the hold-out evaluation that followed failed because every
`/api/analyze/aspect` request exceeded Laravel's 600 s timeout — the model was fine, the
measurement could not run. It is called in `finally` *after* `asyncio.to_thread` returns, because
only then are the optimizer and gradients of `_retrain_sync`'s frame actually free. The server
also sets `MALLOC_ARENA_MAX=2` (2 890 → 1 698 MB before trimming) and `MemoryHigh=6G`/`MemoryMax=7G`;
the drop-in is documented in `../text-analysis-web/DEPLOYMENT.md` §8.

### Retraining

**Training runs in a worker thread, and inference is locked out while it does.** Both
`retrain()` methods are thin `async` wrappers over `_retrain_sync` via `asyncio.to_thread`; neither
body contains an `await`, so the split is purely mechanical. Before this, the whole PyTorch loop ran
**inside the event loop**, so every other request — `/health` included — hung for the duration.
Invisible while retraining was an admin button pressed at a quiet moment; Laravel now triggers it
**automatically** once user corrections reach a threshold, so its timing is no longer chosen by a
human. Measured: `/health` during training went from hanging to 0.0–0.5s.

Moving it to a thread **opened a race that did not exist before**, and closing it is the other half
of the fix. Training calls `self.model.train()` — dropout on — then updates weights batch by batch,
so a concurrent analysis reads half-trained weights with dropout active and returns wrong numbers
**with no error**. Reachable, not theoretical: an `/api/analyze/sentiment` did land mid-training and
got a `200`.

`_use_models()` in `main.py` holds **one lock per model, shared by the inference and training paths**:

- **`asyncio.Lock`, not `threading.Lock`.** Inference runs on the event loop, training in a worker
  thread; a threading lock would block the loop while training holds it — reinstating the exact
  problem just fixed. The asyncio lock suspends the inference coroutine instead, so `/health` stays
  at 0.0–0.5s even with an analysis queued behind training.
- **Analysis waits; it is not refused.** A user's analysis must not fail because retraining happened
  to be running (measured: waited 3.2s, then served normally). This costs nothing in throughput —
  `analyze()` bodies are entirely synchronous, so inference *already* serialised on the event loop.
- **Zero RAM cost.** The alternative — train on a copy, swap at the end — is ~500 MB extra, exactly
  the budget that got the container killed when two sentiment models were loaded.
- `/api/analyze/topic` takes **no** lock (the topic model is never retrained through the API), so
  topic analysis keeps running at full speed during a retrain. `combined` holds both locks across
  the *whole* sequence, so one response can never mix two different models. Locks are acquired in
  fixed alphabetical order, which is what prevents deadlock between the two-lock and one-lock paths.

**Waiting is only right when the wait is short, so there is an opt-in alternative**
(`RETRAIN_REJECT_INFERENCE=true`): an analysis that needs a model *under training* gets
**503 + `Retry-After`** (estimated remaining time, clamped 30–600 s) instead of queueing.
Reason, measured with a real uvicorn server: when the client times out, Starlette does
**not** cancel the handler — the queued analysis still runs once the lock frees, with
nobody waiting for it, while Laravel has already sent a fresh copy. Training takes 13–60
minutes against Laravel's 600-second HTTP limit, so each give-up becomes phantom work
stacked right after training. Topic analysis is never rejected; analysis waiting on
another *analysis* still queues. Off by default, because a caller that does not re-queue
on `Retry-After` would turn today's successful short waits into failures. The custom
`HTTPException` handler used to **drop `exc.headers`**, which would have silently eaten
`Retry-After`; it now forwards them (tested to fail without the fix).

The same "server keeps working after the client gave up" fact applies to retraining:
Laravel's retrain HTTP timeout is 1 200 s, and a slower machine that exceeds it gets a
run recorded as `failed` while the service still saves the checkpoint. Keep the retrain
HTTP timeout above realistic training time (see `docs/JAWABAN_HOSTING_SATU_VPS.md` T1).

A second concurrent retrain of the same model gets **409**, not a queue slot — two would load,
train, and write a checkpoint to the same path. The lock is released even when training raises, so
one failure cannot wedge the endpoint until restart.

**Seven endpoints used to swallow their own status codes.** A 503 ("service not available") was
raised *inside* the `try` and caught by the `except Exception` below it, reaching Laravel as a **500**
reading `"Sentiment analysis failed: 503: ..."` — so the caller could not tell "not ready yet, retry"
from a genuine failure. `except HTTPException: raise` now precedes each handler.

All of the above is covered by `tests/test_retrain_concurrency.py`, and each test was **verified to
fail with its fix removed** — not merely to pass with it.

`/api/retrain/aspect` takes `[{text, aspects: [...]}]` and auto-derives BIO tags via `_generate_bio_tags`
(whole-word span matching, longest aspect first, `-100` on special tokens). `/api/retrain/sentiment`
takes `[{text, label}]`. Both accept `force` (default false) and share one discipline, implemented in
`app/utils/training.py` so both paths behave identically and the rules are unit-testable:

- **Seed** (`settings.retrain_seed`, default 42) is set before anything random, so split and results
  reproduce — required for numbers quoted in the thesis.
- **Stratified split** instead of torch `random_split`, so every class reaches the validation set.
  Sentiment stratifies on the label; aspect stratifies on whether a sample has any aspect token.
- **Class weights** (inverse frequency) in the sentiment loss. The correction data is ~89% negative;
  unweighted training collapses to always predicting the majority class while accuracy *rises*.
- **Measure before and after** on the same validation set, then `should_accept_checkpoint` decides.
  A checkpoint that lowers the metric (sentiment: weighted F1; aspect: aspect-token F1) is **not
  saved**, and the previous weights are restored by reloading from the untouched on-disk source —
  no `state_dict` copy, which matters on Railway's RAM budget. `force: true` overrides.
- Aspect returns a `bio_report` counting annotations that never matched their text (e.g. a correction
  of `pelayanan` on a sentence that says `layanannya`), which would otherwise silently teach the model
  that the sentence has no aspect at all.
- **Sentiment keeps the BEST epoch, not the last one.** Small, imbalanced correction data peaks at
  epoch 1–2 and then overfits, so running 3 epochs used to throw away the best result already
  reached. Each epoch is scored, the best is written to `{save_path}.best`, and the weights are
  restored from it before the accept/reject comparison — so that comparison always uses the best
  candidate, never the incidentally-last one. Saved to **disk**, not a `state_dict` copy: that
  copy costs ~500 MB, the same RAM constraint that makes rollback a reload. The temp dir is
  removed afterwards. `best_epoch` and `epoch_history` ship in the response so overfitting is visible.
- **Sentiment padding length is derived from the data, not pinned at 512.** Correction data is short
  comments and self-attention is quadratic in sequence length, so padding to 512 spent most of the
  compute on padding tokens. Measured on 92 samples, CPU, one full epoch: **296.7s → 43.6s (6.8x)**.
  On Railway's GPU-less container that is the difference between a retrain that finishes and one
  that times out. Capped at 512, rounded up to a multiple of 8.

`POST /api/retrain/preview` runs all of the above reporting **without training** — label distribution,
imbalance ratio, split composition, class weights, majority-class baseline accuracy, and warnings.
Call it before dispatching a retrain. Laravel does not use it yet.

**Checkpoints survive restarts.** Both services resolve `settings.model_path` at construction
(`sentiment_retrained/` dir, `aspect_retrained.pt` file) and fall back to the base HuggingFace model
when absent; a corrupt sentiment checkpoint falls back to base rather than to rule-based. `/health`
reports `models.{sentiment_source, sentiment_retrained, aspect_checkpoint, aspect_retrained}` so
Laravel can see which weights are live. `models/` is gitignored, so on Railway retrained weights still
vanish on redeploy unless a volume is mounted at `MODEL_PATH`.

## Conventions

- Settings come from `app/config.py` (`pydantic-settings`, reads `.env`). It also sets `HF_HOME` /
  `TRANSFORMERS_CACHE` / telemetry env vars **at import time, before transformers is imported** — keep
  `from app.config import settings` first in any module that touches HuggingFace.
- CORS origins are assembled at startup by `_build_allowed_origins()` from `settings.allowed_origins`
  plus `LARAVEL_PUBLIC_URL`, `RAILWAY_PUBLIC_DOMAIN`, and `APP_ENV=production`.
- Comments and docstrings are a Bahasa Indonesia/English mix; `summary` strings returned to the client
  are Indonesian prose meant for direct display in the Laravel UI. Match this, don't translate it away.
- Schemas use pydantic v2 idioms (`@field_validator`, `min_length`). `Settings` and
  `RetrainPreviewRequest` set `protected_namespaces=()` because `model_path` / `model_type` collide
  with pydantic's reserved `model_` prefix.
- `TextCleaner` accepts a dict *or* a pydantic model — `/api/preprocess` passes the model directly.
- `/api/preprocess` accepts an optional **`task`** (`transformer` / `bag_of_words` / `span`) so the
preview shows what a given analysis module will *actually* see, and echoes back `applied_policy`.
Without it the preview lies: a user turns on stemming, sees stemmed text, then runs sentiment
analysis which silently disables it. Omitting `task` keeps the old raw-config behaviour.

`SYSTEM_ARCHITECTURE_SERVICE.md` is the only architecture doc left here (the byte-identical
  `SYSTEM_ARCHITECTURE.md` and the misplaced `SYSTEM_ARCHITECTURE_WEB.md` were removed; both remain
  committed in `../text-analysis-web`). Its §5 states the project's style rules (PEP 8, 100-col, type hints, Google docstrings, `# ──` section
  dividers, CRISP-ML(Q) comment blocks recording metrics when swapping models) and is worth reading
  before non-trivial changes.

## Aspect training data (`scripts/`, `data/`)

Two builders produce BIO training data for `/api/retrain/aspect`. Both output a Label Studio JSON
(for human review) and a `_retrain.json` payload, and both enforce consistency mechanically — every
occurrence of an accepted term is labelled, everywhere.

- `build_aspect_dataset.py` — **in-domain**. Projects `data/aspect_lexicon.json` (34 terms verified by
  humans through the Laravel feedback UI, plus 23 vetted expansions) onto a raw corpus. Used for the
  YouTube tax/corruption comments: 76% coverage, 2 692 spans, 0.07% unmatched.
- `annotate_aspects.py` — **general domain**. Harvests candidate terms from an existing annotated file,
  normalises them (strips parentheticals, trailing years, over-long phrases), rejects the systematic
  errors LLM annotation makes (verb-headed spans, opinion words, proper names, numerals), then projects
  the survivors consistently. On the news corpus: 2 487 candidates → 2 041 accepted, 4 203 spans, 91%
  coverage.

Why not use the LLM output directly: `label_aspect.py` (the earlier Gemini script) located spans with
`str.find()`, which marks only the **first** occurrence, and batched 200 sentences per prompt. Measured
result: 56% of occurrences of its own top phrases went unlabelled — contradictory supervision for a
token classifier. After reprocessing, genuinely-unmarked occurrences are 0%; the ~30% the audit reports
as "missed" are all positions covered by a longer overlapping span, which is correct behaviour.

`data/aspect_lexicon.json` carries a `_rejected` block recording why terms were excluded (e.g.
`kebijakan` stems to `bijak`, so it wrongly matched "Rakyat **bijak** bayar pajak"). Keep that habit —
it is the audit trail for the annotation decisions.

`prepare_external_dataset.py` converts IndoNLU CoNLL files (`data/external/{terma,keps}/`) into both
a lossless `*_gold.json` (character offsets kept) and a `*_retrain.json` payload. That payload carries
a `spans` key, and `_prepare_bio_dataset` uses those offsets verbatim when present — deriving BIO from
strings instead reproduces ~105% of the gold spans, i.e. it labels occurrences the human annotator
deliberately left `O`.

`train_aspect.py` trains from any such payload; `evaluate_aspect.py` scores a checkpoint against a
`*_gold.json`, reporting both token-level and strict span-level (exact boundary) P/R/F1.

**Measured, and the reason external data matters:** the checkpoint trained on the lexicon-projected
in-domain set scores F1 0.97 on its own validation split but **0.029 span-level on TermA's test set**.
It memorised the term list. Any generalisation claim has to be backed by training on human labels and
evaluating cross-domain.

**Caveat that matters for the thesis:** both datasets are produced by projecting a term list, so a
model trained on them scores very high (F1 0.97 in-domain) partly because it is learning to reproduce
the annotation function. That number is not evidence of generalisation. Report it as such, and measure
real quality on a held-out set labelled by hand, or on an external benchmark
(IndoNLU **TermA** is the closest fit: Indonesian, span-level IOB aspect terms, 3 000/1 000/1 000).

## Sentiment behavioural testing (`scripts/behavioral_sentiment.py`)

249 templated cases in the CheckList style (Ribeiro et al. 2020) — MFT (label known by
construction), INV (surface changes that must not flip the label), DIR (changes that must move
the prediction one way). **It needs no labelled data**, which is why it exists: the production
corpus has no human annotation, so this is the only way to measure capability right now.

Current model scores **237/249 (0.952)**, up from `mdhugol`'s 230 (0.924). What the breakdown
says, and it is more useful than the total:

- **INV is 1.000 across every perturbation** — slang spelling, emoji, repeated characters, ALL
  CAPS, excess punctuation. None of them flips a label. That is exactly the register of the
  production domain, and most of the credit belongs to `TextCleaner`'s normalisation.
- **Contrastive "X buruk tapi Y bagus" was the reason to switch models**: 0.417 → **0.917**.
  Real comments are dense with that construction, so a 58% failure rate there was not academic.
- **Negation over negative words is still 0.542**, but every failure is the "kurang buruk" /
  "kurang mengecewakan" pattern, while "tidak buruk" passes 100%. "kurang + negative word" is
  awkward Indonesian, so part of that score penalises constructions native speakers rarely use.

**Do not read 0.952 as field accuracy.** Templated sentences are short, single-clause, and free
of sarcasm or context — far easier than real comments. The suite checks that basic capabilities
are intact, nothing more; field quality still needs the case-study evaluation.

Pass `--model <hf-id>` to score a candidate through the same production path, and
`--show-failures` to see what broke.

## Sentiment evaluation data (`scripts/evaluate_sentiment.py`, `data/external/`)

`data/external/smsa/{train,valid,test}_gold.json` — SmSA / IndoNLU (Prosa), 11 000 / 1 260 / 500
docs, 3 classes. This is the base model's **own** benchmark (it was fine-tuned on SmSA train), so
`test_gold.json` measures in-domain quality and `valid_gold.json` is the right place to tune any
decision rule before reporting on test — which is how the neutral-calibration sweep was run.

**The obvious cross-domain benchmark is contaminated, and this was checked rather than assumed.**
NusaX-senti (Indonesian) shares source material with SmSA: of its 400 test rows, **111 match SmSA
exactly** and **328 (82%) share an 8-gram** with it. Using it as a "cross-domain" test would report
memorisation as generalisation. `data/external/nusax/clean_gold.json` is the surviving subset after
removing exact and 6-gram overlaps — only **57 docs**. On it the model scores accuracy 0.9474 /
macro F1 0.9361, i.e. quality does not collapse off SmSA; but n=57 is a sanity check, not evidence,
and the text is still reviews rather than social media. Re-run the overlap check before adopting any
new Indonesian sentiment benchmark.

There is **no labelled data in the production domain** (YouTube tax/corruption comments). That is
precisely why `review_queue` and the active-learning loop exist, and it is the honest limitation to
state in the thesis.

`data/experiments/sentiment_errors.json` holds the highest-confidence mistakes, for qualitative
error analysis. Several are arguably annotation disagreements rather than model errors ("perempuan
yang malang itu sekarang sudah tutup usia" is labelled neutral in SmSA).

## Reliability: the timeout chain and memory budget

**The single worst defect in the system was a queue misconfiguration.** `config/queue.php` had
`retry_after = 90` while `ProcessTextAnalysis::$timeout = 1800`. Laravel requires `retry_after` to
**exceed** the job timeout; below it, the worker decides a still-running job is lost, releases it back
to the queue, and a second worker picks it up. Measured analysis times are **95–235 seconds**, so
practically *every* analysis ran two or more times concurrently. The knock-on effects explain the
"random" failures: the NLP service runs `--workers 1`, so duplicate requests serialise and slow
everything further, results overwrite each other, and the job eventually exhausts `tries = 3`.

The chain is now coherent, and `tests/Feature/QueueTimeoutConfigTest.php` asserts each link:

| layer | value | constraint |
|---|---|---|
| HTTP request to NLP (`NLP_API_TIMEOUT`) | 600s | **<** job timeout, so the job — which can log a cause and retry — is what stops stuck work |
| Job `$timeout` | 1800s | — |
| Queue `retry_after` (`DB_QUEUE_RETRY_AFTER`) | 2100s | **>** job timeout |
| Job `$uniqueFor` | 2100s | **>** job timeout |

`.env` overrode the config default (`NLP_API_TIMEOUT=3600`), so fixing `config/services.php` alone
was not enough — both files were updated. `ProcessTextAnalysis` also implements `ShouldBeUnique`
keyed on the analysis id: defence in depth against double-dispatch from the controller, a
double-clicked button, or extra workers.

**Two copies of the sentiment model were loaded in production.** `main.py` builds a global
`SentimentService` and `AspectService` used to build *another one internally* — ~500 MB duplicated
alongside the aspect model and the topic encoder. On Railway's budget that is enough to get the
container killed mid-analysis. `AspectService(sentiment_service=...)` now takes the shared instance.
This also fixes a **correctness** bug: after `/api/retrain/sentiment`, the global instance reloaded
the new weights but the aspect service's private copy did not, so per-aspect polarity silently kept
using the old model and user corrections never reached aspect analysis.

**LDA ran with `n_jobs=-1`.** joblib spawned one worker per core, each inheriting a torch-loaded
parent; under memory pressure the OS killed them and LDA fell through to the crude frequency
fallback — observed as *"A worker process managed by the executor was unexpectedly terminated"*.
LDA is already the *fallback* path on small corpora, so the parallelism bought nothing. Now `n_jobs=1`,
for the same reason the service runs `--workers 1`.

**Lazy loading is now visible and pre-emptable.** `/health` reports `weights_loaded` per model, and
`POST /api/warmup` loads them on demand (measured warm: 6.7s total). `ProcessTextAnalysis` calls
`NLPApiService::warmUp()` before analysing. Without this, a cold container's first request pays
minutes of model download and is indistinguishable from a hung analysis. Warmup failure is logged and
ignored — it is an optimisation, not a prerequisite, since every service has a fallback chain.

**A near-miss worth remembering.** Adding `low_cpu_mem_usage=True` without the `accelerate` package
made `from_pretrained` raise ImportError, which `_ensure_loaded`'s except block caught, dropping the
service to the wordlist path **with nothing surfaced to the caller**. SmSA accuracy fell
**0.9160 → 0.6740** (exactly the fallback's score) while the entire test suite stayed green, because
tests deliberately avoid loading weights. `accelerate` is now pinned in `requirements.txt`, and
`tests/test_model_loading_smoke.py` asserts a real prediction comes back with `method == 'indobert'`
— it skips cleanly when weights are unavailable, so it does not force a download on a clean checkout.

**Model-loading fixtures are module-scoped** (`tests/conftest.py`) and the suite shares one
`SentimentService`. Making the aspect fixture session-scoped was tried and made things *worse* — it
keeps the aspect model alive for the whole run so it overlaps the sentiment model, and the suite
segfaulted in 3 of 4 runs (`Windows fatal exception: access violation` inside torch tensor storage).
Module scope releases each model when its file finishes; 7 consecutive full runs pass.

**Dead settings removed or wired.** `request_timeout` was deleted (never read; timeouts belong to the
caller and the HTTP server). `max_batch_size` and `max_text_length` were defined but the schema
hard-coded `10000` instead — both are now enforced in `TextAnalysisRequest`, and `.env` was corrected
from `MAX_BATCH_SIZE=100`, which would have rejected the 885-text requests the evaluation scripts and
combined analysis rely on. `use_gpu`, `app_env`, and `log_file` are now read too.

## Keamanan (`app/security.py`)

**Sebelum modul ini, SETIAP endpoint terbuka.** `CORSMiddleware` hanya
membatasi peramban — `curl` mengabaikannya — jadi siapa pun yang tahu URL bisa
memanggil `/api/retrain/sentiment` dan menimpa bobot. Endpointnya bahkan
mendokumentasikan dirinya di `/docs`.

- **Kunci API pada semua jalur kecuali `/` dan `/health`.** Dipasang sebagai
  `dependencies=[Depends(wajib_kunci_api)]` di tingkat **aplikasi**, bukan per
  endpoint — supaya endpoint baru otomatis terlindungi. Memasangnya satu per
  satu adalah pola yang sudah pernah gagal di repo ini: `except HTTPException`
  terlewat di tujuh endpoint sekaligus. `/health` wajib terbuka karena platform
  hosting memakainya sebagai healthcheck dan tidak bisa mengirim header.
- **Gagal-tertutup saat start.** `verifikasi_startup()` melempar
  `RuntimeError` bila `APP_ENV=production` dan `API_KEY` kosong, dan ia
  dipanggil **di luar** `try` pada lifespan — konfigurasi tidak aman harus
  menghentikan start, bukan ditelan penangan galat lalu berjalan terbuka.
- **Batas laju per IP mendahului autentikasi.** Urutan itu disengaja: ia
  melindungi dari penebakan kunci dan menahan banjir permintaan sebelum
  menyentuh kode aplikasi. Konsekuensinya pemanggil yang kena kuota melihat 429
  dan bukan 401. Keadaannya di memori proses, yang **benar** di sini karena
  `--workers 1`; Redis baru perlu bila replika ditambah. `RATE_LIMIT_MAX=0`
  mematikannya. `JALUR_TERBUKA` dikecualikan — healthcheck yang kena 429
  membuat kontainer dianggap mati lalu di-restart, mengubah pembatas laju
  menjadi penyebab downtime.
- **Pemanggil berkunci sah dibebaskan dari batas laju** (`kunci_sah`). Versi
  pertama membatasi semua orang per IP — padahal satu-satunya pemanggil sah,
  Laravel, datang dari SATU IP dan memecah analisis per 50 teks: analisis TNI
  (3.366 baris) berarti ~136 permintaan. Terukur dengan pembebasan dimatikan
  dan batas 3: permintaan Laravel ke-4 sudah 429, jadi analisis besar gagal di
  tengah jalan. Tanpa kunci atau kunci salah tetap dibatasi, sehingga
  perlindungan dari penebakan kunci tidak berkurang.
- **`X-Forwarded-For` hanya dipercaya bila `TRUST_PROXY=true`.** Header itu
  ditulis klien; mempercayainya tanpa proxy di depan membuat batas laju bisa
  dilewati hanya dengan mengarang header.
- `/docs`, `/redoc`, `/openapi.json` menjadi 404 di produksi kecuali
  `EXPOSE_DOCS=true` — skema terbuka memberi tahu persis bentuk muatan
  retraining.

Tes perilaku di `tests/test_security.py`. Pembatas laju hidup di global modul,
jadi `tests/conftest.py` mereset-nya per tes — tanpa itu kuota bocor antar-tes
dan kegagalannya bergantung pada urutan eksekusi.

## Penyimpanan bobot (`app/weights_store.py`)

`models/` masuk `.gitignore`, jadi bobot hasil retraining **tidak ikut image**.
Tanpa volume di `MODEL_PATH` atau `HF_WEIGHTS_REPO`, redeploy menghapusnya dan
layanan diam-diam kembali ke bobot publik — tanpa galat, hanya angka memburuk.

Hub dipilih bukan sekadar tempat menaruh berkas: ia berbasis git, sehingga tiap
putaran retraining menjadi satu commit beserta `provenance.json`-nya. Itu jejak
audit yang lebih kuat daripada disk lokal, dan sekaligus penyimpanan bersama
yang dibutuhkan bila replika inferensi ditambah.

- `unduh_bobot()` dipanggil **sebelum** service dibangun — tiap service
  menyelesaikan path checkpoint di konstruktor dan jatuh ke bobot publik bila
  berkasnya belum ada.
- Bobot lokal yang sudah ada **tidak** ditimpa: di mesin pengembangan, disk
  lokal adalah sumber kebenaran.
- `unggah_bobot()` hanya dipanggil bila `result['saved']` — pelatihan yang
  ditolak tidak menulis bobot baru, jadi mengunggahnya akan mendorong
  checkpoint **lama** sebagai commit baru dan membuat riwayat Hub berbohong.
- Seluruh operasinya **tidak fatal**. Kegagalan unggah tidak boleh membatalkan
  pelatihan yang sudah berjalan dua puluh menit.

## Deployment

Railway via Nixpacks (`nixpacks.toml` pins python311 + gcc/openblas; `railway.json` sets healthcheck
`/health` with a 300s timeout for cold model loads). `.cache/` and `models/` are gitignored, so
HuggingFace weights are downloaded at runtime on each fresh container.
