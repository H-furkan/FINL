# Progress Tracker

## Implementation Status

### Phase 0: Project Setup
- Status: COMPLETE
- Tasks:
  - [x] uv project initialized (pyproject.toml, uv.lock)
  - [x] Dev tooling configured (ruff, pytest, prek)
  - [x] run_pipeline.py created (standalone script)
  - [x] tests/test_retrieval.py created (3 unit tests)
  - [x] ROADMAP.md finalized
  - [x] Memory bank populated
  - [x] jCodeMunch MCP configured

### Phase 1: Data Analysis & Preprocessing
- Status: COMPLETE
- Implemented in: `rag/data.py`, `tests/test_data.py`, `eval_pipeline.py`
- Tasks:
  - [x] Parse drug names from all 100 docs (_extract_drug_name, regex: Drug [A-Z]{1,2})
  - [x] Classify docs as usage vs side_effect (_classify_doc_type, keyword matching)
  - [x] Build drug_index dictionary (build_drug_index, all 50 drugs)
  - [x] Build condition_index reverse lookup (build_condition_index, 30+ conditions)
  - [x] Build side_effect_index reverse lookup (build_side_effect_index, 60+ effects)
  - [x] Structured CSV export (drug_knowledge.csv, export_knowledge_csv, load_knowledge_csv)
  - [x] Knowledge graph visualization (visualize_kg.py, kg_full.png)
  - [x] 50 evaluation questions (eval_questions.md, 8 categories)

### Phase 2: Hybrid Retrieval Engine
- Status: COMPLETE
- Implemented in: `rag/retrieval.py` (HybridRetriever class)
- Tasks:
  - [x] Layer 1: Exact drug match via regex (_layer_drug_match, score=1.0)
  - [x] Layer 2: TF-IDF + cosine similarity (_layer_tfidf, weight=0.5)
  - [x] Layer 3: Condition + side-effect keyword match (_layer_keyword, 0.8/0.6)
  - [x] Score fusion (weighted combination in retrieve())
  - [x] Sibling doc linking (_link_siblings, score=0.3 for siblings)
  - [x] drug_knowledge.csv integration (enriched context + fallback answers)
  - [x] Tests: tests/test_retrieval_hybrid.py (24 tests including KG CSV)

### Phase 3: Query Understanding
- Status: COMPLETE
- Implemented in: `rag/retrieval.py` (classify_query function)
- Tasks:
  - [x] Query classifier: drug_specific, condition_lookup, comparison, multi_hop, safety, general
  - [x] Unanswerable detection (UNANSWERABLE_THRESHOLD = 0.10)

### Phase 4: LLM + Prompts
- Status: COMPLETE
- Implemented in: `rag/generation.py` (RAGPipeline class), `eval_pipeline.py`
- Tasks:
  - [x] Few-shot prompt templates (3 gold examples from example_queries.md)
  - [x] Dynamic template per query type (5 templates: drug_specific, condition_lookup, comparison, multi_hop, safety)
  - [x] Enriched context: raw docs + structured drug_knowledge.csv summaries
  - [x] Fallback system (build_fallback_answer from structured CSV, 74% without LLM)
  - [x] RAGPipeline class: single .answer() method returns {answer, sources, query_type, mode, max_score}
  - [x] Eval upgraded to 50 questions with tqdm + category scoring
  - [x] Tests: tests/test_generation.py (8 tests)
  - [x] With LLM: 7/7 pass on held-out queries (95%)
  - [x] Full 50-question eval: 90% overall (47/50 passed)
    - 100%: direct_fact, usage, reverse_lookup, complex, unanswerable, multi_hop
    - 61%: comparison | 71%: safety

### Phase 5: UI (Gradio)
- Status: COMPLETE
- Implemented in: `app.py`
- Tasks:
  - [x] Gradio interface with Blocks layout + Soft theme
  - [x] Model selector dropdown: No LLM, qwen, mistral, llama-3.1 (all free)
  - [x] 10 example buttons covering all query types
  - [x] Answer + Source Documents + Query Info panels
  - [x] Pipeline caching per model (shared retriever, no rebuild)
  - [x] Auto-generates sharable public link (share=True)

### Phase 6: Presentation
- Status: NOT STARTED
- Tasks:
  - [ ] Architecture diagram
  - [ ] 5-minute demo script
  - [ ] Backup screenshots

## Test Suite: 54 tests, all passing
- tests/test_data.py: 19 tests (data parsing, indexes)
- tests/test_retrieval.py: 3 tests (basic TF-IDF retrieval)
- tests/test_retrieval_hybrid.py: 24 tests (hybrid retriever, query classifier, KG CSV)
- tests/test_generation.py: 8 tests (RAGPipeline, fallback, query types)

## Observations & Lessons
- Sibling doc linking fixed the asthma multi-hop failure from starter notebook
- drug_knowledge.csv gives cleaner LLM answers via enriched context
- Fallback mode (no LLM) achieves 74% -- strong demo point
- Q8 "drugs affect brain" misses Drug D because text says "central nervous system" not "brain"
- Comparison/safety categories weakest (61%/71%) -- broad multi-drug queries need wider retrieval
- Score fusion can exceed 1.0 when multiple layers agree (feature, not bug)
- 50-question eval takes ~16 min on free tier (~20s per LLM call)
