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
- Implemented in: `rag/data.py` (295 lines), `tests/test_data.py` (138 lines), `eval_pipeline.py` (153 lines)
- Tasks:
  - [x] Parse drug names from all 100 docs (_extract_drug_name, regex: Drug [A-Z]{1,2})
  - [x] Classify docs as usage vs side_effect (_classify_doc_type, keyword matching)
  - [x] Build drug_index dictionary (build_drug_index, all 50 drugs with usage/side_effect docs, conditions, side_effects, text)
  - [x] Build condition_index reverse lookup (build_condition_index, 30+ conditions mapped)
  - [x] Build side_effect_index reverse lookup (build_side_effect_index, 60+ known side effects)
  - [x] Batch eval script (eval_pipeline.py, 10 gold queries)

### Phase 2: Hybrid Retrieval Engine
- Status: COMPLETE
- Implemented in: `rag/retrieval.py` (HybridRetriever class)
- Tasks:
  - [x] Layer 1: Exact drug match via regex (_layer_drug_match)
  - [x] Layer 2: TF-IDF + cosine similarity (_layer_tfidf)
  - [x] Layer 3: Condition + side-effect keyword match (_layer_keyword)
  - [x] Score fusion (weighted combination in retrieve())
  - [x] Sibling doc linking (_link_siblings)
  - [x] Tests: tests/test_retrieval_hybrid.py (18 tests)

### Phase 3: Query Understanding
- Status: COMPLETE
- Implemented in: `rag/retrieval.py` (classify_query function)
- Tasks:
  - [x] Query classifier: drug_specific, condition_lookup, comparison, multi_hop, safety, general
  - [x] Unanswerable detection (UNANSWERABLE_THRESHOLD = 0.10, stored in results.attrs)

### Phase 4: LLM + Prompts
- Status: NOT STARTED
- Tasks:
  - [ ] Few-shot prompt templates (3 gold examples)
  - [ ] Dynamic template per query type
  - [ ] Fallback system (template answers)
  - [ ] Test qwen vs mistral

### Phase 5: UI & Evaluation
- Status: NOT STARTED
- Tasks:
  - [ ] Gradio interface
  - [ ] Automated eval on 7 held-out queries
  - [ ] Score reporting

### Phase 6: Presentation
- Status: NOT STARTED
- Tasks:
  - [ ] Architecture diagram
  - [ ] 5-minute demo script
  - [ ] Backup screenshots

## Observations & Lessons
(Updated during development)
