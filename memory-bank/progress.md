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
- Status: NOT STARTED
- Tasks:
  - [ ] Parse drug names from all 100 docs
  - [ ] Classify docs as usage vs side_effect
  - [ ] Build drug_index dictionary
  - [ ] Build condition_index reverse lookup
  - [ ] Clean text (lowercase for matching, preserve originals)

### Phase 2: Hybrid Retrieval Engine
- Status: NOT STARTED
- Tasks:
  - [ ] Layer 1: Exact drug match via regex
  - [ ] Layer 2: TF-IDF + cosine similarity
  - [ ] Layer 3: Condition keyword match
  - [ ] Score fusion function
  - [ ] Sibling doc linking

### Phase 3: Query Understanding
- Status: NOT STARTED
- Tasks:
  - [ ] Query classifier (keyword rules)
  - [ ] Unanswerable detection (threshold gating)

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
