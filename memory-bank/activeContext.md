# Active Context

## Current Phase
**Phase 6** -- Presentation prep. Phases 1-5 complete.

## What's Been Done
- [x] Analyzed dataset structure (50 drugs x 2 docs)
- [x] Reviewed starter notebook and identified weaknesses
- [x] Created comprehensive ROADMAP.md with 6 phases
- [x] Decided on few-shot prompting over DPO (API models can't be fine-tuned)
- [x] Integrated cursor-memory-bank for cross-team AI continuity
- [x] Integrated jcodemunch-mcp for token-efficient code navigation
- [x] Set up uv project with pyproject.toml, ruff, pytest, prek
- [x] Created run_pipeline.py (standalone script version of notebook)
- [x] Created tests/test_retrieval.py (retrieval unit tests)
- [x] **Phase 1 COMPLETE** (by teammate):
  - rag/data.py: load_data, build_drug_index, build_condition_index, build_side_effect_index
  - All 50 drugs parsed, tests/test_data.py: 19 tests
- [x] **Phase 1 extras** (by teammate):
  - drug_knowledge.csv: structured CSV export of all 50 drugs
  - visualize_kg.py: knowledge graph visualization (networkx + matplotlib)
  - kg_full.png: rendered full KG image (for presentation)
  - eval_questions.md: 50 evaluation questions across 8 categories
- [x] **Phase 2 COMPLETE**: Hybrid 3-layer retrieval engine
  - rag/retrieval.py: HybridRetriever (drug match + TF-IDF + keyword, sibling linking)
  - drug_knowledge.csv integrated: enriched context + structured fallback
  - tests/test_retrieval_hybrid.py: 24 tests (including knowledge CSV integration)
- [x] **Phase 3 COMPLETE**: Query classifier
  - classify_query: drug_specific, condition_lookup, comparison, multi_hop, safety, general
  - Unanswerable detection (threshold = 0.10)
- [x] **Phase 4 COMPLETE**: LLM generation + prompts
  - rag/generation.py: RAGPipeline class (answer, fallback, source formatting)
  - Few-shot prompts (3 gold examples), dynamic templates per query type
  - Enriched context from drug_knowledge.csv sent to LLM
  - Structured fallback when LLM unavailable (74% without LLM)
  - tests/test_generation.py: 8 tests
  - eval_pipeline.py upgraded: 50 questions, tqdm progress, category scoring
  - 54 total tests, all passing

- [x] **Phase 5 COMPLETE**: Gradio UI
  - app.py: full Gradio interface with model selector dropdown
  - 4 modes: No LLM (retrieval only), qwen, mistral, llama-3.1
  - 10 example buttons, answer + sources + query info panels
  - Pipelines cached per model (no retriever rebuild on switch)
  - Run: `python3 app.py` (auto-generates sharable link)

## 50-Question Eval Results (full pipeline with LLM)
- Overall: **90% (47/50 passed)**
- Direct Fact: 100% | Usage: 100% | Reverse Lookup: 100%
- Complex: 100% | Unanswerable: 100% | Multi-Hop: 100%
- Comparison: 61% | Safety: 71%

## What's Next
- [ ] Phase 6: Presentation prep

## Key Decisions Made
- Model: qwen/qwen3.6-plus:free on OpenRouter (default), 3 models available
- Unanswerable threshold: 0.10
- Few-shot split: 3 in prompt, 50 questions for eval
- drug_knowledge.csv used for enriched LLM context AND structured fallback

## Important Notes
- Run app: `python3 app.py`
- Run eval: `python3 eval_pipeline.py`
- Run tests: `python3 -m pytest tests/ -v`
- kg_full.png available for presentation architecture slide
