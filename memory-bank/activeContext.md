# Active Context

## Current Phase
**Phase 4** -- LLM integration + few-shot prompts. Phases 1-3 complete.

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
  - Drug name extraction (regex), doc type classification, condition mapping, side effect extraction
  - All 50 drugs parsed with usage + side_effect docs, conditions, and side effects
  - tests/test_data.py: 14 tests covering all data structures
  - eval_pipeline.py: upgraded to HybridRetriever, 7/7 pass, 95% score
- [x] **Phase 1 extras** (by teammate):
  - drug_knowledge.csv: structured CSV export of all 50 drugs
  - visualize_kg.py: knowledge graph visualization (networkx + matplotlib)
  - kg_full.png: rendered full KG image (great for presentation slide 1!)
  - rag/data.py: added export_knowledge_csv(), load_knowledge_csv()

## What's Next
- [x] Phase 2: Hybrid 3-layer retrieval engine (rag/retrieval.py - HybridRetriever)
- [x] Phase 3: Query classifier (rag/retrieval.py - classify_query)
- [ ] Phase 4: LLM integration with few-shot prompts
- [ ] Phase 5: Gradio UI + automated evaluation
- [ ] Phase 6: Presentation prep

## Key Decisions Pending
- Final model choice: qwen vs mistral (test both during Phase 4)
- Similarity threshold for unanswerable detection (start with 0.15, tune)

## Important Notes
- API keys must NOT be hardcoded -- use environment variables
- Each team member should use their own OpenRouter API key
- Run `uv sync` to install dependencies before starting
- eval_pipeline.py upgraded -- uses HybridRetriever + few-shot prompts, 7/7 pass
- kg_full.png available for presentation architecture slide
