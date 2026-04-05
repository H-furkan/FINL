# Active Context

## Current Phase
**Pre-implementation** -- Roadmap finalized, tooling set up, ready to begin Phase 1.

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

## What's Next
- [ ] Phase 1: Build drug_index and condition_index from drug_docs.csv
- [ ] Phase 2: Implement hybrid 3-layer retrieval engine
- [ ] Phase 3: Add query classifier
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
