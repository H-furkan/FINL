# Active Context

## Current State
**Project complete and publicly released.** All development phases finished. Repo restructured for public consumption.

## What's Been Done
- [x] Phase 1: Data analysis & preprocessing (rag/data.py)
- [x] Phase 2: Hybrid 3-layer retrieval engine (rag/retrieval.py)
- [x] Phase 3: Query classifier + unanswerable detection
- [x] Phase 4: LLM generation + few-shot prompts + fallback (rag/generation.py)
- [x] Phase 5: Gradio UI (app.py)
- [x] Phase 6: Presentation website (docs/index.html on GitHub Pages)
- [x] Deployed to HuggingFace Spaces (permanent hosting)
- [x] API key rotation with 5 keys (stored as HF Space secrets)
- [x] Repo restructured: removed datathon boilerplate, new README, cleaned API keys
- [x] 50-question eval: **90% overall (47/50 passed)**

## Deployment
- **Live Demo**: https://huggingface.co/spaces/Fo-zh/drug-knowledge-assistant
- **Website**: https://h-furkan.github.io/FINL/
- **GitHub**: https://github.com/H-furkan/FINL
- **API Keys**: stored as HF Space secret `OPENROUTER_API_KEYS` (never in code)

## Current LLM Models (free tier, subject to change)
- google/gemma-3-12b-it:free (default)
- qwen/qwen3.6-plus:free
- nvidia/nemotron-3-super-120b-a12b:free
- openai/gpt-oss-120b:free

## Key Decisions
- API keys read from environment variables only (OPENROUTER_API_KEYS)
- Gradio not in GitHub dependencies (only on HF Spaces)
- memory-bank/ kept for AI continuity across sessions
- Free model availability changes frequently on OpenRouter -- check before updating

## Commands
- Run eval: `uv run python eval_pipeline.py`
- Run tests: `uv run pytest`
- Format: `uv run ruff format .`
- Lint: `uv run ruff check --fix`

## Future Development (see README TODO)
- Real medicine database (FDA OpenFDA API, DrugBank)
- Semantic embeddings (4th retrieval layer)
- Drug interaction detection
- Multi-language support
- Conversation memory for follow-up questions
