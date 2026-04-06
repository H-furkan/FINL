# Technology Context

## Stack
- **Language**: Python 3.10+
- **Package Manager**: uv (with uv.lock)
- **Data**: pandas (CSV loading, DataFrame operations)
- **Retrieval**: scikit-learn (TF-IDF, cosine similarity)
- **LLM**: OpenAI SDK pointed at OpenRouter (free tier)
- **UI**: Gradio (on HF Spaces only, not in GitHub deps)
- **Visualization**: matplotlib + networkx (knowledge graph)
- **Progress**: tqdm (eval progress bars)
- **Formatting**: ruff (line-length 88, double quotes, space indent)
- **Linting**: ruff check (E, W, F, I, UP, B rules)
- **Testing**: pytest (tests/ directory, 54 tests)
- **Hosting**: HuggingFace Spaces (Gradio app), GitHub Pages (website)
- **No GPU required**

## Dev Commands
```bash
uv sync                          # install all deps
uv run pytest                    # run 54 tests
uv run python eval_pipeline.py   # run 50-question eval
uv run ruff format .             # format code
uv run ruff check --fix          # lint + autofix
```

## LLM Configuration
- **Provider**: OpenRouter (https://openrouter.ai/api/v1)
- **API Keys**: env var `OPENROUTER_API_KEYS` (comma-separated, rotated per request)
- **Current models** (free tier, may change):
  - google/gemma-3-12b-it:free (default)
  - qwen/qwen3.6-plus:free
  - nvidia/nemotron-3-super-120b-a12b:free
  - openai/gpt-oss-120b:free
- **Fallback**: Structured answers from drug_knowledge.csv when LLM unavailable

## Key Files
| File | Purpose |
|------|---------|
| `rag/data.py` | Drug knowledge graph + indexes + CSV export |
| `rag/retrieval.py` | HybridRetriever + classify_query |
| `rag/generation.py` | RAGPipeline: LLM + fallback |
| `app.py` | Gradio UI (deployed to HF Spaces) |
| `drug_docs.csv` | Source dataset (100 docs, 50 drugs) |
| `drug_knowledge.csv` | Structured knowledge CSV (50 drugs) |
| `eval_pipeline.py` | 50-question eval with scoring |
| `benchmark_prompts.py` | Prompt strategy comparison |
| `eval_questions.md` | All 50 eval questions with expected answers |
| `visualize_kg.py` | Knowledge graph visualization |
| `generate_web_graphs.py` | Web-optimized graph images for docs/ |
| `docs/index.html` | GitHub Pages website |
| `tests/` | 54 tests across 4 files |
| `memory-bank/` | AI context persistence |
