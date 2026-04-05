# Technology Context

## Stack
- **Language**: Python 3.10+
- **Package Manager**: uv (with uv.lock)
- **Data**: pandas (CSV loading, DataFrame operations)
- **Retrieval**: scikit-learn (TF-IDF, cosine similarity)
- **LLM**: OpenAI SDK pointed at OpenRouter (free tier)
- **UI**: Gradio (lightweight, sharable link)
- **Formatting**: ruff (line-length 88, double quotes, space indent)
- **Linting**: ruff check (E, W, F, I, UP, B rules)
- **Testing**: pytest (tests/ directory)
- **Pre-commit**: prek (ruff format + ruff check + pytest)
- **No GPU required**

## Dependencies (pyproject.toml)
```
pandas>=2.3.3
scikit-learn>=1.7.2
openai>=2.30.0
ipykernel>=7.2.0
```

Dev dependencies:
```
ruff>=0.15.9
pytest>=9.0.2
prek>=0.3.8
```

## Dev Commands
```bash
uv sync                          # install all deps
uv run python run_pipeline.py    # run the pipeline
uv run pytest                    # run tests
uv run ruff format .             # format code
uv run ruff check --fix          # lint + autofix
```

## LLM Configuration
- **Provider**: OpenRouter (https://openrouter.ai/api/v1)
- **Model**: `qwen/qwen3.6-plus:free` or `mistralai/mistral-7b-instruct`
- **API Key**: Each team member needs their own from openrouter.ai
- **Fallback**: Template-based answers from drug_index if API fails

## Key Files
| File | Purpose |
|------|---------|
| `drug_docs.csv` | Source dataset (100 docs, 50 drugs) |
| `run_pipeline.py` | Standalone RAG pipeline script |
| `example_working_pipeline.ipynb` | Starter notebook (baseline TF-IDF RAG) |
| `tests/test_retrieval.py` | Retrieval unit tests |
| `pyproject.toml` | Project config (deps, ruff, pytest) |
| `prek.toml` | Pre-commit hooks (format, lint, test) |
| `rag/data.py` | Drug knowledge graph + indexes (Phase 1) |
| `eval_pipeline.py` | Batch eval against 10 gold queries |
| `tests/test_data.py` | 14 tests for data structures |
| `ROADMAP.md` | Our strategy and implementation plan |

## AI Tooling
- **jCodeMunch MCP**: Token-efficient code retrieval (AST-based symbol indexing)
- **cursor-memory-bank**: Persistent AI memory across sessions for team continuity
