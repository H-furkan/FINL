# Technology Context

## Stack
- **Language**: Python 3.10+
- **Package Manager**: uv (with uv.lock)
- **Data**: pandas (CSV loading, DataFrame operations)
- **Retrieval**: scikit-learn (TF-IDF, cosine similarity)
- **LLM**: OpenAI SDK pointed at OpenRouter (free tier)
- **Visualization**: matplotlib + networkx (knowledge graph)
- **Progress**: tqdm (eval progress bars)
- **UI**: Gradio (planned -- Phase 5)
- **Formatting**: ruff (line-length 88, double quotes, space indent)
- **Linting**: ruff check (E, W, F, I, UP, B rules)
- **Testing**: pytest (tests/ directory, 54 tests)
- **Pre-commit**: prek (ruff format + ruff check + pytest)
- **No GPU required**

## Dev Commands
```bash
uv sync                          # install all deps
python3 -m pytest tests/ -v      # run 54 tests
python3 eval_pipeline.py         # run 50-question eval with tqdm
python3 run_pipeline.py          # interactive pipeline
python3 visualize_kg.py          # regenerate knowledge graph image
```

## LLM Configuration
- **Provider**: OpenRouter (https://openrouter.ai/api/v1)
- **Model**: `qwen/qwen3.6-plus:free`
- **API Key**: Set OPENROUTER_API_KEY env var
- **Fallback**: Structured answers from drug_knowledge.csv when LLM unavailable

## Key Files
| File | Purpose |
|------|---------|
| `rag/data.py` | Drug knowledge graph + indexes + CSV export (Phase 1) |
| `rag/retrieval.py` | HybridRetriever + classify_query (Phase 2-3) |
| `rag/generation.py` | RAGPipeline: LLM + fallback (Phase 4) |
| `drug_docs.csv` | Source dataset (100 docs, 50 drugs) |
| `drug_knowledge.csv` | Structured CSV (50 drugs, pre-parsed columns) |
| `eval_pipeline.py` | 50-question eval with tqdm + category scoring |
| `eval_questions.md` | 50 questions across 8 categories |
| `visualize_kg.py` | Knowledge graph visualization (3 views) |
| `kg_full.png` | Rendered full knowledge graph (for presentation) |
| `tests/` | 54 tests across 4 test files |
| `run_pipeline.py` | Interactive pipeline script |
| `ROADMAP.md` | Hackathon strategy |

## AI Tooling
- **jCodeMunch MCP**: Token-efficient code retrieval (AST-based symbol indexing)
- **Memory Bank**: Persistent AI memory across sessions for team continuity
