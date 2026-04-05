# CLAUDE.md -- AI Agent Instructions for Drug Knowledge Assistant

## Project Overview
RAG system for EPAM datathon. 50 drugs, 100 docs, 3-hour build.
See `ROADMAP.md` for full strategy and `memory-bank/` for persistent context.

## AI Tool Usage

### jCodeMunch MCP (Token-Efficient Code Retrieval)
Use jcodemunch-mcp for code lookup whenever available. Prefer symbol search,
outlines, and targeted retrieval over reading full files. This saves ~95% tokens
on code exploration tasks.

Install: `pip install jcodemunch-mcp`
Init: `jcodemunch-mcp init`

### Cursor Memory Bank (Cross-Session Continuity)
Before starting work, ALWAYS read `memory-bank/activeContext.md` and
`memory-bank/progress.md` to understand current state. After completing
any significant work, update these files.

## Code Conventions
- Python 3.10+, managed via uv
- Formatting: ruff (line-length 88, double quotes)
- Linting: ruff check (E, W, F, I, UP, B rules)
- Testing: pytest (tests/ directory)
- Pre-commit: prek (ruff format + ruff check + pytest)
- API keys go in environment variables, NEVER hardcoded
- All retrieval functions should return a DataFrame with a `score` column
- Fallback answers must work without LLM

## Dev Commands
```bash
uv sync                  # install dependencies
uv run python run_pipeline.py  # run the pipeline
uv run pytest            # run tests
uv run ruff format .     # format code
uv run ruff check --fix  # lint + autofix
```

## Key Architecture Decisions
- Hybrid 3-layer retrieval (drug match + TF-IDF + condition match)
- Few-shot prompting (NOT DPO -- we use API models)
- Drug-pair sibling linking (always pull both docs for a drug)
- Query classification before retrieval
- Unanswerable detection via score threshold
