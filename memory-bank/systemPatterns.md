# System Patterns & Architecture

## Architecture: 3-Layer Hybrid RAG

```
User Query
    |
    v
[Query Classifier] --> drug_specific | condition_lookup | comparison | multi_hop | safety | unanswerable
    |
    v
[Hybrid Retriever]
    |--- Layer 1: Exact Drug Match (regex, score=1.0)
    |--- Layer 2: TF-IDF + Cosine Similarity (score=0.0-1.0)
    |--- Layer 3: Condition/Keyword Match (reverse index, score=0.8)
    |
    v
[Sibling Doc Linker] --> Always pull both usage + side_effect docs for matched drugs
    |
    v
[LLM Generator] --> Dynamic prompt template per query type, few-shot examples
    |
    v
[Fallback Handler] --> Template answer from drug_index if LLM fails
    |
    v
Answer + Source Documents
```

## Project Structure
```
.
├── run_pipeline.py              # Main pipeline script
���── drug_docs.csv                # Dataset (100 docs, 50 drugs)
├── example_working_pipeline.ipynb  # Starter notebook
├── tests/
│   ├── __init__.py
│   └── test_retrieval.py        # Retrieval unit tests
├── pyproject.toml               # uv project config + ruff + pytest
├── prek.toml                    # Pre-commit hooks
├── uv.lock                      # Dependency lock
├── CLAUDE.md                    # AI agent instructions
├── ROADMAP.md                   # Hackathon strategy
├── .jcodemunch.jsonc            # Code indexing config
└── memory-bank/                 # Persistent AI context
```

## Key Design Decisions

### Drug-Pair Structure Exploitation
The dataset has 50 drugs x 2 docs each. We build a structured index at startup
to enable direct lookup instead of relying solely on vector similarity.

### Few-Shot over DPO
We use 3 gold Q&A pairs as few-shot examples in prompts (instant, free) rather
than DPO fine-tuning (impossible with API models, too few examples, no GPU).
Remaining 7 Q&A pairs are held out for evaluation.

### Query Classification Before Retrieval
Simple keyword rules route queries to the optimal retrieval strategy.
No ML needed -- regex + keyword matching is sufficient for this domain.

### Unanswerable Detection
If no document scores above threshold (0.15), the system refuses to answer
rather than hallucinating. Critical for evaluation criteria.

## Data Structures

### drug_index (built at startup)
```python
{
    "Drug A": {
        "usage_docs": [0],
        "side_effect_docs": [1],
        "all_docs": [0, 1],
        "conditions": ["hypertension"],
        "side_effects": ["dizziness", "headache", "fatigue"]
    }
}
```

### condition_index (reverse lookup)
```python
{
    "hypertension": ["Drug A", "Drug V"],
    "diabetes": ["Drug B"],
    "asthma": ["Drug F"]
}
```
