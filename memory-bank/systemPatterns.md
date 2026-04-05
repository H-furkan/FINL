# System Patterns & Architecture

## Architecture: 3-Layer Hybrid RAG

```
User Query
    |
    v
[Query Classifier] --> drug_specific | condition_lookup | comparison | multi_hop | safety | general
    |
    v
[Hybrid Retriever]
    |--- Layer 1: Exact Drug Match (regex, score=1.0)
    |--- Layer 2: TF-IDF + Cosine Similarity (weight=0.5)
    |--- Layer 3: Condition/Side-Effect Keyword Match (0.8/0.6)
    |
    v
[Sibling Doc Linker] --> Always pull both usage + side_effect docs (score=0.3)
    |
    v
[Enriched Context Builder] --> Raw docs + structured summaries from drug_knowledge.csv
    |
    v
[RAGPipeline.answer()]
    |--- LLM mode: query-type-aware prompt + few-shot examples -> OpenRouter
    |--- Fallback mode: structured answer from drug_knowledge.csv (74% accuracy)
    |--- Refused mode: unanswerable detection (max_score < 0.10)
    |
    v
{answer, sources, query_type, mode, max_score}
```

## Project Structure
```
rag/
    __init__.py
    data.py                  # Drug knowledge graph + indexes (Phase 1)
    retrieval.py             # HybridRetriever + classify_query (Phase 2-3)
    generation.py            # RAGPipeline: LLM + fallback (Phase 4)
tests/
    __init__.py
    test_retrieval.py        # Basic retrieval tests (3)
    test_data.py             # Data/index tests (19)
    test_retrieval_hybrid.py # Hybrid retriever + KG CSV tests (24)
    test_generation.py       # RAGPipeline tests (8)
run_pipeline.py              # Interactive pipeline script
eval_pipeline.py             # 50-question eval with tqdm + category scoring
eval_questions.md            # 50 questions across 8 categories
drug_docs.csv                # Source dataset (100 docs, 50 drugs)
drug_knowledge.csv           # Structured CSV (50 drugs, pre-parsed)
visualize_kg.py              # Knowledge graph visualization
kg_full.png                  # Rendered full knowledge graph
pyproject.toml               # uv project config
CLAUDE.md                    # AI agent instructions
ROADMAP.md                   # Hackathon strategy
memory-bank/                 # Persistent AI context
```

## Key Design Decisions

### Drug-Pair Structure Exploitation
The dataset has 50 drugs x 2 docs each. We build a structured index at startup
and use drug_knowledge.csv for enriched context and fallback answers.

### Few-Shot over DPO
3 gold Q&A pairs as few-shot examples in prompts. Remaining questions held out
for evaluation. DPO impossible with API models.

### Query Classification Before Retrieval
Simple keyword rules route queries to optimal retrieval + prompt strategy.

### Unanswerable Detection
If no document scores above 0.10, the system refuses to answer.

### Enriched Context
LLM receives both raw retrieved docs AND structured summaries from drug_knowledge.csv,
giving it cleaner deduplicated facts to work with.

### Structured Fallback
When LLM is unavailable, build_fallback_answer generates formatted answers from
drug_knowledge.csv columns. Achieves 74% without any LLM calls.

## Data Structures (rag/data.py)

### drug_index
```python
{"Drug A": {"usage_docs": [0], "side_effect_docs": [1], "all_docs": [0, 1],
            "conditions": ["hypertension"], "side_effects": ["dizziness", "headache", "fatigue"],
            "usage_text": "...", "side_effect_text": "..."}}
```

### condition_index
```python
{"hypertension": ["Drug A", "Drug V"], "diabetes": ["Drug B"]}
```

### side_effect_index
```python
{"dizziness": ["Drug A", "Drug H", "Drug V", ...], "dependency risks": ["Drug D", "Drug J"]}
```
