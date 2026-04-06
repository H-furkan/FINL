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
[Score Drop-off Filter] --> Cut when next score < 40% of previous
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
    data.py                  # Drug knowledge graph + indexes
    retrieval.py             # HybridRetriever + classify_query
    generation.py            # RAGPipeline: LLM + fallback
tests/                       # 54 tests across 4 files
app.py                       # Gradio UI (deployed to HF Spaces)
eval_pipeline.py             # 50-question eval suite
benchmark_prompts.py         # Prompt strategy comparison
eval_questions.md            # 50 eval questions with expected answers
drug_docs.csv                # Source dataset (100 docs, 50 drugs)
drug_knowledge.csv           # Structured knowledge CSV (50 drugs)
visualize_kg.py              # Knowledge graph visualization
generate_web_graphs.py       # Web-optimized graph images
docs/                        # GitHub Pages site
memory-bank/                 # AI context persistence
```

## Key Design Decisions

### Drug-Pair Structure Exploitation
The dataset has 50 drugs x 2 docs each. We build a structured index at startup
and use drug_knowledge.csv for enriched context and fallback answers.

### API Key Rotation
Multiple OpenRouter API keys stored as HF Space secrets. Each request cycles
to the next key. If rate-limited, retries with remaining keys before falling back.

### Few-Shot over DPO
3 gold Q&A pairs as few-shot examples in prompts. DPO impossible with API models.

### Unanswerable Detection
If no document scores above 0.10, the system refuses to answer rather than hallucinate.

### Structured Fallback
When LLM is unavailable, build_fallback_answer generates formatted answers from
drug_knowledge.csv columns. Achieves 74% without any LLM calls.
