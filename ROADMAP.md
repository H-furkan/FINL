# ROADMAP: Winning the Drug Knowledge Assistant Datathon

## Key Insight Most Teams Will Miss

The dataset has **50 drugs** (A through AX), each with **exactly 2 documents**:
one for **usage/description** and one for **side effects**. This structure is the
key to dominating retrieval quality. While other teams fumble with generic TF-IDF,
we exploit this structure directly.

---

## Time Budget (3 Hours)

| Phase | Time | Focus |
|-------|------|-------|
| Phase 1 | 0:00 - 0:30 | Data analysis + smart preprocessing |
| Phase 2 | 0:30 - 1:15 | Hybrid retrieval engine |
| Phase 3 | 1:15 - 1:45 | Query understanding + drug-aware linking |
| Phase 4 | 1:45 - 2:15 | LLM integration + prompt engineering |
| Phase 5 | 2:15 - 2:40 | UI (Gradio/Streamlit) + evaluation |
| Phase 6 | 2:40 - 3:00 | Presentation prep + polish |

---

## Phase 1 -- Data Analysis & Smart Preprocessing (30 min)

### 1.1 Build a Drug Knowledge Graph

Parse every document and extract:
- **Drug name** (regex: `Drug [A-Z]+`)
- **Document type**: "usage" or "side_effect" (keyword detection)
- **Conditions treated** (e.g., hypertension, diabetes)
- **Side effects list** (e.g., dizziness, headache)

Store this as a structured lookup:
```python
drug_index = {
    "Drug A": {
        "usage_docs": [0],       # doc_ids for usage
        "side_effect_docs": [1], # doc_ids for side effects
        "conditions": ["hypertension"],
        "side_effects": ["dizziness", "headache", "fatigue"]
    },
    ...
}
```

**Why this wins**: When someone asks "What are the side effects of Drug A?",
other teams' TF-IDF might retrieve random docs. Ours directly fetches the
exact docs for Drug A.

### 1.2 Text Cleaning

- Lowercase all text for matching
- Preserve original text for display
- Build a condition-to-drug reverse index:
  ```python
  condition_index = {
      "hypertension": ["Drug A", "Drug V"],
      "diabetes": ["Drug B"],
      ...
  }
  ```

---

## Phase 2 -- Hybrid Retrieval Engine (45 min)

### 2.1 Three-Layer Retrieval (This Is the Killer Feature)

Instead of one retrieval method, use three and combine scores:

**Layer 1: Exact Drug Match (highest priority)**
- Extract drug names from the query via regex
- If found, directly pull that drug's usage + side effect docs
- Score: 1.0 (perfect match)

**Layer 2: TF-IDF + Cosine Similarity**
- Same as the starter notebook but with cleaned text
- Score: 0.0 - 1.0 (cosine similarity)

**Layer 3: Keyword/Condition Match**
- Check if the query mentions a known condition (diabetes, asthma, etc.)
- Pull all drugs linked to that condition from the reverse index
- Score: 0.8 (condition match)

### 2.2 Score Fusion

```python
def hybrid_retrieve(query, top_k=5):
    scores = {}

    # Layer 1: exact drug match
    drug_names = re.findall(r"Drug [A-Z]+", query)
    for drug in drug_names:
        for doc_id in drug_index[drug]["all_docs"]:
            scores[doc_id] = scores.get(doc_id, 0) + 1.0

    # Layer 2: TF-IDF
    tfidf_scores = get_tfidf_scores(query)
    for doc_id, score in tfidf_scores.items():
        scores[doc_id] = scores.get(doc_id, 0) + 0.5 * score

    # Layer 3: condition match
    for condition, drugs in condition_index.items():
        if condition in query.lower():
            for drug in drugs:
                for doc_id in drug_index[drug]["all_docs"]:
                    scores[doc_id] = scores.get(doc_id, 0) + 0.8

    # Return top-k by combined score
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return ranked[:top_k]
```

### 2.3 Drug-Aware Document Linking

Critical rule: **When any doc for a drug is retrieved, always pull its sibling doc too.**

If we retrieve Drug F's usage doc, automatically include Drug F's side effect
doc. This directly fixes the weakness in the starter notebook where it answered
"I don't know its side effects" for asthma.

---

## Phase 3 -- Query Understanding (30 min)

### 3.1 Query Classification

Classify each query into a type before retrieval:

| Query Type | Example | Strategy |
|------------|---------|----------|
| `drug_specific` | "Side effects of Drug A?" | Direct lookup |
| `condition_lookup` | "What treats diabetes?" | Reverse index |
| `comparison` | "Which drugs are for infections?" | Multi-drug retrieval |
| `multi_hop` | "Drug for asthma + side effects?" | Linked doc retrieval |
| `safety` | "Which drugs have dependency risks?" | Keyword scan across all docs |
| `unanswerable` | "Which drug cures cancer completely?" | Detect & refuse |

Classification can be done with simple keyword rules -- no ML needed:
```python
def classify_query(query):
    q = query.lower()
    drug_mentioned = bool(re.search(r"drug [a-z]+", q))
    asks_side_effects = any(w in q for w in ["side effect", "risk", "cause", "danger"])
    asks_usage = any(w in q for w in ["used for", "treat", "prescribed", "manage"])
    asks_comparison = any(w in q for w in ["which drugs", "what drugs", "compare"])

    if drug_mentioned and asks_side_effects and asks_usage:
        return "multi_hop"
    elif drug_mentioned:
        return "drug_specific"
    elif asks_comparison:
        return "comparison"
    elif asks_usage:
        return "condition_lookup"
    elif asks_side_effects:
        return "safety"
    else:
        return "general"
```

### 3.2 Unanswerable Detection

Before sending to LLM, check: if no retrieved document scores above a threshold
(e.g., 0.15), flag the query as likely unanswerable. This prevents hallucination
-- a major evaluation criterion.

---

## Phase 4 -- LLM Integration & Prompt Engineering (30 min)

### 4.1 Few-Shot Prompting with Example Q&A Pairs (Instead of DPO)

`example_queries.md` contains 10 gold Q&A pairs. We split them strategically:

- **2-3 pairs go INTO the prompt** as few-shot examples (style alignment)
- **7-8 pairs become the eval test suite** (Phase 5)

**Why not DPO?** DPO (Direct Preference Optimization) requires: (a) access to model
weights (we use API models -- can't fine-tune them), (b) hundreds of preference pairs
with both chosen AND rejected answers (we only have 10 correct answers), and (c) GPU
compute + training time we don't have. Few-shot prompting gives us ~80% of the style
alignment benefit for free, instantly.

### 4.2 Dynamic Prompt Templates

Use different prompts depending on query type, each with few-shot examples baked in:

**For drug-specific queries:**
```
You are a medical knowledge assistant. Answer using ONLY the provided documents.
Cite which document supports each fact. If the answer is not in the documents,
say "This information is not available in the database."

Example:
Q: What are the side effects of Drug A?
A: Drug A may cause: dizziness, headache, and fatigue.

Q: Which drug is used for diabetes?
A: Drug B is used for type 2 diabetes. It helps lower blood sugar by increasing insulin sensitivity.

Now answer the following:

Documents:
{context}

Question: {query}

Answer in structured format with bullet points.
```

**For comparison queries:**
```
You are a medical knowledge assistant. Based on the documents below, list ALL
drugs that match the question. For each drug, provide a one-line summary.

Example:
Q: Which drugs are used for infections?
A:
- Drug C: bacterial infections (pneumonia, bronchitis)
- Drug AQ: respiratory infections
- Drug AH: eye infections

Now answer the following:

Documents:
{context}

Question: {query}

Format:
- Drug X: [brief explanation]
- Drug Y: [brief explanation]
```

### 4.3 Context Window Optimization

- For single-drug queries: send 2 docs (usage + side effects)
- For comparison queries: send up to 6 docs (3 drugs x 2 docs)
- For safety queries: send all matching docs (filtered by keyword)

### 4.4 Fallback System

If the LLM API fails:
1. Return retrieved documents formatted nicely
2. Use the structured drug_index to generate a template answer:
   ```
   "Drug A is used for hypertension. Known side effects: dizziness, headache, fatigue."
   ```
   This alone beats most teams' raw document dump.

### 4.5 Model Choice

Use `qwen/qwen3.6-plus:free` or `mistralai/mistral-7b-instruct` on OpenRouter.
Both are free. Test both during development and pick whichever gives cleaner
structured output.

---

## Phase 5 -- UI & Evaluation (25 min)

### 5.1 Gradio App (Quick Win for Presentation Points)

```python
import gradio as gr

def answer_question(query):
    query_type = classify_query(query)
    retrieved = hybrid_retrieve(query)
    answer = generate_answer(query, retrieved, query_type)
    sources = format_sources(retrieved)
    return answer, sources, query_type

demo = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(label="Ask about any drug..."),
    outputs=[
        gr.Textbox(label="Answer"),
        gr.Textbox(label="Source Documents"),
        gr.Textbox(label="Query Type Detected")
    ],
    title="Drug Knowledge Assistant",
    examples=[
        "What are the side effects of Drug A?",
        "Which drug is used for diabetes?",
        "Which drugs have dependency risks?",
        "Which drug cures cancer completely?"
    ]
)
demo.launch()
```

**Why Gradio**: Installs in 10 seconds (`pip install gradio`), looks professional,
provides a sharable link, has example buttons. Judges love a polished UI.

### 5.2 Automated Evaluation (Using the Held-Out Q&A Pairs)

From the 10 example queries in `example_queries.md`:
- **3 are used as few-shot examples in prompts** (Q1: side effects of Drug A,
  Q2: diabetes drug, Q5: drugs for infections)
- **7 are held out for evaluation** (Q3, Q4, Q6, Q7, Q8, Q9, Q10)

This split is important -- evaluating on the same examples you put in the prompt
would be cheating. Held-out eval shows the system genuinely generalizes.

```python
# These 7 queries were NOT shown to the LLM as few-shot examples
eval_cases = [
    {"query": "What treats hypertension?",
     "expected_keywords": ["Drug A", "blood vessels"]},
    {"query": "What are the side effects of Drug M?",
     "expected_keywords": ["fatigue", "hair loss", "nausea"]},
    {"query": "Which drug is used for asthma and what are its side effects?",
     "expected_keywords": ["Drug F", "tremors", "nervousness", "heart rate"]},
    {"query": "Which drugs have dependency risks?",
     "expected_keywords": ["Drug D", "Drug J", "dependency"]},
    {"query": "Which drugs affect the brain?",
     "expected_keywords": ["Drug E", "Drug D", "Drug O"]},
    {"query": "Which drug cures cancer completely?",
     "expected_keywords": ["don't know", "not available", "not in"]},
    {"query": "Which drug is used for cholesterol and what risks does it have?",
     "expected_keywords": ["Drug G", "muscle pain", "liver"]},
]

print("=" * 60)
print("EVALUATION RESULTS (7 held-out queries)")
print("=" * 60)

total = 0
for tc in eval_cases:
    answer = pipeline(tc["query"])
    hits = [kw for kw in tc["expected_keywords"] if kw.lower() in answer.lower()]
    score = len(hits) / len(tc["expected_keywords"])
    total += score
    print(f"\nQ: {tc['query']}")
    print(f"Score: {score:.0%} ({len(hits)}/{len(tc['expected_keywords'])} keywords)")
    print(f"Hits: {hits}")

print(f"\nOVERALL: {total/len(eval_cases):.0%}")
```

---

## Phase 6 -- Presentation Strategy (20 min)

### The 5-Minute Script

**Minute 1: Problem & Approach (show architecture diagram)**
- "We built a 3-layer hybrid retrieval system with query understanding"
- Show a simple flow diagram: Query -> Classify -> Retrieve -> Generate -> Answer

**Minute 2: The Killer Feature -- Drug-Aware Retrieval**
- "Other systems treat docs independently. We discovered the dataset has
  a drug-pair structure and exploit it."
- Show how vanilla TF-IDF misses side effects for the asthma query
- Show how our system nails it by linking sibling docs

**Minute 3: Live Demo (3 queries)**
1. Simple: "What are the side effects of Drug A?" (shows exact retrieval)
2. Reverse: "What treats hypertension?" (shows condition index)
3. Hard: "Which drugs have dependency risks?" (shows multi-doc comparison)

**Minute 4: Evaluation Results**
- Show the automated scoring table
- Highlight the unanswerable query: "Which drug cures cancer completely?"
  -> Our system correctly says "not in the database" (anti-hallucination)

**Minute 5: Improvements & Wrap-up**
- "With more time: semantic embeddings (sentence-transformers), caching,
  user feedback loop, drug interaction detection"

### Presentation Tips
- Have a backup screenshot of every demo query in case the API goes down
- Show the architecture diagram FIRST -- judges remember visual structure
- Use the word "hybrid" -- it signals sophistication
- Mention "hallucination prevention" -- shows awareness of real-world AI issues

---

## What Makes This Beat Other Teams

| What others will do | What we do instead |
|--------------------|--------------------|
| Plain TF-IDF | 3-layer hybrid retrieval with score fusion |
| Treat all 100 docs independently | Exploit the drug-pair structure with linked retrieval |
| One generic prompt | Query-type-aware dynamic prompts |
| Jupyter notebook demo | Polished Gradio UI with example buttons |
| Manual testing | Automated evaluation with scoring table |
| No fallback | Template-based fallback that works without LLM |
| Hope the LLM doesn't hallucinate | Explicit unanswerable detection + threshold gating |
| No style control over LLM output | Few-shot prompting with gold Q&A pairs (practical alternative to DPO) |
| Evaluate on same examples in prompt | Proper train/eval split: 3 few-shot, 7 held-out |

---

## Quick Dependency Install

```bash
uv sync                          # install all deps from pyproject.toml
pip install gradio               # for UI (add to pyproject.toml if keeping)
```

---

## Emergency Shortcuts

If running behind schedule, cut in this order:
1. Drop Gradio UI -> demo in notebook (saves 10 min)
2. Drop automated evaluation -> test manually (saves 10 min)
3. Drop query classification -> use hybrid retrieval for all queries (saves 15 min)

**Never cut**: the drug-aware linking and hybrid retrieval. That's the core differentiator.
