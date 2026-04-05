"""Batch evaluation of the RAG pipeline against example_queries.md gold answers."""

from __future__ import annotations

import os

from openai import OpenAI

from rag.retrieval import HybridRetriever, classify_query

API_KEY = os.environ.get(
    "OPENROUTER_API_KEY",
    "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",
)
MODEL_NAME = "qwen/qwen3.6-plus:free"

retriever = HybridRetriever()
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=API_KEY)

# ---------------------------------------------------------------------------
# Few-shot examples (3 used in prompts, NOT in eval set)
# ---------------------------------------------------------------------------

FEW_SHOT_EXAMPLES = """Example:
Q: What are the side effects of Drug A?
A: Drug A may cause: dizziness, headache, and fatigue.

Q: Which drug is used for diabetes?
A: Drug B is used for type 2 diabetes. It helps lower blood sugar by increasing insulin sensitivity.

Q: Which drugs are used for infections?
A:
- Drug C: bacterial infections (pneumonia, bronchitis)
- Drug AQ: respiratory infections
- Drug AH: eye infections"""

# ---------------------------------------------------------------------------
# Prompt templates per query type
# ---------------------------------------------------------------------------

PROMPT_TEMPLATES = {
    "drug_specific": """You are a medical knowledge assistant. Answer using ONLY the provided documents.
If the answer is not in the documents, say "I don't know based on the provided data."

{few_shot}

Now answer the following:

Documents:
{context}

Question: {query}

Answer in structured format with bullet points.""",
    "condition_lookup": """You are a medical knowledge assistant. Answer using ONLY the provided documents.
Name the drug(s) and briefly explain how they work.
If the answer is not in the documents, say "I don't know based on the provided data."

{few_shot}

Now answer the following:

Documents:
{context}

Question: {query}""",
    "comparison": """You are a medical knowledge assistant. Based on the documents below, list ALL
drugs that match the question. For each drug, provide a one-line summary.
If the answer is not in the documents, say "I don't know based on the provided data."

{few_shot}

Now answer the following:

Documents:
{context}

Question: {query}

Format:
- Drug X: [brief explanation]""",
    "multi_hop": """You are a medical knowledge assistant. Answer using ONLY the provided documents.
The question asks about both usage AND side effects -- address both parts.
If the answer is not in the documents, say "I don't know based on the provided data."

{few_shot}

Now answer the following:

Documents:
{context}

Question: {query}

Answer in structured format with bullet points.""",
    "safety": """You are a medical knowledge assistant. Based on the documents below, identify ALL
drugs relevant to the safety concern in the question.
If the answer is not in the documents, say "I don't know based on the provided data."

{few_shot}

Now answer the following:

Documents:
{context}

Question: {query}

Format:
- Drug X: [relevant safety info]""",
}


def generate_answer(query: str, retrieved_docs, query_type: str) -> str:
    """Generate an answer using the LLM with query-type-aware prompts."""
    # Unanswerable detection
    if not retrieved_docs.attrs.get("is_answerable", True):
        return "I don't know based on the provided data."

    # Use enriched context (raw docs + structured drug_knowledge.csv data)
    context = retriever.build_enriched_context(retrieved_docs)

    template = PROMPT_TEMPLATES.get(query_type, PROMPT_TEMPLATES["drug_specific"])
    prompt = template.format(
        few_shot=FEW_SHOT_EXAMPLES,
        context=context,
        query=query,
    )

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return response.choices[0].message.content
    except Exception as e:
        # Fallback: structured answer from drug_knowledge.csv
        fallback = retriever.build_fallback_answer(retrieved_docs)
        return f"[LLM unavailable] {fallback}"


# ---------------------------------------------------------------------------
# Eval cases (7 held-out -- NOT used as few-shot examples)
# ---------------------------------------------------------------------------

EVAL_CASES = [
    {
        "id": 3,
        "query": "What treats hypertension?",
        "expected_keywords": ["Drug A", "blood vessels"],
        "should_refuse": False,
    },
    {
        "id": 4,
        "query": "What are the side effects of Drug M?",
        "expected_keywords": ["fatigue", "hair loss", "nausea"],
        "should_refuse": False,
    },
    {
        "id": 6,
        "query": "Which drug is used for asthma and what are its side effects?",
        "expected_keywords": ["Drug F", "tremors", "nervousness", "heart rate"],
        "should_refuse": False,
    },
    {
        "id": 7,
        "query": "Which drugs have dependency risks?",
        "expected_keywords": ["Drug D", "Drug J", "dependency"],
        "should_refuse": False,
    },
    {
        "id": 8,
        "query": "Which drugs affect the brain?",
        "expected_keywords": ["Drug E", "Drug D", "Drug O"],
        "should_refuse": False,
    },
    {
        "id": 9,
        "query": "Which drug cures cancer completely?",
        "expected_keywords": [],
        "should_refuse": True,
    },
    {
        "id": 10,
        "query": "Which drug is used for cholesterol and what risks does it have?",
        "expected_keywords": ["Drug G", "muscle pain", "liver"],
        "should_refuse": False,
    },
]

REFUSAL_PATTERNS = [
    "don't know",
    "do not know",
    "not available",
    "not in the",
    "no information",
    "cannot determine",
    "no drug",
    "does not",
    "no evidence",
    "not mentioned",
    "not provided",
    "cannot be determined",
    "no data",
    "completely cure",  # restating impossibility
]


def run_eval():
    print("=" * 70)
    print("EVALUATION: 7 Held-Out Queries (3 used as few-shot, not evaluated)")
    print(f"Retriever: HybridRetriever (3-layer + sibling linking)")
    print(f"Model: {MODEL_NAME}")
    print("=" * 70)

    total_score = 0
    results_summary = []

    for tc in EVAL_CASES:
        query = tc["query"]
        query_type = classify_query(query)
        should_refuse = tc["should_refuse"]

        print(f"\n{'─' * 70}")
        print(f"Q{tc['id']}: {query}")
        print(f"  Type: {query_type}")

        retrieved = retriever.retrieve(query)
        print(
            f"  Retrieved {len(retrieved)} docs, "
            f"top score: {retrieved.attrs.get('max_score', 0):.3f}, "
            f"answerable: {retrieved.attrs.get('is_answerable', '?')}"
        )

        answer = generate_answer(query, retrieved, query_type)
        print(f"  Answer: {answer[:250]}{'...' if len(answer) > 250 else ''}")

        if should_refuse:
            # For unanswerable queries: check if the LLM correctly refused
            answer_lower = answer.lower()
            did_refuse = any(p in answer_lower for p in REFUSAL_PATTERNS)
            score = 1.0 if did_refuse else 0.0
            status = "PASS" if did_refuse else "FAIL"
            print(f"  Expected: REFUSAL (unanswerable query)")
            print(f"  Refused: {'Yes' if did_refuse else 'No'}")
            print(f"  Score: {score:.0%}")
        else:
            # For answerable queries: keyword matching
            hits = [
                kw for kw in tc["expected_keywords"] if kw.lower() in answer.lower()
            ]
            misses = [
                kw
                for kw in tc["expected_keywords"]
                if kw.lower() not in answer.lower()
            ]
            score = len(hits) / len(tc["expected_keywords"])
            status = "PASS" if score >= 0.5 else "FAIL"
            print(f"  Score: {score:.0%} ({len(hits)}/{len(tc['expected_keywords'])})")
            print(f"  Hits: {hits}")
            if misses:
                print(f"  Misses: {misses}")

        total_score += score
        icon = "+" if status == "PASS" else "x"
        print(f"  Result: [{icon}] {status}")

        results_summary.append(
            {"query": query, "score": score, "status": status, "type": query_type}
        )

    print(f"\n{'=' * 70}")
    avg = total_score / len(EVAL_CASES)
    print(f"OVERALL SCORE: {avg:.0%} ({total_score:.1f}/{len(EVAL_CASES)})")
    print(f"{'=' * 70}")

    passed = sum(1 for r in results_summary if r["status"] == "PASS")
    print(f"Passed: {passed}/{len(EVAL_CASES)}")

    return results_summary


if __name__ == "__main__":
    run_eval()
