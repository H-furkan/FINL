"""Batch evaluation of the RAG pipeline against eval_questions.md (50 questions)."""

from __future__ import annotations

import os

from openai import OpenAI
from tqdm import tqdm

from rag.retrieval import HybridRetriever, classify_query

API_KEY = os.environ.get(
    "OPENROUTER_API_KEY",
    "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",
)
MODEL_NAME = "qwen/qwen3.6-plus:free"

retriever = HybridRetriever()
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=API_KEY)

# ---------------------------------------------------------------------------
# Few-shot examples (3 from example_queries.md, NOT in eval set)
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
    if not retrieved_docs.attrs.get("is_answerable", True):
        return "I don't know based on the provided data."

    context = retriever.build_enriched_context(retrieved_docs)
    template = PROMPT_TEMPLATES.get(query_type, PROMPT_TEMPLATES["drug_specific"])
    prompt = template.format(
        few_shot=FEW_SHOT_EXAMPLES, context=context, query=query
    )

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return response.choices[0].message.content
    except Exception:
        fallback = retriever.build_fallback_answer(retrieved_docs)
        return f"[LLM unavailable] {fallback}"


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
]

# ---------------------------------------------------------------------------
# All 50 eval cases from eval_questions.md
# ---------------------------------------------------------------------------

EVAL_CASES = [
    # Category 1: Direct Fact Retrieval (Side Effects)
    {"id": 1, "cat": "direct_fact", "query": "What are the side effects of Drug E?",
     "expected_keywords": ["insomnia", "dry mouth", "changes in appetite"], "should_refuse": False},
    {"id": 2, "cat": "direct_fact", "query": "What side effects does Drug O cause?",
     "expected_keywords": ["dizziness", "fatigue", "coordination problems"], "should_refuse": False},
    {"id": 3, "cat": "direct_fact", "query": "What are the side effects of Drug AE?",
     "expected_keywords": ["weight gain", "tremors", "fatigue"], "should_refuse": False},
    {"id": 4, "cat": "direct_fact", "query": "Does Drug Z have any side effects?",
     "expected_keywords": ["blurred vision", "eye irritation", "dryness"], "should_refuse": False},
    {"id": 5, "cat": "direct_fact", "query": "What adverse reactions can Drug K cause?",
     "expected_keywords": ["stomach pain", "bleeding"], "should_refuse": False},
    {"id": 6, "cat": "direct_fact", "query": "What are the side effects of Drug AD?",
     "expected_keywords": ["insomnia", "heart rate", "anxiety"], "should_refuse": False},
    {"id": 7, "cat": "direct_fact", "query": "Tell me the side effects of Drug AN.",
     "expected_keywords": ["bleeding", "bruising", "fatigue"], "should_refuse": False},

    # Category 2: Usage-Based Questions
    {"id": 8, "cat": "usage", "query": "Which drug is used to treat epilepsy?",
     "expected_keywords": ["Drug O"], "should_refuse": False},
    {"id": 9, "cat": "usage", "query": "What drug should be prescribed for gout?",
     "expected_keywords": ["Drug AF"], "should_refuse": False},
    {"id": 10, "cat": "usage", "query": "Which drug is used for Parkinson disease?",
     "expected_keywords": ["Drug Y"], "should_refuse": False},
    {"id": 11, "cat": "usage", "query": "What drug treats chronic kidney disease?",
     "expected_keywords": ["Drug X"], "should_refuse": False},
    {"id": 12, "cat": "usage", "query": "Which drug is prescribed for ADHD?",
     "expected_keywords": ["Drug W"], "should_refuse": False},
    {"id": 13, "cat": "usage", "query": "What drug is used for glaucoma?",
     "expected_keywords": ["Drug Z"], "should_refuse": False},
    {"id": 14, "cat": "usage", "query": "Which drug treats multiple sclerosis?",
     "expected_keywords": ["Drug AM"], "should_refuse": False},

    # Category 3: Reverse Lookup (Condition -> Drug)
    {"id": 15, "cat": "reverse_lookup", "query": "What treats bipolar disorder?",
     "expected_keywords": ["Drug AE"], "should_refuse": False},
    {"id": 16, "cat": "reverse_lookup", "query": "What can I take for acid reflux?",
     "expected_keywords": ["Drug T"], "should_refuse": False},
    {"id": 17, "cat": "reverse_lookup", "query": "Is there a drug for osteoporosis?",
     "expected_keywords": ["Drug S"], "should_refuse": False},
    {"id": 18, "cat": "reverse_lookup", "query": "What medication is available for insomnia?",
     "expected_keywords": ["Drug L"], "should_refuse": False},
    {"id": 19, "cat": "reverse_lookup", "query": "What treats anemia?",
     "expected_keywords": ["Drug AA"], "should_refuse": False},
    {"id": 20, "cat": "reverse_lookup", "query": "What drug can help with obesity?",
     "expected_keywords": ["Drug AD"], "should_refuse": False},

    # Category 4: Comparison / Multi-Drug
    {"id": 21, "cat": "comparison", "query": "Which drugs cause nausea as a side effect?",
     "expected_keywords": ["Drug B", "Drug H", "Drug I", "Drug M"], "should_refuse": False},
    {"id": 22, "cat": "comparison", "query": "Which drugs are used to treat anxiety?",
     "expected_keywords": ["Drug J", "Drug AT"], "should_refuse": False},
    {"id": 23, "cat": "comparison", "query": "Which drugs cause dizziness?",
     "expected_keywords": ["Drug A", "Drug H", "Drug O"], "should_refuse": False},
    {"id": 24, "cat": "comparison", "query": "Which drugs treat skin conditions?",
     "expected_keywords": ["Drug P", "Drug AC", "Drug AR"], "should_refuse": False},
    {"id": 25, "cat": "comparison", "query": "Which drugs cause fatigue as a side effect?",
     "expected_keywords": ["Drug A", "Drug I", "Drug M", "Drug O"], "should_refuse": False},
    {"id": 26, "cat": "comparison", "query": "What drugs treat hormonal or thyroid problems?",
     "expected_keywords": ["Drug R", "Drug AJ"], "should_refuse": False},
    {"id": 27, "cat": "comparison", "query": "Which drugs are used for migraines?",
     "expected_keywords": ["Drug H", "Drug AP"], "should_refuse": False},

    # Category 5: Complex / Combined
    {"id": 28, "cat": "complex", "query": "Which drug is used for depression and what are its side effects?",
     "expected_keywords": ["Drug E", "insomnia", "dry mouth"], "should_refuse": False},
    {"id": 29, "cat": "complex", "query": "What treats arthritis and what risks does it have?",
     "expected_keywords": ["Drug K", "stomach pain", "bleeding"], "should_refuse": False},
    {"id": 30, "cat": "complex", "query": "Which drug is prescribed for heart failure and what are its side effects?",
     "expected_keywords": ["Drug Q", "dizziness", "low blood pressure", "fatigue"], "should_refuse": False},
    {"id": 31, "cat": "complex", "query": "What drug treats COPD and what are the side effects?",
     "expected_keywords": ["Drug AG", "cough", "throat irritation", "dizziness"], "should_refuse": False},
    {"id": 32, "cat": "complex", "query": "Which drug treats fungal infections and how does it work?",
     "expected_keywords": ["Drug U", "fungal cell membranes"], "should_refuse": False},
    {"id": 33, "cat": "complex", "query": "What drug is used for cancer therapy and what side effects should I expect?",
     "expected_keywords": ["Drug M", "fatigue", "hair loss", "nausea"], "should_refuse": False},
    {"id": 34, "cat": "complex", "query": "Which drug treats ulcers and what are its side effects?",
     "expected_keywords": ["Drug AK", "constipation", "nausea", "headache"], "should_refuse": False},

    # Category 6: Safety / Risk
    {"id": 35, "cat": "safety", "query": "Which drugs can cause liver problems?",
     "expected_keywords": ["Drug G", "Drug AB"], "should_refuse": False},
    {"id": 36, "cat": "safety", "query": "Are there any drugs that increase infection risk?",
     "expected_keywords": ["Drug AB", "Drug AX"], "should_refuse": False},
    {"id": 37, "cat": "safety", "query": "Which drugs can cause weight gain?",
     "expected_keywords": ["Drug B", "Drug AE", "Drug AJ"], "should_refuse": False},
    {"id": 38, "cat": "safety", "query": "Which drugs have risks related to the heart?",
     "expected_keywords": ["Drug F", "Drug AD"], "should_refuse": False},
    {"id": 39, "cat": "safety", "query": "Is Drug D safe for long-term use?",
     "expected_keywords": ["dependency"], "should_refuse": False},
    {"id": 40, "cat": "safety", "query": "Which drugs can cause allergic reactions?",
     "expected_keywords": ["Drug C", "rash"], "should_refuse": False},
    {"id": 41, "cat": "safety", "query": "Does Drug AC have sensitivity warnings?",
     "expected_keywords": ["sensitivity to sunlight", "dry skin"], "should_refuse": False},

    # Category 7: Unanswerable
    {"id": 42, "cat": "unanswerable", "query": "What is the recommended dosage for Drug B?",
     "expected_keywords": [], "should_refuse": True},
    {"id": 43, "cat": "unanswerable", "query": "Can Drug A and Drug V be taken together?",
     "expected_keywords": [], "should_refuse": True},
    {"id": 44, "cat": "unanswerable", "query": "Which drug is used for malaria?",
     "expected_keywords": [], "should_refuse": True},
    {"id": 45, "cat": "unanswerable", "query": "What is the price of Drug M?",
     "expected_keywords": [], "should_refuse": True},
    {"id": 46, "cat": "unanswerable", "query": "Is Drug F safe during pregnancy?",
     "expected_keywords": [], "should_refuse": True},

    # Category 8: Multi-Hop Reasoning
    {"id": 47, "cat": "multi_hop", "query": "If a patient has both hypertension and a dry cough, which drug might be causing the cough?",
     "expected_keywords": ["Drug V", "dry cough"], "should_refuse": False},
    {"id": 48, "cat": "multi_hop", "query": "A patient on a medication for chronic pain is experiencing confusion. Which drug are they likely taking?",
     "expected_keywords": ["Drug D", "confusion"], "should_refuse": False},
    {"id": 49, "cat": "multi_hop", "query": "Which drugs used for mental health conditions can cause weight gain?",
     "expected_keywords": ["Drug AE"], "should_refuse": False},
    {"id": 50, "cat": "multi_hop", "query": "A patient is being treated for allergies but is feeling very drowsy. What drug might they be on and why?",
     "expected_keywords": ["Drug N", "drowsiness"], "should_refuse": False},
]


def run_eval():
    categories = {}
    for tc in EVAL_CASES:
        categories.setdefault(tc["cat"], []).append(tc)

    total_score = 0
    total_count = len(EVAL_CASES)
    results_summary = []
    cat_scores = {}

    print("=" * 70)
    print(f"EVALUATION: {total_count} Questions from eval_questions.md")
    print(f"Retriever: HybridRetriever (3-layer + sibling linking)")
    print(f"Model: {MODEL_NAME}")
    print("=" * 70)

    pbar = tqdm(EVAL_CASES, desc="Evaluating", unit="q", ncols=80)
    for tc in pbar:
        query = tc["query"]
        query_type = classify_query(query)
        should_refuse = tc["should_refuse"]
        cat = tc["cat"]

        pbar.set_postfix_str(f"Q{tc['id']} ({cat})")

        retrieved = retriever.retrieve(query)
        answer = generate_answer(query, retrieved, query_type)

        if should_refuse:
            answer_lower = answer.lower()
            did_refuse = any(p in answer_lower for p in REFUSAL_PATTERNS)
            score = 1.0 if did_refuse else 0.0
            status = "PASS" if did_refuse else "FAIL"
        else:
            hits = [
                kw for kw in tc["expected_keywords"] if kw.lower() in answer.lower()
            ]
            score = len(hits) / len(tc["expected_keywords"]) if tc["expected_keywords"] else 0.0
            status = "PASS" if score >= 0.5 else "FAIL"

        total_score += score
        cat_scores.setdefault(cat, []).append(score)
        results_summary.append({
            "id": tc["id"],
            "query": query,
            "score": score,
            "status": status,
            "type": query_type,
            "cat": cat,
            "answer": answer[:200],
        })

    # -- Detailed results --
    print(f"\n\n{'=' * 70}")
    print("DETAILED RESULTS")
    print(f"{'=' * 70}")

    current_cat = None
    for r in results_summary:
        if r["cat"] != current_cat:
            current_cat = r["cat"]
            cat_avg = sum(cat_scores[current_cat]) / len(cat_scores[current_cat])
            print(f"\n--- {current_cat.upper()} (avg: {cat_avg:.0%}) ---")

        icon = "+" if r["status"] == "PASS" else "x"
        print(f"  [{icon}] Q{r['id']}: {r['query'][:60]}  -> {r['score']:.0%}")

    # -- Category summary --
    print(f"\n{'=' * 70}")
    print("CATEGORY SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Category':<25} {'Avg Score':>10} {'Passed':>10}")
    print(f"{'─' * 45}")
    for cat, scores in cat_scores.items():
        avg = sum(scores) / len(scores)
        passed = sum(1 for s in scores if s >= 0.5)
        print(f"{cat:<25} {avg:>9.0%} {passed:>6}/{len(scores)}")

    # -- Overall --
    avg = total_score / total_count
    passed = sum(1 for r in results_summary if r["status"] == "PASS")
    print(f"\n{'=' * 70}")
    print(f"OVERALL: {avg:.0%} ({total_score:.1f}/{total_count}) | Passed: {passed}/{total_count}")
    print(f"{'=' * 70}")

    return results_summary


if __name__ == "__main__":
    run_eval()
