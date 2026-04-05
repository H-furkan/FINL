"""Batch evaluation of the RAG pipeline against example_queries.md gold answers."""

import pandas as pd
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

API_KEY = "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"
MODEL_NAME = "qwen/qwen3.6-plus:free"

df = pd.read_csv("drug_docs.csv")
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["text"])

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=API_KEY)


def retrieve_documents(query, top_k=3):
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]
    results = df.iloc[top_indices].copy()
    results["score"] = similarities[top_indices]
    return results


def generate_answer(query, retrieved_docs):
    context = "\n".join(retrieved_docs["text"].tolist())
    prompt = f"""
You are a helpful medical assistant.

Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{query}
"""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[LLM ERROR: {e}]\nFallback:\n{context}"


EVAL_CASES = [
    {
        "id": 1,
        "query": "What are the side effects of Drug A?",
        "expected_keywords": ["dizziness", "headache", "fatigue"],
    },
    {
        "id": 2,
        "query": "Which drug is used for diabetes?",
        "expected_keywords": ["Drug B", "insulin", "blood sugar"],
    },
    {
        "id": 3,
        "query": "What treats hypertension?",
        "expected_keywords": ["Drug A", "blood vessels"],
    },
    {
        "id": 4,
        "query": "What are the side effects of Drug M?",
        "expected_keywords": ["fatigue", "hair loss", "nausea"],
    },
    {
        "id": 5,
        "query": "Which drugs are used for infections?",
        "expected_keywords": ["Drug C", "Drug AQ", "Drug AH"],
    },
    {
        "id": 6,
        "query": "Which drug is used for asthma and what are its side effects?",
        "expected_keywords": ["Drug F", "tremors", "nervousness", "heart rate"],
    },
    {
        "id": 7,
        "query": "Which drugs have dependency risks?",
        "expected_keywords": ["Drug D", "Drug J", "dependency"],
    },
    {
        "id": 8,
        "query": "Which drugs affect the brain?",
        "expected_keywords": ["Drug E", "Drug D", "Drug O"],
    },
    {
        "id": 9,
        "query": "Which drug cures cancer completely?",
        "expected_keywords": ["don't know", "not available", "not in"],
    },
    {
        "id": 10,
        "query": "Which drug is used for cholesterol and what risks does it have?",
        "expected_keywords": ["Drug G", "muscle pain", "liver"],
    },
]


def run_eval():
    print("=" * 70)
    print("EVALUATION: 10 Example Queries from example_queries.md")
    print("=" * 70)

    total_score = 0
    results_summary = []

    for tc in EVAL_CASES:
        print(f"\n{'─' * 70}")
        print(f"Q{tc['id']}: {tc['query']}")

        retrieved = retrieve_documents(tc["query"])
        print("  Retrieved docs (top scores): ", end="")
        print(", ".join(f"{row['score']:.3f}" for _, row in retrieved.iterrows()))

        answer = generate_answer(tc["query"], retrieved)
        print(f"  Answer: {answer[:200]}{'...' if len(answer) > 200 else ''}")

        hits = [kw for kw in tc["expected_keywords"] if kw.lower() in answer.lower()]
        misses = [
            kw for kw in tc["expected_keywords"] if kw.lower() not in answer.lower()
        ]
        score = len(hits) / len(tc["expected_keywords"])
        total_score += score

        status = "PASS" if score >= 0.5 else "FAIL"
        print(f"  Score: {score:.0%} ({len(hits)}/{len(tc['expected_keywords'])})")
        print(f"  Hits: {hits}")
        if misses:
            print(f"  Misses: {misses}")
        print(f"  Status: {status}")

        results_summary.append({"query": tc["query"], "score": score, "status": status})

    print(f"\n{'=' * 70}")
    avg = total_score / len(EVAL_CASES)
    print(f"OVERALL SCORE: {avg:.0%} ({total_score:.1f}/{len(EVAL_CASES)})")
    print(f"{'=' * 70}")

    passed = sum(1 for r in results_summary if r["status"] == "PASS")
    print(f"Passed: {passed}/{len(EVAL_CASES)}")

    return results_summary


if __name__ == "__main__":
    run_eval()
