"""LLM generation with query-type-aware prompts and structured fallback."""

from __future__ import annotations

import os

import pandas as pd
from openai import OpenAI

from rag.retrieval import HybridRetriever, classify_query

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
MODEL_NAME = os.environ.get("OPENROUTER_MODEL", "qwen/qwen3.6-plus:free")
BASE_URL = "https://openrouter.ai/api/v1"

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
- Drug AH: eye infections

Q: Which drug is used for asthma and what are its side effects?
A: Drug F is used for asthma. Side effects include tremors, nervousness, and increased heart rate."""

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

Answer clearly in natural language like the examples (short sentences; simple bullets only when listing several items).""",
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
The question asks about both what the drug treats and its side effects (or risks). Address both parts in plain language.
If the answer is not in the documents, say "I don't know based on the provided data."

Style: match the examples — conversational sentences, not a form. Do not use bold headings, and do not use label lines like "Used for:" and "Side effects:" as separate fields under the drug name.

{few_shot}

Now answer the following:

Documents:
{context}

Question: {query}

Answer:""",
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


# ---------------------------------------------------------------------------
# RAG Pipeline
# ---------------------------------------------------------------------------


class RAGPipeline:
    """Full RAG pipeline: retrieve + generate with fallback."""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.retriever = HybridRetriever()
        self.api_key = api_key or API_KEY
        self.model = model or MODEL_NAME

        if self.api_key:
            self.client = OpenAI(base_url=BASE_URL, api_key=self.api_key)
        else:
            self.client = None

    def answer(self, query: str) -> dict:
        """Answer a query. Returns dict with answer, sources, query_type, mode."""
        query_type = classify_query(query)
        results = self.retriever.retrieve(query)

        # Unanswerable detection
        if not results.attrs.get("is_answerable", True):
            return {
                "answer": "I don't know based on the provided data.",
                "sources": "",
                "query_type": query_type,
                "mode": "refused",
                "max_score": results.attrs.get("max_score", 0),
            }

        # Build enriched context
        context = self.retriever.build_enriched_context(results)

        # Try LLM generation
        answer_text, mode = self._generate(query, context, query_type)

        # Format source docs
        sources = self._format_sources(results)

        return {
            "answer": answer_text,
            "sources": sources,
            "query_type": query_type,
            "mode": mode,
            "max_score": results.attrs.get("max_score", 0),
        }

    def _generate(self, query: str, context: str, query_type: str) -> tuple[str, str]:
        """Try LLM, fall back to structured answer."""
        if self.client is None:
            return self._fallback(query), "fallback"

        template = PROMPT_TEMPLATES.get(query_type, PROMPT_TEMPLATES["drug_specific"])
        prompt = template.format(
            few_shot=FEW_SHOT_EXAMPLES,
            context=context,
            query=query,
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            return response.choices[0].message.content, "llm"
        except Exception:
            return self._fallback(query), "fallback"

    def _fallback(self, query: str) -> str:
        """Generate answer from structured data without LLM."""
        results = self.retriever.retrieve(query)
        return self.retriever.build_fallback_answer(results)

    def _format_sources(self, results: pd.DataFrame) -> str:
        """Format retrieved docs as source citations."""
        lines = []
        for i, (_, row) in enumerate(results.iterrows()):
            lines.append(f"[{i + 1}] {row['text']} (score: {row['score']:.3f})")
        return "\n".join(lines)
