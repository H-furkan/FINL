"""Hybrid 3-layer retrieval engine with query classification."""

from __future__ import annotations

import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from rag.data import (
    KG_CSV_PATH,
    _DRUG_NAME_RE,
    build_condition_index,
    build_drug_index,
    build_side_effect_index,
    load_data,
    load_knowledge_csv,
)

# ---------------------------------------------------------------------------
# Query classification
# ---------------------------------------------------------------------------

QUERY_TYPES = (
    "drug_specific",
    "condition_lookup",
    "comparison",
    "multi_hop",
    "safety",
    "general",
)


def classify_query(query: str) -> str:
    """Classify a query into a type to route retrieval strategy."""
    q = query.lower()
    drug_mentioned = bool(re.search(r"\bdrug [a-z]{1,2}\b", q))
    asks_side_effects = any(
        w in q for w in ["side effect", "risk", "cause", "danger", "dependency"]
    )
    asks_usage = any(
        w in q for w in ["used for", "treat", "prescribed", "manage", "help"]
    )
    asks_comparison = any(
        w in q for w in ["which drugs", "what drugs", "compare", "list all"]
    )
    asks_brain_body = any(
        w in q for w in ["affect the brain", "affect the body", "act on"]
    )

    # Multi-hop: asks about BOTH usage and side effects (with or without drug name)
    if asks_side_effects and asks_usage:
        return "multi_hop"
    elif drug_mentioned:
        return "drug_specific"
    # Safety: side effects / risks without a specific drug, even if "which drugs"
    elif asks_side_effects:
        return "safety"
    elif asks_comparison or asks_brain_body:
        return "comparison"
    elif asks_usage:
        return "condition_lookup"
    else:
        return "general"


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

UNANSWERABLE_THRESHOLD = 0.10
SCORE_DROP_RATIO = 0.4  # cut off when next score < 40% of previous score
MIN_RESULTS = 2  # always keep at least 2 docs


class HybridRetriever:
    """3-layer hybrid retriever with drug-aware sibling linking."""

    def __init__(self, df: pd.DataFrame | None = None):
        if df is None:
            df = load_data()
        self.df = df

        # Phase 1 indexes
        self.drug_index = build_drug_index(df)
        self.condition_index = build_condition_index(self.drug_index)
        self.side_effect_index = build_side_effect_index(self.drug_index)

        # Structured knowledge from drug_knowledge.csv
        if KG_CSV_PATH.exists():
            self.knowledge_df = load_knowledge_csv()
            self._kg_lookup = {
                row["drug_name"]: row for _, row in self.knowledge_df.iterrows()
            }
        else:
            self.knowledge_df = None
            self._kg_lookup = {}

        # TF-IDF layer
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.vectorizer.fit_transform(df["text"])

    # -- Layer 1: exact drug name match ----------------------------------

    _DRUG_NAME_CI = re.compile(r"\bdrug ([a-z]{1,2})\b", re.IGNORECASE)

    def _layer_drug_match(self, query: str) -> dict[int, float]:
        """Extract drug names from query and score their docs at 1.0."""
        scores: dict[int, float] = {}
        matches = self._DRUG_NAME_CI.findall(query)
        for letter in matches:
            drug_name = f"Drug {letter.upper()}"
            if drug_name in self.drug_index:
                for doc_id in self.drug_index[drug_name]["all_docs"]:
                    scores[doc_id] = scores.get(doc_id, 0) + 1.0
        return scores

    # -- Layer 2: TF-IDF cosine similarity --------------------------------

    def _layer_tfidf(self, query: str) -> dict[int, float]:
        """Score every doc by TF-IDF cosine similarity."""
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        return {i: float(sims[i]) for i in range(len(sims)) if sims[i] > 0}

    # -- Layer 3: condition / side-effect keyword match -------------------

    def _layer_keyword(self, query: str) -> dict[int, float]:
        """Match conditions and side-effect keywords from reverse indexes."""
        scores: dict[int, float] = {}
        q = query.lower()

        # Condition match
        for condition, drugs in self.condition_index.items():
            if condition in q:
                for drug in drugs:
                    for doc_id in self.drug_index[drug]["all_docs"]:
                        scores[doc_id] = scores.get(doc_id, 0) + 0.8

        # Side-effect keyword match (for safety queries)
        for side_effect, drugs in self.side_effect_index.items():
            if side_effect in q:
                for drug in drugs:
                    for doc_id in self.drug_index[drug]["all_docs"]:
                        scores[doc_id] = scores.get(doc_id, 0) + 0.6

        return scores

    # -- Sibling linker ---------------------------------------------------

    def _link_siblings(self, scored_doc_ids: dict[int, float]) -> dict[int, float]:
        """For every retrieved drug, ensure both usage + side_effect docs are included."""
        expanded = dict(scored_doc_ids)
        for doc_id in list(scored_doc_ids.keys()):
            # Find which drug this doc belongs to
            for drug_name, info in self.drug_index.items():
                if doc_id in info["all_docs"]:
                    # Add sibling docs with a small bonus
                    for sibling_id in info["all_docs"]:
                        if sibling_id not in expanded:
                            expanded[sibling_id] = 0.3
                    break
        return expanded

    # -- Score fusion + main retrieve ------------------------------------

    def retrieve(self, query: str, top_k: int = 5) -> pd.DataFrame:
        """Run hybrid retrieval: 3 layers + sibling linking + score fusion.

        Returns a DataFrame with columns: doc_id, text, score, is_answerable.
        """
        query_type = classify_query(query)

        # Gather scores from all layers
        scores: dict[int, float] = {}

        # Layer 1: exact drug match (weight 1.0)
        for doc_id, s in self._layer_drug_match(query).items():
            scores[doc_id] = scores.get(doc_id, 0) + s

        # Layer 2: TF-IDF (weight 0.5)
        for doc_id, s in self._layer_tfidf(query).items():
            scores[doc_id] = scores.get(doc_id, 0) + 0.5 * s

        # Layer 3: keyword match (already weighted inside)
        for doc_id, s in self._layer_keyword(query).items():
            scores[doc_id] = scores.get(doc_id, 0) + s

        # Sibling linking -- pull both docs for any matched drug
        scores = self._link_siblings(scores)

        if not scores:
            return self._empty_result()

        # Rank and take top-k
        ranked = sorted(scores.items(), key=lambda x: -x[1])

        # For comparison/safety queries, allow more docs
        if query_type in ("comparison", "safety"):
            top_k = max(top_k, 10)

        top_ids = [doc_id for doc_id, _ in ranked[:top_k]]
        top_scores = [s for _, s in ranked[:top_k]]

        # Score drop-off filter: cut where score drops sharply
        cut = len(top_scores)
        for i in range(1, len(top_scores)):
            if top_scores[i - 1] > 0 and top_scores[i] / top_scores[i - 1] < SCORE_DROP_RATIO:
                cut = max(i, MIN_RESULTS)
                break
        top_ids = top_ids[:cut]
        top_scores = top_scores[:cut]

        results = self.df.iloc[top_ids].copy()
        results["score"] = top_scores

        # Unanswerable detection: check if best score is too low
        max_score = top_scores[0] if top_scores else 0
        results.attrs["is_answerable"] = max_score >= UNANSWERABLE_THRESHOLD
        results.attrs["query_type"] = query_type
        results.attrs["max_score"] = max_score

        return results

    # -- Structured knowledge enrichment -----------------------------------

    def get_drug_summary(self, drug_name: str) -> str | None:
        """Get a structured summary for a drug from drug_knowledge.csv."""
        if drug_name not in self._kg_lookup:
            return None
        row = self._kg_lookup[drug_name]
        parts = [f"{drug_name}:"]
        if row["conditions"]:
            parts.append(f"  Treats: {row['conditions']}")
        if row["side_effects"]:
            parts.append(f"  Side effects: {row['side_effects']}")
        return "\n".join(parts)

    def get_matched_drugs(self, results: pd.DataFrame) -> list[str]:
        """Extract unique drug names from retrieval results."""
        drugs = []
        for _, row in results.iterrows():
            match = _DRUG_NAME_RE.search(row["text"])
            if match:
                name = f"Drug {match.group(1)}"
                if name not in drugs:
                    drugs.append(name)
        return drugs

    def build_enriched_context(self, results: pd.DataFrame) -> str:
        """Build context that combines raw docs + structured knowledge summaries.

        Returns a string with raw retrieved docs followed by structured summaries
        from drug_knowledge.csv for each matched drug.
        """
        # Raw documents
        raw_docs = "\n".join(
            f"[Doc {i + 1}] {row['text']}" for i, (_, row) in enumerate(results.iterrows())
        )

        # Structured summaries from drug_knowledge.csv
        drugs = self.get_matched_drugs(results)
        summaries = []
        for drug in drugs:
            s = self.get_drug_summary(drug)
            if s:
                summaries.append(s)

        if summaries:
            structured = "\n".join(summaries)
            return f"Retrieved Documents:\n{raw_docs}\n\nStructured Drug Data:\n{structured}"

        return raw_docs

    def build_fallback_answer(self, results: pd.DataFrame) -> str:
        """Build a clean answer from structured data when LLM is unavailable."""
        drugs = self.get_matched_drugs(results)
        if not drugs:
            return "No relevant information found in the database."

        parts = []
        for drug in drugs:
            if drug in self._kg_lookup:
                row = self._kg_lookup[drug]
                lines = [f"**{drug}**"]
                if row["conditions"]:
                    lines.append(f"- Used for: {row['conditions']}")
                if row["side_effects"]:
                    lines.append(f"- Side effects: {row['side_effects']}")
                parts.append("\n".join(lines))
            else:
                # Fall back to raw drug_index
                info = self.drug_index.get(drug)
                if info:
                    parts.append(f"**{drug}**\n- {info['usage_text']}")

        return "\n\n".join(parts) if parts else "No relevant information found."

    def _empty_result(self) -> pd.DataFrame:
        result = pd.DataFrame(columns=["doc_id", "text", "score"])
        result.attrs["is_answerable"] = False
        result.attrs["query_type"] = "general"
        result.attrs["max_score"] = 0.0
        return result
