"""Tests for the hybrid retrieval engine and query classifier."""

from rag.retrieval import HybridRetriever, classify_query


class TestClassifyQuery:
    def test_drug_specific(self):
        assert classify_query("What are the side effects of Drug A?") == "drug_specific"

    def test_condition_lookup(self):
        assert classify_query("What treats hypertension?") == "condition_lookup"

    def test_comparison(self):
        assert classify_query("Which drugs are used for infections?") == "comparison"

    def test_multi_hop(self):
        assert (
            classify_query(
                "Which drug is used for asthma and what are its side effects?"
            )
            == "multi_hop"
        )

    def test_safety(self):
        assert classify_query("Which drugs have dependency risks?") == "safety"

    def test_brain_as_comparison(self):
        assert classify_query("Which drugs affect the brain?") == "comparison"

    def test_general(self):
        assert classify_query("Tell me about Drug Z") == "drug_specific"

    def test_lowercase_drug(self):
        assert classify_query("tell me about drug b") == "drug_specific"

    def test_mixed_case_drug(self):
        assert classify_query("What are the side effects of drug A?") == "drug_specific"


class TestHybridRetriever:
    def setup_method(self):
        self.retriever = HybridRetriever()

    def test_drug_a_side_effects_retrieves_both_docs(self):
        results = self.retriever.retrieve("What are the side effects of Drug A?")
        texts = " ".join(results["text"].tolist())
        # Should retrieve both usage and side-effect docs for Drug A
        assert "dizziness" in texts.lower() or "headache" in texts.lower()
        assert "hypertension" in texts.lower()

    def test_hypertension_finds_drug_a(self):
        results = self.retriever.retrieve("What treats hypertension?")
        texts = " ".join(results["text"].tolist())
        assert "Drug A" in texts

    def test_diabetes_finds_drug_b(self):
        results = self.retriever.retrieve("Which drug is used for diabetes?")
        texts = " ".join(results["text"].tolist())
        assert "Drug B" in texts

    def test_asthma_multi_hop_gets_usage_and_side_effects(self):
        results = self.retriever.retrieve(
            "Which drug is used for asthma and what are its side effects?"
        )
        texts = " ".join(results["text"].tolist())
        # Must find Drug F usage AND side effects (sibling linking)
        assert "Drug F" in texts
        assert "asthma" in texts.lower()
        assert "tremors" in texts.lower() or "nervousness" in texts.lower()

    def test_dependency_safety_query(self):
        results = self.retriever.retrieve("Which drugs have dependency risks?")
        texts = " ".join(results["text"].tolist())
        assert "Drug D" in texts or "Drug J" in texts
        assert "dependency" in texts.lower()

    def test_infections_comparison(self):
        results = self.retriever.retrieve("Which drugs are used for infections?")
        texts = " ".join(results["text"].tolist())
        # Should find multiple infection-related drugs
        infection_drugs = ["Drug C", "Drug AQ", "Drug AH"]
        found = sum(1 for d in infection_drugs if d in texts)
        assert found >= 2

    def test_unanswerable_detection(self):
        results = self.retriever.retrieve("What is the meaning of life?")
        # Low relevance query should be flagged
        assert results.attrs["max_score"] < 1.0

    def test_scores_descending(self):
        results = self.retriever.retrieve("What are the side effects of Drug A?")
        scores = results["score"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_returns_dataframe_with_score(self):
        results = self.retriever.retrieve("Drug B")
        assert "score" in results.columns
        assert "text" in results.columns
        assert len(results) > 0

    def test_query_type_attached(self):
        results = self.retriever.retrieve("What treats diabetes?")
        assert results.attrs["query_type"] == "condition_lookup"

    def test_lowercase_drug_b(self):
        results = self.retriever.retrieve("what are the side effects of drug b?")
        texts = " ".join(results["text"].tolist())
        assert "Drug B" in texts
        assert "nausea" in texts.lower()

    def test_mixed_case_drug_f(self):
        results = self.retriever.retrieve("drug F side effects")
        texts = " ".join(results["text"].tolist())
        assert "Drug F" in texts

    def test_cholesterol_multi_hop(self):
        results = self.retriever.retrieve(
            "Which drug is used for cholesterol and what risks does it have?"
        )
        texts = " ".join(results["text"].tolist())
        assert "Drug G" in texts
        assert "muscle pain" in texts.lower() or "liver" in texts.lower()

    def test_score_dropoff_filters_low_relevance(self):
        results = self.retriever.retrieve("What are the side effects of Drug A?")
        scores = results["score"].tolist()
        # Should keep the high-scoring Drug A docs, drop the low-scoring noise
        # All kept docs should be reasonably relevant (no huge gap)
        for i in range(1, len(scores)):
            if scores[i - 1] > 0:
                # No kept doc should be below 40% of its predecessor
                assert scores[i] / scores[i - 1] >= 0.4 or i < 2

    def test_score_dropoff_keeps_minimum(self):
        results = self.retriever.retrieve("Drug A")
        assert len(results) >= 2


class TestKnowledgeCSVIntegration:
    def setup_method(self):
        self.retriever = HybridRetriever()

    def test_drug_summary(self):
        summary = self.retriever.get_drug_summary("Drug A")
        assert summary is not None
        assert "hypertension" in summary.lower()
        assert "dizziness" in summary.lower()

    def test_drug_summary_missing(self):
        assert self.retriever.get_drug_summary("Drug ZZZ") is None

    def test_get_matched_drugs(self):
        results = self.retriever.retrieve("What treats diabetes?")
        drugs = self.retriever.get_matched_drugs(results)
        assert "Drug B" in drugs

    def test_enriched_context_has_both_sections(self):
        results = self.retriever.retrieve("What are the side effects of Drug A?")
        context = self.retriever.build_enriched_context(results)
        assert "Retrieved Documents:" in context
        assert "Structured Drug Data:" in context
        assert "Drug A" in context

    def test_fallback_answer_structured(self):
        results = self.retriever.retrieve("What treats asthma?")
        fallback = self.retriever.build_fallback_answer(results)
        assert "Drug F" in fallback
        assert "asthma" in fallback.lower()

    def test_fallback_empty_results(self):
        empty = self.retriever._empty_result()
        fallback = self.retriever.build_fallback_answer(empty)
        assert "No relevant" in fallback
