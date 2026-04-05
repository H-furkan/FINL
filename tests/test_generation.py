"""Tests for the RAG generation pipeline."""

from rag.generation import RAGPipeline


class TestRAGPipeline:
    def setup_method(self):
        # No API key -> always uses fallback mode
        self.pipeline = RAGPipeline(api_key="")

    def test_answer_returns_dict(self):
        result = self.pipeline.answer("What are the side effects of Drug A?")
        assert "answer" in result
        assert "sources" in result
        assert "query_type" in result
        assert "mode" in result

    def test_fallback_mode_without_api_key(self):
        result = self.pipeline.answer("What treats hypertension?")
        assert result["mode"] == "fallback"
        assert "Drug A" in result["answer"]

    def test_drug_specific_query_type(self):
        result = self.pipeline.answer("What are the side effects of Drug F?")
        assert result["query_type"] == "drug_specific"
        assert "Drug F" in result["answer"]

    def test_multi_hop_has_both_parts(self):
        result = self.pipeline.answer(
            "Which drug is used for asthma and what are its side effects?"
        )
        assert result["query_type"] == "multi_hop"
        assert "Drug F" in result["answer"]
        assert "asthma" in result["answer"].lower()

    def test_comparison_finds_multiple_drugs(self):
        result = self.pipeline.answer("Which drugs are used for infections?")
        assert result["query_type"] == "comparison"
        answer = result["answer"]
        infection_drugs = ["Drug C", "Drug AQ", "Drug AH"]
        found = sum(1 for d in infection_drugs if d in answer)
        assert found >= 2

    def test_safety_finds_dependency(self):
        result = self.pipeline.answer("Which drugs have dependency risks?")
        assert result["query_type"] == "safety"
        assert "Drug D" in result["answer"] or "Drug J" in result["answer"]

    def test_sources_included(self):
        result = self.pipeline.answer("What treats diabetes?")
        assert "score:" in result["sources"]
        assert len(result["sources"]) > 0

    def test_max_score_present(self):
        result = self.pipeline.answer("Drug G")
        assert result["max_score"] > 0
