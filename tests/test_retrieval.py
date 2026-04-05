from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def make_retriever(df):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(df["text"])

    def retrieve(query, top_k=3):
        query_vec = vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]
        results = df.iloc[top_indices].copy()
        results["score"] = similarities[top_indices]
        return results

    return retrieve


def test_retrieve_returns_top_k():
    df = pd.DataFrame(
        {
            "doc_id": [1, 2, 3, 4],
            "text": [
                "Drug A treats headaches and migraines",
                "Drug B is used for diabetes",
                "Side effects of Drug A include nausea",
                "Drug C helps with blood pressure",
            ],
        }
    )
    retrieve = make_retriever(df)
    results = retrieve("Drug A side effects", top_k=2)
    assert len(results) == 2


def test_retrieve_scores_are_sorted_descending():
    df = pd.DataFrame(
        {
            "doc_id": [1, 2, 3],
            "text": [
                "aspirin treats pain",
                "ibuprofen treats pain and inflammation",
                "paracetamol treats fever",
            ],
        }
    )
    retrieve = make_retriever(df)
    results = retrieve("pain treatment")
    scores = results["score"].tolist()
    assert scores == sorted(scores, reverse=True)


def test_retrieve_with_real_data():
    df = pd.read_csv(PROJECT_ROOT / "drug_docs.csv", skipinitialspace=True)
    retrieve = make_retriever(df)
    results = retrieve("What are the side effects of Drug A?", top_k=3)
    assert len(results) == 3
    assert all(results["score"] > 0)
