import pandas as pd
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

API_KEY = "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"
MODEL_NAME = "qwen/qwen3.6-plus:free"

df = pd.read_csv("drug_docs.csv")
print("✅ Dataset loaded. Number of documents:", len(df))

vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["text"])
print("✅ TF-IDF vectorizer ready.")


def retrieve_documents(query, top_k=3):
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]
    results = df.iloc[top_indices].copy()
    results["score"] = similarities[top_indices]
    return results


client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY,
)


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
        print("⚠️ LLM error:", e)
        print("Returning fallback answer...\n")
        return "Fallback answer:\n" + context


print("\n💊 Drug Knowledge Assistant is ready!")
print("Type 'exit' to quit.\n")

while True:
    query = input("🔎 Ask a question: ")
    if query.lower() == "exit":
        print("👋 Goodbye!")
        break

    retrieved = retrieve_documents(query)
    print("\n📄 Retrieved Documents:")
    for _i, row in retrieved.iterrows():
        print(f"- {row['text']} (score={row['score']:.3f})")

    answer = generate_answer(query, retrieved)
    print("\n🤖 Answer:")
    print(answer)
    print("\n" + "=" * 50 + "\n")
