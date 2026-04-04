# 🆓 Using Free LLM APIs in This Datathon

During this challenge, you are expected to build a **RAG (Retrieval-Augmented Generation) system**.

👉 The good news:
You can do everything **completely free** — no paid APIs required.

---

## 🎯 What You Should Do

Your system should:

1. Retrieve relevant documents from the dataset
2. Use an LLM (optional but recommended) to generate answers

---

## 🔑 Option 1: Use OpenRouter (Recommended ⭐)

This is the easiest and most reliable way to use LLMs for free.

---

### 🧑‍💻 Step 1: Create an Account

* Go to: [https://openrouter.ai](https://openrouter.ai)
* Sign up
* Generate your API key

---

### 📦 Step 2: Install Library

```bash
pip install openai
```

---

### ⚙️ Step 3: Use It in Your Code

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="YOUR_API_KEY"
)

def generate_answer_with_llm(query, context):
    prompt = f"""
    Answer the question using only the context below.

    Context:
    {context}

    Question:
    {query}
    """

    response = client.chat.completions.create(
        model="mistralai/mistral-7b-instruct",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content
```

---

### 💡 Suggested Free Models

You can use any of these:

* `mistralai/mistral-7b-instruct` ⭐ (recommended)
* `google/gemma-7b-it`
* `meta-llama/llama-3-8b-instruct`

---

## 🔑 Option 2: Use Hugging Face

If you prefer, you can also use Hugging Face.

---

### 📦 Install

```bash
pip install huggingface_hub
```

---

### ⚙️ Example

```python
from huggingface_hub import InferenceClient

client = InferenceClient()

response = client.text_generation(
    prompt="Explain side effects of a drug",
    model="mistralai/Mistral-7B-Instruct-v0.2"
)

print(response)
```

---

## 🧩 How to Connect This to Your RAG System

### Step 1: Retrieve documents

```python
docs = retrieve_documents(query)
context = "\n".join(docs["text"].tolist())
```

---

### Step 2: Send to LLM

```python
answer = generate_answer_with_llm(query, context)
```

---

## ⚠️ Important Guidelines

* Do NOT send the entire dataset to the LLM
* Only send **top 2–3 relevant documents**
* Keep your prompt short and clear
* Make sure your system works even without LLM

---

## 💡 Suggested Workflow

👉 Step 1: Make retrieval work (TF-IDF or similarity)
👉 Step 2: Test with sample queries
👉 Step 3: Add LLM for better answers

---

## 🚨 If APIs Don’t Work

No problem 👍

You can still complete the task by:

* returning the most relevant documents
* combining them into an answer

👉 This is totally acceptable.

---

## 🏆 Bonus Tip

If you use LLM effectively (good prompts + good context),
you may win:

👉 **“Best Use of AI”**

---

## 🎯 Final Advice

* Keep it simple
* Focus on a working pipeline
* Explain your approach clearly
