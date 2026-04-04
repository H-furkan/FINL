# 🧪 Mini Datathon: Drug Knowledge Assistant (RAG Challenge)

This repository contains all materials for a **mini datathon** organized by EPAM Systems.

Participants will build a **Retrieval-Augmented Generation (RAG)** system to answer drug-related questions using a provided dataset.

---

## 🎯 Challenge Overview

You are an AI engineer working on a healthcare solution.

A pharmaceutical company has a collection of internal documents about drugs, including:

* usage
* side effects
* descriptions

👉 Your task is to build a system that:

1. Retrieves relevant documents
2. Generates answers based on them

---

## 📦 Repository Structure

```
.
├── drug_docs.csv                  # Dataset (100 drug-related documents)
├── example_working_pipeline.ipynb # Starter RAG notebook
├── guideline.md                   # Datathon instructions
├── example_queries.md            # Sample questions for testing
├── about_free_llm_apis.md        # Guide for using free LLM APIs
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd <repo-name>
```

---

### 2. Install Dependencies

```bash
pip install pandas scikit-learn openai
```

---

### 3. Run the Starter Notebook

Open:

```
example_working_pipeline.ipynb
```

Follow the steps to:

* load data
* build retrieval system
* integrate LLM
* test queries

---

## 🧠 What is RAG?

RAG (Retrieval-Augmented Generation) combines:

* 🔍 **Retrieval** → finding relevant documents
* 🤖 **Generation** → producing answers using an LLM

---

## 🛠️ Suggested Approach

1. Build a retrieval system (TF-IDF or embeddings)
2. Retrieve top relevant documents
3. Pass them as context to an LLM
4. Generate an answer

---

## 🆓 Free LLM Usage

You are expected to use **free LLM APIs**.

Recommended:

* OpenRouter (easiest)
* Hugging Face

📄 See:

```
about_free_llm_apis.md
```

---

## 💡 Example Questions

* What are the side effects of Drug A?
* Which drug is used for diabetes?
* What treats hypertension?

📄 See:

```
example_queries.md
```

---

## 🏆 Evaluation Criteria

| Criteria                       | Weight |
| ------------------------------ | ------ |
| Functionality (working system) | 40%    |
| Retrieval Quality              | 25%    |
| Approach & Design              | 20%    |
| Presentation                   | 15%    |

---

## ⚠️ Important Notes

* Start simple → retrieval first
* Do NOT send entire dataset to LLM
* Use top 2–3 documents as context
* Your system should still work without LLM (fallback)

---

## 🎯 Goal

This challenge simulates real-world AI engineering work at EPAM Systems:

> Turning unstructured data into intelligent, usable systems.

---

## 🤝 Good Luck!

Focus on:

* building a working system
* explaining your approach
* thinking like an engineer

🚀 Have fun and innovate!