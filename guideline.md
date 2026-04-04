# Build a Drug Knowledge Assistant (RAG Challenge)

---

## 🎯 Challenge Overview

You are working as an **AI engineer at EPAM**.

A pharmaceutical company has a large collection of internal documents about drugs (usage, effects, descriptions), but:

* information is hard to search
* users cannot easily get answers

👉 Your mission:

> Build a simple **RAG (Retrieval-Augmented Generation) system** that can answer questions using the provided documents.

---

## 📦 Dataset Information

You will be given a dataset containing **drug-related documents**.

### Example format (drug_docs.csv)

| doc_id | text                                     |
| ------ | ---------------------------------------- |
| 1      | Drug A is used to treat hypertension...  |
| 2      | Common side effects of Drug B include... |

### Notes:

* Each row is a **short document (2–5 sentences)**
* Documents may include:

  * drug usage
  * indications
  * side effects
  * descriptions

---

## 🛠️ Task

Your system should:

### 1. Accept a user question

Examples:

* “What are the side effects of Drug X?”
* “Which drugs are used for diabetes?”

---

### 2. Retrieve relevant documents

* Find the most relevant text(s) from the dataset

---

### 3. Generate an answer

* Return a clear answer based on retrieved documents

---

## ⚙️ Allowed Approaches

You are free to choose your approach.

### 🔹 Basic

* Keyword search
* TF-IDF
* Return most relevant documents

---

### 🔹 Intermediate

* Embeddings + similarity search
* Top-k document retrieval

---

### 🔹 Advanced

* Full RAG pipeline with an LLM
* Combine retrieved documents into a generated answer

---

## 📤 Expected Output

Each team should present:

### 1. Working System (Demo)

* Input: a question
* Output: an answer

---

### 2. Retrieval Approach

* How does your system find relevant documents?

---

### 3. Example Queries

* Show at least **2–3 sample questions and answers**

---

### 4. Improvements

* What would you improve with more time?

---

## 🏆 Evaluation Criteria

| Criteria                                     | Weight |
| -------------------------------------------- | ------ |
| 🤖 Functionality (End-to-end working system) | 40%    |
| 🔍 Retrieval Quality (Relevance of results)  | 25%    |
| 🧠 Approach & Design                         | 20%    |
| 🎤 Clarity of Demo & Explanation             | 15%    |

---

## 💡 Hints

* Start simple: even basic keyword search works
* Focus on **retrieval first**, then improve answers
* You don’t need a perfect model—working pipeline is key
* Try returning top 2–3 documents instead of just one
* Clean and structure your text before searching

---

## ⚠️ Constraints

* Time limit: **3 hours**
* Keep your solution **simple and explainable**
* Focus on **working prototype over perfection**

---

## 🎤 Final Presentation

* **5 minutes per team**
* Suggested structure:

  1. Approach
  2. Demo
  3. Key ideas
  4. Improvements

---

## 🚀 Goal

This challenge simulates a real-world task at EPAM Systems:

> Building AI systems that turn unstructured data into useful insights.