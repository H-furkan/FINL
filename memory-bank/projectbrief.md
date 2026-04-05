# Project Brief: Drug Knowledge Assistant

## Overview
RAG (Retrieval-Augmented Generation) system for a mini datathon organized by EPAM Systems.
The system answers drug-related questions using a provided dataset of 100 documents.

## Core Requirements
1. Accept a user question about drugs
2. Retrieve relevant documents from the dataset
3. Generate a clear answer based on retrieved documents

## Dataset
- `drug_docs.csv`: 100 documents covering 50 drugs (A through AX)
- Each drug has exactly 2 documents: one for usage/description, one for side effects
- Documents are short (1-2 sentences each)

## Constraints
- Time limit: 3 hours
- Must use free LLM APIs (OpenRouter or Hugging Face)
- Must work without LLM as fallback
- Do NOT send entire dataset to LLM -- use top 2-3 docs as context

## Evaluation Criteria
| Criteria | Weight |
|----------|--------|
| Functionality (working end-to-end) | 40% |
| Retrieval Quality | 25% |
| Approach & Design | 20% |
| Presentation (5-min demo) | 15% |

## Success Definition
A working prototype that demonstrates hybrid retrieval, query understanding,
hallucination prevention, and a polished UI -- beating teams using vanilla TF-IDF.
