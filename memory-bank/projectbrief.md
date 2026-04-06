# Project Brief: Drug Knowledge Assistant

## Overview
A hybrid RAG system that answers drug-related questions. Originally built for the EPAM Mini Datathon, now publicly released and deployed.

## What It Does
1. Accepts natural language questions about drugs
2. Retrieves relevant documents using 3-layer hybrid retrieval
3. Generates answers using free LLMs (with structured fallback)

## Dataset
- `drug_docs.csv`: 100 documents covering 50 drugs (A through AX)
- Each drug has exactly 2 documents: one for usage, one for side effects

## Performance
- 90% accuracy on 50-question eval suite
- 74% accuracy in fallback mode (no LLM)

## Deployment
- Live demo on HuggingFace Spaces
- Project website on GitHub Pages
- Source code on GitHub
