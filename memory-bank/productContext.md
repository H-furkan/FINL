# Product Context

## Problem Statement
A pharmaceutical company has drug information scattered across documents.
Users cannot easily search or extract insights. We build an AI system to
answer natural language questions using this data.

## Target Users
- Datathon judges evaluating our system
- Simulated end-users: healthcare professionals seeking drug information

## User Experience Goals
1. Ask a question in plain English, get a structured answer
2. See which source documents the answer came from (transparency)
3. System gracefully handles unanswerable questions (no hallucination)
4. Works even if the LLM API is down (fallback mode)

## Competitive Differentiators (vs other hackathon teams)
- Drug-pair structure exploitation (not generic TF-IDF)
- 3-layer hybrid retrieval with score fusion
- Query classification routing
- Few-shot prompted LLM with train/eval split
- Polished Gradio UI with example buttons
- Automated evaluation with scoring table
- Hallucination prevention via threshold gating

## Presentation Angle
"We discovered the dataset's internal structure and built a system that
exploits it, rather than treating it as a bag of documents."
