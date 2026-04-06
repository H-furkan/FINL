# Product Context

## What This Is
A Drug Knowledge Assistant — a hybrid RAG system that answers drug-related questions using a dataset of 100 documents covering 50 drugs. Built for the EPAM Mini Datathon, now publicly released.

## Live Deployment
- **Demo**: https://huggingface.co/spaces/Fo-zh/drug-knowledge-assistant
- **Website**: https://h-furkan.github.io/FINL/
- **GitHub**: https://github.com/H-furkan/FINL

## What Makes It Different
1. **Drug-pair structure exploitation** — every drug has exactly 2 docs (usage + side effects); sibling linking ensures both are always retrieved together
2. **3-layer hybrid retrieval** with score fusion (not just TF-IDF)
3. **Query classification** routes to optimal retrieval strategy and prompt template
4. **Hallucination prevention** via score threshold gating
5. **Works without LLM** — structured fallback achieves 74% accuracy from CSV alone
6. **API key rotation** — cycles through multiple OpenRouter keys to handle free tier rate limits

## Performance
- 90% accuracy on 50-question eval (47/50 passed)
- 100% on 6 of 8 categories
- Weakest areas: comparison (61%) and safety (71%) queries

## Future Direction
The current dataset uses synthetic "Drug A" through "Drug AX". The next step is replacing this with real medicine data (FDA OpenFDA API, DrugBank) to make the system practically useful.
