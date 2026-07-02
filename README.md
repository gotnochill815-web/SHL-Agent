# SHL Assessment Recommendation System

## Overview

This project is an AI-powered conversational recommendation system that recommends the most suitable SHL assessments based on natural language hiring requirements.

Unlike traditional keyword search, the system combines semantic retrieval, sparse retrieval, knowledge graph expansion, business-rule reasoning, and cross-encoder reranking to produce high-quality recommendations while supporting conversational interactions.

The system is deployed using FastAPI and exposed through ngrok for real-time inference.

---

# Motivation

Recruiters often describe hiring requirements in natural language rather than using structured filters. Translating those requirements into relevant assessments requires understanding skills, domains, seniority, assessment types, and conversational context.

This project addresses that challenge using a hybrid retrieval pipeline that combines multiple retrieval strategies with neural reranking.

---

# Relation to My Previous Projects

This project builds directly upon techniques I explored in my previous AI and retrieval research projects.

### Hybrid Retrieval

Previously, I implemented hybrid retrieval systems combining dense embeddings with lexical retrieval for scientific and molecular datasets. The same retrieval principles are applied here by combining:

- FAISS semantic retrieval
- BM25 keyword retrieval
- Metadata retrieval
- Knowledge Graph expansion

instead of relying on a single search technique.

---

### Knowledge Graph Reasoning

In my earlier Drug Repurposing and Molecular AI projects, I used graph-based representations to propagate information across related biological entities.

This project adopts a similar idea by constructing an Assessment Knowledge Graph where assessments are connected through shared metadata such as:

- Skills
- Categories
- Job Levels
- Assessment Types

Graph expansion allows discovery of relevant assessments that are not retrieved directly by semantic search.

---

### Neural Re-ranking

During previous work with Cross Encoders and retrieval pipelines, I experimented with reranking retrieved candidates using transformer models.

The same architecture is used here:

Candidate Retrieval
↓

Cross Encoder

↓

Final Ranking

which significantly improves recommendation quality over embedding similarity alone.

---

### Business Rule Integration

Many recommendation systems optimize only semantic similarity.

Inspired by ranking pipelines used in production recommendation systems, this project additionally incorporates business-aware ranking signals including:

- Exact skill matching
- Assessment title matching
- Domain relevance
- Job level compatibility
- Assessment metadata

These signals improve ranking quality while keeping recommendations interpretable.

---

### Conversational AI

Unlike my previous retrieval systems that processed single queries, this project extends retrieval into a conversational recommendation agent capable of:

- Clarification questions
- Multi-turn conversations
- Recommendation refinement
- Assessment comparison
- Out-of-scope detection

making the system significantly closer to a production conversational assistant.

---

# System Architecture

User Query
        │
        ▼
Intent Parser
        │
        ▼
Hybrid Retrieval
 ├── Semantic Search (FAISS)
 ├── BM25
 ├── Metadata Retrieval
 └── Knowledge Graph Expansion
        │
        ▼
Business Rule Ranking
        │
        ▼
Cross Encoder Re-ranking
        │
        ▼
Recommendation Generator
        │
        ▼
FastAPI REST API
        │
        ▼
Conversational Response

---

# Features

- Conversational recommendation engine
- Hybrid Retrieval
- FAISS semantic search
- BM25 sparse retrieval
- Metadata retrieval
- Knowledge Graph expansion
- Cross Encoder reranking
- Business rule ranking
- Clarification handling
- Recommendation refinement
- Assessment comparison
- Out-of-scope detection
- FastAPI deployment
- ngrok deployment

---

# Technologies

- Python
- FastAPI
- Sentence Transformers
- Cross Encoder
- FAISS
- Rank-BM25
- NetworkX
- NumPy
- Pydantic

---

# Evaluation

| Metric | Score |
|---------|------:|
| Precision@5 | 0.25 |
| Recall@5 | 0.7917 |
| Hit Rate | 1.000 |
| MRR | 0.875 |
| NDCG@5 | 0.7881 |

These results demonstrate that the hybrid retrieval and reranking pipeline consistently retrieves relevant SHL assessments while ranking the most appropriate recommendations at the top.

---

# API

## POST /chat

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Python developer"
    }
  ],
  "top_k": 5
}
```

Response

```json
{
  "reply": "...",
  "recommendations": [
    {
      "name": "Python (New)",
      "url": "...",
      "test_type": "K"
    }
  ],
  "end_of_conversation": true
}
```

---

## GET /health

Returns the server status.

---

# Deployment

The system is deployed using:

- FastAPI
- Uvicorn
- ngrok

allowing external access for evaluation and demonstration.

---

# Future Improvements

- Learning-to-Rank models
- Personalized recruiter preferences
- LLM-based query rewriting
- Better assessment comparison
- Feedback-driven ranking optimization

---

# Repository Structure

```
src/
    api.py
    retriever.py
    reranker.py
    parser.py
    explainer.py
    memory.py

data/
graph/
embeddings/
evaluation/
README.md
```

---

# Acknowledgements

Built as part of the SHL Generative AI Assessment Recommendation Challenge.
