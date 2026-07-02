# SHL Assessment Recommendation System

## Overview

This project is a conversational SHL Assessment Recommendation API that recommends the most suitable SHL assessments from natural language hiring requirements.

The system combines hybrid retrieval, semantic search, neural reranking, and conversational clarification to recommend grounded SHL assessments while supporting multi-turn interactions.

The API is built with **FastAPI** and deployed on **Railway**.

---

# Features

- Conversational recommendation engine
- Clarification questions for incomplete hiring requirements
- Multi-turn conversation support
- Recommendation refinement using conversation history
- Assessment comparison
- Out-of-scope request handling
- Hybrid Retrieval (BM25 + Dense Retrieval)
- FAISS semantic search
- Reciprocal Rank Fusion (RRF)
- Cross-Encoder reranking
- Explainable recommendations
- FastAPI REST API
- Railway deployment

---

# Motivation

Recruiters typically describe hiring requirements in natural language rather than structured filters.

For example,

> Senior Java Spring developer with AWS experience.

must be translated into relevant SHL assessments.

This project addresses that challenge through a hybrid retrieval pipeline that combines lexical search, semantic retrieval, metadata filtering, and neural reranking.

---

# Relation to Previous Projects

This work extends techniques developed in my previous AI and retrieval projects.

## Hybrid Retrieval

Previously, I developed hybrid retrieval systems combining lexical and semantic retrieval for scientific and molecular datasets.

The same principles are applied here through:

- BM25 sparse retrieval
- FAISS dense retrieval
- Metadata filtering

---

## Neural Re-ranking

In previous retrieval systems I explored transformer-based reranking to improve search quality.

The same architecture is adopted here:

```
Candidate Retrieval
        │
        ▼
Cross Encoder
        │
        ▼
Final Ranking
```

Cross-Encoder reranking improves recommendation quality beyond embedding similarity alone.

---

## Conversational AI

Unlike earlier retrieval projects that processed single queries, this system supports:

- Clarification questions
- Multi-turn conversations
- Recommendation refinement
- Assessment comparison
- Out-of-scope detection

making the system closer to a production conversational assistant.

---

# System Architecture

```
                 User Query
                      │
                      ▼
               Intent Parser
                      │
                      ▼
            Structured Intent
                      │
                      ▼
         Hybrid Candidate Retrieval
          ┌───────────────────────┐
          │ BM25                  │
          │ FAISS Dense Retrieval │
          │ Metadata Retrieval    │
          └───────────────────────┘
                      │
                      ▼
        Reciprocal Rank Fusion (RRF)
                      │
                      ▼
          Cross-Encoder Re-ranking
                      │
                      ▼
        Recommendation Explainer
                      │
                      ▼
             FastAPI REST API
                      │
                      ▼
              Conversational Response
```

---

# Technologies

- Python
- FastAPI
- Sentence Transformers
- Cross Encoder
- FAISS
- Rank-BM25
- NumPy
- NetworkX
- Pydantic

---

# Evaluation

The retrieval pipeline was evaluated using representative hiring scenarios covering Java, Python, SQL, and Docker roles.

| Metric | Score |
|--------|------:|
| Precision@5 | **0.2500** |
| Recall@5 | **0.7917** |
| Hit Rate | **1.0000** |
| MRR | **0.8750** |
| NDCG@5 | **0.7881** |

These results demonstrate that the hybrid retrieval pipeline consistently retrieves relevant SHL assessments while ranking the most appropriate recommendations near the top of the result list.

---

# API

## Health Check

```
GET /health
```

Response

```json
{
  "status": "ok"
}
```

---

## Recommendation API

```
POST /chat
```

Request

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
  "reply": "I found 5 SHL assessments...",
  "recommendations": [
    {
      "name": "Python (New)",
      "url": "https://www.shl.com/...",
      "test_type": "K",
      "duration": 11,
      "reason": [
        "Matched skills: python",
        "Recommended by hybrid retrieval"
      ]
    }
  ],
  "end_of_conversation": true
}
```

---

# Deployment

**Production API**

https://shl-agent-production-2bdb.up.railway.app

**Health Endpoint**

https://shl-agent-production-2bdb.up.railway.app/health

**Swagger Documentation**

https://shl-agent-production-2bdb.up.railway.app/docs

---

# Repository Structure

```
src/
│── api.py
│── parser.py
│── retriever.py
│── reranker.py
│── explainer.py
│── fusion.py

data/
graph/
embeddings/
evaluation/

README.md
requirements.txt
```

---

# Future Improvements

- Learning-to-Rank models
- Better behavioral competency extraction
- LLM-assisted intent extraction
- Feedback-driven ranking optimization
- Personalized recruiter preferences

---

# Acknowledgements

Developed as part of the **SHL Generative AI Assessment Recommendation Challenge**.
