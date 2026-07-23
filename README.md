#  Assessment Recommendation System

## Overview

This project is a conversational  Assessment Recommendation API that recommends the most suitable SHL assessments from natural language hiring requirements.

The system combines **hybrid retrieval**, **semantic search**, **neural reranking**, and **conversational clarification** to recommend grounded SHL assessments while supporting multi-turn interactions.

The API is built with **FastAPI** and deployed on **Railway**.

---

# Features

- Conversational recommendation engine
- Clarification questions for incomplete hiring requirements
- Multi-turn conversation support
- Recommendation refinement using conversation history
- Assessment comparison using SHL catalog data
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

Recruiters usually describe hiring requirements in natural language rather than structured filters.

For example,

> Senior Java Spring developer with AWS experience.

must be translated into relevant SHL assessments.

This project addresses that challenge through a hybrid retrieval pipeline combining lexical search, semantic retrieval, metadata filtering, and neural reranking.

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
- CrossEncoder
- FAISS
- Rank-BM25
- NetworkX
- NumPy
- Pydantic

---

# Evaluation

The recommendation pipeline was evaluated on representative hiring scenarios covering Java, Python, SQL, and Docker roles.

The evaluation simulates conversational interactions by automatically answering clarification questions before measuring recommendation quality.

| Metric | Score |
|--------|------:|
| Precision@10 | **0.1500** |
| Recall@10 | **0.9167** |
| Hit Rate | **1.0000** |
| MRR | **1.0000** |
| NDCG@10 | **0.9413** |

These results demonstrate that the hybrid retrieval and reranking pipeline consistently retrieves relevant SHL assessments while ranking the most relevant assessments at the top of the recommendation list.

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

### Request

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Python developer"
    }
  ],
  "top_k": 10
}
```

### Response

```json
{
  "reply": "I found 10 SHL assessments that best match your requirements.",
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

## Production API

```
https://shl-agent-production-2bdb.up.railway.app
```

### Health Endpoint

```
https://shl-agent-production-2bdb.up.railway.app/health
```

### Swagger UI

```
https://shl-agent-production-2bdb.up.railway.app/docs
```

---

# Repository Structure

```
SHL-Agent/
│
├── src/
│   ├── api.py
│   ├── parser.py
│   ├── retriever.py
│   ├── reranker.py
│   ├── explainer.py
│   ├── fusion.py
│   └── metadata_retriever.py
│
├── data/
├── graph/
├── embeddings/
├── evaluation/
│   ├── evaluate.py
│   └── test_queries.json
│
├── README.md
├── requirements.txt
└── main.py
```

---

# Future Improvements

- Learning-to-Rank models
- Better competency extraction
- LLM-assisted intent extraction
- Personalized recruiter preferences
- Feedback-driven ranking optimization
- Automatic conversation memory summarization

---

