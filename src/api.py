from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  
from pydantic import BaseModel

from src.explainer import RecommendationExplainer
from src.parser import IntentParser
from src.retriever import HybridRetriever
from src.reranker import CrossEncoderReranker


# ============================================================
# FastAPI
# ============================================================

app = FastAPI(
    title="SHL Assessment Recommendation API",
    description="Conversational SHL Assessment Recommender",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Load Components
# ============================================================

parser = IntentParser()

retriever = HybridRetriever(
    catalog_path="data/catalog/catalog.json",
    graph_path="graph/assessment_graph.pkl",
    embedding_path="embeddings/assessment_embeddings.npy",
    faiss_path="embeddings/assessment.index",
    ids_path="embeddings/assessment_ids.json",
)

reranker = CrossEncoderReranker()

explainer = RecommendationExplainer()

print("SHL Agent Ready")


# ============================================================
# API Models
# ============================================================

class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]
    top_k: int = 5


# ============================================================
# Helper Functions
# ============================================================

def build_context(messages: List[Message]):
    return "\n".join(
        message.content
        for message in messages
    )


def latest_user_message(messages):
    for message in reversed(messages):
        if message.role.lower() == "user":
            return message.content

    return ""


def needs_clarification(intent):
    if len(intent["skills"]) > 0:
        return False

    if len(intent["domains"]) > 0:
        return False

    if len(intent["assessment_types"]) > 0:
        return False

    return True


def build_reply(intent, recommendations):
    if len(recommendations) == 0:
        return (
            "I couldn't find a suitable SHL assessment. "
            "Could you provide more details about the role?"
        )

    skills = ", ".join(intent["skills"])

    if skills:
        return (
            f"I found {len(recommendations)} SHL assessments "
            f"that match a role requiring {skills}."
        )

    return (
        f"I found {len(recommendations)} SHL assessments "
        "matching your requirements."
    )


def get_test_type(assessment):
    categories = " ".join(
        assessment.get("category", [])
    ).lower()

    if "personality" in categories:
        return "P"

    if "simulation" in categories:
        return "S"

    return "K"


# ============================================================
# Comparison Helpers
# ============================================================

def is_comparison_query(query: str):
    q = query.lower()

    return (
        "compare" in q
        or "difference" in q
        or "vs" in q
        or "versus" in q
    )


def is_out_of_scope(query: str):
    q = query.lower()

    blocked = [
        "salary",
        "investment",
        "stock",
        "crypto",
        "bitcoin",
        "politics",
        "medical advice",
        "legal advice",
    ]

    return any(word in q for word in blocked)


# ============================================================
# Health Endpoint
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "ok"
    }


# ============================================================
# Chat Endpoint
# ============================================================

@app.post("/chat")
def chat(request: ChatRequest):
    try:
        # --------------------------------------------------
        # Conversation Context
        # --------------------------------------------------
        conversation = build_context(
            request.messages
        )

        latest_query = latest_user_message(
            request.messages
        )

        # --------------------------------------------------
        # Out-of-scope handling
        # --------------------------------------------------
        if is_out_of_scope(conversation):
            return {
                "reply": "I can only answer questions related to SHL assessments and recommend assessments from the SHL catalog.",
                "recommendations": [],
                "end_of_conversation": True,
            }

        # --------------------------------------------------
        # Comparison
        # --------------------------------------------------
        import re

        if is_comparison_query(conversation):
            names = re.split(
                r"\b(?:vs|versus|compare|and)\b",
                latest_query,
                flags=re.IGNORECASE,
            )

            names = [
                n.strip()
                for n in names
                if len(n.strip()) > 2
            ]

            if len(names) >= 2:
                a = retriever.get_assessment_by_name(
                    names[0]
                )

                b = retriever.get_assessment_by_name(
                    names[1]
                )

                if a and b:
                    comparison = f"""
Comparison

{names[0]}

Category: {', '.join(a.get('category', []))}
Duration: {a.get('duration_minutes')} minutes
Remote: {a.get('remote')}
Adaptive: {a.get('adaptive')}

{names[1]}

Category: {', '.join(b.get('category', []))}
Duration: {b.get('duration_minutes')} minutes
Remote: {b.get('remote')}
Adaptive: {b.get('adaptive')}
"""

                    return {
                        "reply": comparison,
                        "recommendations": [],
                        "end_of_conversation": True,
                    }

            return {
                "reply": "Please specify the two SHL assessments you want to compare.",
                "recommendations": [],
                "end_of_conversation": False,
            }

        # --------------------------------------------------
        # Intent
        # --------------------------------------------------
        intent = parser.parse(
            conversation
        )

        # --------------------------------------------------
        # Clarification
        # --------------------------------------------------
        if needs_clarification(intent):
            return {
                "reply": (
                    "Could you tell me more about the role? "
                    "For example, required skills, seniority, "
                    "or whether you're looking for technical, "
                    "personality, or leadership assessments."
                ),
                "recommendations": [],
                "end_of_conversation": False,
            }

        # --------------------------------------------------
        # Retrieval
        # --------------------------------------------------
        candidates = retriever.retrieve(
            intent,
            top_k=50,
        )

        # --------------------------------------------------
        # Cross Encoder
        # --------------------------------------------------
        ranked = reranker.rerank(
            conversation,
            candidates,
            top_k=request.top_k,
        )

        # --------------------------------------------------
        # Build Recommendations
        # --------------------------------------------------
        recommendations = []

        for item in ranked:
            assessment = item["assessment"]

            recommendations.append({
                "name": assessment.get("name"),
                "url": assessment.get("url"),
                "test_type": get_test_type(assessment),
                # ---------- optional debugging ----------
                "category": assessment.get("category", []),
                "duration": assessment.get("duration_minutes"),
                "reason": explainer.explain(
                    assessment,
                    intent,
                    item.get("source", "rrf"),
                ),
            })

        # --------------------------------------------------
        # Reply
        # --------------------------------------------------
        reply = build_reply(
            intent,
            recommendations,
        )

        return {
            "reply": reply,
            "recommendations": recommendations,
            "end_of_conversation": True,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


# ============================================================
# Root
# ============================================================

@app.get("/")
def root():
    return {
        "message": "SHL Conversational Assessment Recommender",
        "version": "3.0.0",
        "health": "/health",
        "chat": "/chat",
        "docs": "/docs",
    }