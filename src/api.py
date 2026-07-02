from typing import List
import re

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


def needs_clarification(intent, conversation):
    """
    Decide whether enough information is available
    to recommend assessments.
    
    Fix 1: Uses parser output instead of searching raw conversation.
    """
    skills = intent.get("skills", [])
    assessment_types = intent.get("assessment_types", [])
    job_level = intent.get("job_level")
    
    # No useful information at all
    if not skills and not assessment_types:
        return True
    
    # We have skills but no job level
    if skills and job_level is None:
        return True
    
    return False


def has_unknown_level_phrase(conversation):
    """
    Check if the user explicitly says they don't know or have no preference.
    
    Fix 2: Detects phrases that indicate the user wants to skip clarification.
    """
    conversation_lower = conversation.lower()
    
    unknown_phrases = [
        "not sure",
        "don't know",
        "do not know",
        "any level",
        "show me some options",
        "no preference",
        "i'm not sure",
        "not certain",
        "doesn't matter",
        "does not matter",
        "whatever",
        "either",
        "all levels",
        "any level is fine",
    ]
    
    return any(
        phrase in conversation_lower
        for phrase in unknown_phrases
    )


def build_reply(intent, recommendations):
    if not recommendations:
        return (
            "I couldn't find a suitable SHL assessment. "
            "Could you provide more details about the role?"
        )

    skills = intent.get("skills", [])

    if skills:
        return (
            f"I found {len(recommendations)} SHL assessments "
            f"that best match a role requiring "
            f"{', '.join(skills)}."
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
                "reply": (
                    "Sorry, I can only answer questions related to "
                    "SHL assessments and recommend assessments from "
                    "the SHL catalog."
                ),
                "recommendations": [],
                "end_of_conversation": True,
            }

        # --------------------------------------------------
        # Comparison
        # --------------------------------------------------
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
SHL Assessment Comparison

Assessment 1
-------------
Name: {a.get("name")}
Category: {", ".join(a.get("category", []))}
Duration: {a.get("duration_minutes")} minutes
Remote Testing: {a.get("remote")}
Adaptive: {a.get("adaptive")}

Assessment 2
-------------
Name: {b.get("name")}
Category: {", ".join(b.get("category", []))}
Duration: {b.get("duration_minutes")} minutes
Remote Testing: {b.get("remote")}
Adaptive: {b.get("adaptive")}
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
        # Intent Parsing
        # --------------------------------------------------
        intent = parser.parse(conversation)

        # Debug output
        print("=" * 60)
        print("Conversation:")
        print(conversation)
        print()
        print("Intent:")
        print(intent)
        print("=" * 60)

        # --------------------------------------------------
        # Clarification - with Fix 2 applied
        # --------------------------------------------------
        if needs_clarification(intent, conversation):
            skills = intent.get("skills", [])
            job_level = intent.get("job_level")
            
            # Check if user said they're not sure or have no preference
            unknown_level = has_unknown_level_phrase(conversation)
            
            if not skills:
                reply = (
                    "I'd be happy to help. "
                    "Could you tell me the primary skills or technologies "
                    "required for this role? "
                    "(For example: Java, Python, SQL, AWS, React.)"
                )
                return {
                    "reply": reply,
                    "recommendations": [],
                    "end_of_conversation": False,
                }
            
            # We have skills but no job level
            if skills and job_level is None:
                # Fix 2: Skip clarification if user says they're not sure
                if unknown_level:
                    # Continue without asking for job level
                    pass
                else:
                    # Ask for job level clarification
                    reply = (
                        "Thanks! Before I recommend assessments, "
                        "what is the seniority level for this role? "
                        "(Entry, Mid, or Senior)"
                    )
                    return {
                        "reply": reply,
                        "recommendations": [],
                        "end_of_conversation": False,
                    }
            
            # If we reach here, we have other missing information
            reply = (
                "Could you also specify whether you're looking for "
                "technical, personality, leadership, or cognitive assessments?"
            )
            return {
                "reply": reply,
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
