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
    """
    skills = intent.get("skills", [])
    assessment_types = intent.get("assessment_types", [])
    job_level = intent.get("job_level")
    has_skills_or_domain = intent.get("has_skills_or_domain", False)
    role_context = intent.get("role_context")
    
    # No useful information at all - no skills, no domains, no role context
    if not skills and not assessment_types and not has_skills_or_domain and not role_context:
        return True
    
    # We have skills/context but no job level
    if (skills or has_skills_or_domain or role_context) and job_level is None:
        return True
    
    return False


def has_unknown_level_phrase(conversation):
    """
    Check if the user explicitly says they don't know or have no preference.
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
    job_level = intent.get("job_level")
    assessment_types = intent.get("assessment_types", [])
    role_context = intent.get("role_context")

    if skills and job_level:
        level_display = {
            "entry": "Entry-level",
            "mid": "Mid-level", 
            "senior": "Senior-level"
        }.get(job_level, job_level.capitalize())
        
        if assessment_types:
            type_text = " and ".join(assessment_types)
            return (
                f"I found {len(recommendations)} SHL assessments "
                f"that best match a {level_display} role requiring "
                f"{', '.join(skills)} with {type_text} assessments."
            )
        
        return (
            f"I found {len(recommendations)} SHL assessments "
            f"that best match a {level_display} role requiring "
            f"{', '.join(skills)}."
        )
    elif skills:
        if assessment_types:
            type_text = " and ".join(assessment_types)
            return (
                f"I found {len(recommendations)} SHL assessments "
                f"that best match a role requiring "
                f"{', '.join(skills)} with {type_text} assessments."
            )
        
        return (
            f"I found {len(recommendations)} SHL assessments "
            f"that best match a role requiring "
            f"{', '.join(skills)}."
        )
    elif role_context:
        context_display = {
            "leadership": "leadership",
            "sales": "sales",
            "contact_centre": "customer service",
            "graduate": "graduate",
            "admin": "administrative",
            "plant_operator": "plant operator",
            "healthcare": "healthcare",
            "engineer": "engineering"
        }.get(role_context, role_context)
        
        if job_level:
            level_display = {
                "entry": "Entry-level",
                "mid": "Mid-level",
                "senior": "Senior-level"
            }.get(job_level, job_level.capitalize())
            return (
                f"I found {len(recommendations)} SHL assessments "
                f"that best match a {level_display} {context_display} role."
            )
        
        return (
            f"I found {len(recommendations)} SHL assessments "
            f"that best match a {context_display} role."
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


def get_assessment_level(assessment):
    """
    Extract the actual level from the assessment name or metadata.
    Returns None if no level can be determined.
    """
    name = assessment.get("name", "").lower()
    
    if "entry" in name or "junior" in name or "beginner" in name:
        return "Entry"
    elif "advanced" in name or "senior" in name or "expert" in name:
        return "Senior"
    elif "mid" in name or "intermediate" in name:
        return "Mid"
    
    # Check metadata if available
    level = assessment.get("level")
    if level:
        return level.capitalize()
    
    return None


# ============================================================
# Filter Functions
# ============================================================

def filter_by_assessment_types(candidates, assessment_types):
    """
    Filter candidates by assessment type.
    """
    if not assessment_types:
        return candidates
    
    filtered = []
    for item in candidates:
        assessment = item if isinstance(item, dict) else item.get("assessment", {})
        categories = assessment.get("category", [])
        categories_lower = [c.lower() for c in categories]
        
        for atype in assessment_types:
            atype_lower = atype.lower()
            if any(atype_lower in cat or cat in atype_lower for cat in categories_lower):
                filtered.append(item)
                break
    
    # If filtering returns too few results, return all candidates (graceful degradation)
    if len(filtered) < 3:
        return candidates
    
    return filtered


def filter_by_duration(candidates, max_duration):
    """
    Filter candidates by maximum duration in minutes.
    """
    if not max_duration:
        return candidates
    
    filtered = []
    for item in candidates:
        assessment = item if isinstance(item, dict) else item.get("assessment", {})
        duration = assessment.get("duration_minutes")
        if duration is not None and duration <= max_duration:
            filtered.append(item)
    
    # If filtering returns too few results, return all candidates (graceful degradation)
    if len(filtered) < 3:
        return candidates
    
    return filtered


def filter_by_remote(candidates, remote_preference):
    """
    Filter candidates by remote testing preference.
    remote_preference: True (remote), False (onsite), None (no preference)
    """
    if remote_preference is None:
        return candidates
    
    filtered = []
    for item in candidates:
        assessment = item if isinstance(item, dict) else item.get("assessment", {})
        remote = assessment.get("remote")
        if remote == remote_preference:
            filtered.append(item)
    
    # If filtering returns too few results, return all candidates (graceful degradation)
    if len(filtered) < 3:
        return candidates
    
    return filtered


def filter_by_adaptive(candidates, adaptive_preference):
    """
    Filter candidates by adaptive preference.
    adaptive_preference: True (adaptive), False (non-adaptive), None (no preference)
    """
    if adaptive_preference is None:
        return candidates
    
    filtered = []
    for item in candidates:
        assessment = item if isinstance(item, dict) else item.get("assessment", {})
        adaptive = assessment.get("adaptive")
        if adaptive == adaptive_preference:
            filtered.append(item)
    
    # If filtering returns too few results, return all candidates (graceful degradation)
    if len(filtered) < 3:
        return candidates
    
    return filtered


def is_turn_limit_reached(messages, max_turns=8):
    """
    Check if the conversation has reached the turn limit.
    """
    return len(messages) >= max_turns


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
        "general hiring advice",
        "resume tips",
        "interview tips",
        "career advice",
    ]
    return any(word in q for word in blocked)


def normalize_job_level(level: str):
    """
    Normalize various job level descriptions to standard values.
    """
    if not level:
        return None
    
    level_lower = level.lower()
    
    entry_keywords = ["entry", "junior", "beginner", "fresher", "graduate"]
    mid_keywords = ["mid", "intermediate", "medium", "regular"]
    senior_keywords = ["senior", "advanced", "expert", "lead", "principal", "staff"]
    
    if any(kw in level_lower for kw in entry_keywords):
        return "entry"
    elif any(kw in level_lower for kw in mid_keywords):
        return "mid"
    elif any(kw in level_lower for kw in senior_keywords):
        return "senior"
    
    return None


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
        # Turn Limit Check
        # --------------------------------------------------
        if is_turn_limit_reached(request.messages, max_turns=8):
            return {
                "reply": "I've reached the maximum number of turns for this conversation. Please start a new conversation or provide all requirements upfront.",
                "recommendations": [],
                "end_of_conversation": True,
            }

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
                    "the SHL catalog. I cannot provide advice on "
                    "salary, hiring, legal, medical, financial, "
                    "or career-related matters."
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
                    level_a = get_assessment_level(a)
                    level_b = get_assessment_level(b)
                    
                    comparison = f"""
SHL Assessment Comparison

Assessment 1
-------------
Name: {a.get("name")}
Category: {", ".join(a.get("category", []))}
Duration: {a.get("duration_minutes")} minutes
Remote Testing: {a.get("remote")}
Adaptive: {a.get("adaptive")}
Level: {level_a if level_a else "Not specified"}

Assessment 2
-------------
Name: {b.get("name")}
Category: {", ".join(b.get("category", []))}
Duration: {b.get("duration_minutes")} minutes
Remote Testing: {b.get("remote")}
Adaptive: {b.get("adaptive")}
Level: {level_b if level_b else "Not specified"}
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
        
        # Normalize job level if present
        raw_level = intent.get("job_level")
        if raw_level:
            intent["job_level"] = normalize_job_level(raw_level)

        # Debug output
        print("=" * 60)
        print("Conversation:")
        print(conversation)
        print()
        print("Intent:")
        print(intent)
        print("=" * 60)

        # --------------------------------------------------
        # Clarification
        # --------------------------------------------------
        if needs_clarification(intent, conversation):
            skills = intent.get("skills", [])
            job_level = intent.get("job_level")
            role_context = intent.get("role_context")
            has_skills_or_domain = intent.get("has_skills_or_domain", False)
            
            # Check if user said they're not sure or have no preference
            unknown_level = has_unknown_level_phrase(conversation)
            
            # No skills, domains, or role context detected
            if not skills and not has_skills_or_domain and not role_context:
                # Check if there's any hint of what they're looking for
                if "leadership" in conversation.lower() or "executive" in conversation.lower():
                    reply = (
                        "Happy to help narrow that down. Who is this meant for?"
                    )
                elif "sales" in conversation.lower():
                    reply = (
                        "For a sales organization, I can recommend several solutions. "
                        "Could you tell me more about the specific roles and what you're trying to assess?"
                    )
                elif "contact" in conversation.lower() or "customer service" in conversation.lower():
                    reply = (
                        "For contact centre roles, language is an important consideration. "
                        "What language will the calls be in?"
                    )
                elif "healthcare" in conversation.lower() or "medical" in conversation.lower():
                    reply = (
                        "For healthcare roles, there are some important considerations around language "
                        "and regulatory requirements. Could you tell me more about the specific roles?"
                    )
                elif "graduate" in conversation.lower() or "trainee" in conversation.lower():
                    reply = (
                        "For graduate roles, I can recommend a full battery including cognitive, "
                        "personality, and situational judgement. What specific areas are you looking to assess?"
                    )
                elif "safety" in conversation.lower() or "plant" in conversation.lower():
                    reply = (
                        "For safety-critical roles, I can recommend assessments focused on dependability "
                        "and safety compliance. Could you tell me more about the specific role and context?"
                    )
                else:
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
            
            # We have skills/context but no job level
            if (skills or has_skills_or_domain or role_context) and job_level is None:
                if unknown_level:
                    return {
                        "reply": (
                            "No problem. The seniority level helps me recommend the "
                            "most appropriate assessments. If you're unsure, you can "
                            "tell me the approximate years of experience or simply "
                            "choose Entry, Mid, or Senior."
                        ),
                        "recommendations": [],
                        "end_of_conversation": False,
                    }
                
                return {
                    "reply": (
                        "Thanks! Before I recommend assessments, "
                        "what is the seniority level for this role? "
                        "(Entry, Mid, or Senior)"
                    ),
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
        # Apply Filters
        # --------------------------------------------------
        
        # Filter by assessment type (e.g., "personality", "cognitive")
        assessment_types = intent.get("assessment_types", [])
        if assessment_types:
            candidates = filter_by_assessment_types(candidates, assessment_types)
            print(f"Filtered by assessment types: {assessment_types}, remaining: {len(candidates)}")
        
        # Filter by duration (e.g., "under 20 minutes")
        duration = intent.get("duration")
        if duration:
            candidates = filter_by_duration(candidates, duration)
            print(f"Filtered by duration: {duration} minutes, remaining: {len(candidates)}")
        
        # Filter by remote preference
        remote = intent.get("remote")
        if remote is not None:
            candidates = filter_by_remote(candidates, remote)
            print(f"Filtered by remote: {remote}, remaining: {len(candidates)}")
        
        # Filter by adaptive preference
        adaptive = intent.get("adaptive")
        if adaptive is not None:
            candidates = filter_by_adaptive(candidates, adaptive)
            print(f"Filtered by adaptive: {adaptive}, remaining: {len(candidates)}")

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
            
            # Get the actual assessment level
            assessment_level = get_assessment_level(assessment)
            
            # Get reasons with correct level
            reasons = explainer.explain(
                assessment,
                intent,
                item.get("source", "rrf"),
            )
            
            # Override the level in reasons if assessment has a known level
            if assessment_level:
                # Remove any existing "Suitable for X candidates" reason
                reasons = [r for r in reasons if not r.startswith("Suitable for")]
                reasons.append(f"Suitable for {assessment_level} candidates")

            recommendations.append({
                "name": assessment.get("name"),
                "url": assessment.get("url"),
                "test_type": get_test_type(assessment),
                "category": assessment.get("category", []),
                "duration": assessment.get("duration_minutes"),
                "reason": reasons,
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
