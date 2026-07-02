import re
from typing import Dict, Any, List, Optional

class IntentParser:
    def __init__(self):
        # --------------------------------------------------
        # Skill Dictionary
        # --------------------------------------------------
        self.skills = {
            "python": ["python", "django", "flask", "fastapi"],
            "java": ["java", "core java", "java 8", "java ee", "j2ee"],
            "spring": ["spring", "spring boot", "spring framework"],
            "sql": ["sql", "mysql", "postgres", "postgresql", "oracle sql"],
            "aws": ["aws", "amazon web services", "ec2", "s3"],
            "docker": ["docker", "containers", "containerization"],
            "kubernetes": ["kubernetes", "k8s"],
            "jenkins": ["jenkins"],
            "azure": ["azure", "microsoft azure"],
            "gcp": ["gcp", "google cloud"],
            "react": ["react", "reactjs", "react.js"],
            "angular": ["angular", "angularjs"],
            "node": ["node", "nodejs", "node.js"],
            "javascript": ["javascript", "ecmascript"],
            "typescript": ["typescript"],
            "mongodb": ["mongodb", "mongo"],
            "linux": ["linux", "unix", "ubuntu"],
            "networking": ["network", "networking", "tcp", "udp"],
            "selenium": ["selenium"],
            "cypress": ["cypress"],
            "tensorflow": ["tensorflow"],
            "pytorch": ["pytorch"],
            "hipaa": ["hipaa"],
            "excel": ["excel", "microsoft excel"],
            "word": ["word", "microsoft word"],
            "statistics": ["statistics"],
            "finance": ["financial", "accounting"],
            "leadership": ["leadership", "executive", "director", "cxo"],
            "go": ["go", "golang"],
            "ruby": ["ruby", "ruby on rails", "rails"],
            "scala": ["scala"],
            "kotlin": ["kotlin"],
            "rust": ["rust"],
            "cplusplus": ["c++", "cpp"],
            "csharp": ["c#", "csharp"],
            "html": ["html"],
            "css": ["css"],
            "git": ["git"],
            "jira": ["jira"],
            "rest": ["rest", "restful", "rest api"],
            "powerpoint": ["powerpoint", "microsoft powerpoint"],
            "outlook": ["outlook", "microsoft outlook"],
            "access": ["access", "microsoft access"],
        }

        # --------------------------------------------------
        # Domain Dictionary (expanded with more aliases)
        # --------------------------------------------------
        self.domain_dictionary = {
            "finance": ["finance", "financial", "accounting", "banking", "investment"],
            "healthcare": ["healthcare", "medical", "hospital", "hipaa", "patient", "health"],
            "manufacturing": ["manufacturing", "industrial", "plant", "factory", "chemical", "facility"],
            "customer service": [
                "customer service", "contact centre", "contact center", 
                "call center", "call centre", "contact centre agents",
                "customer serv", "inbound calls", "customer support"
            ],
            "sales": ["sales", "selling", "retail", "re-skill", "sales organization", "sales team"],
            "software": ["software", "developer", "engineer", "programmer", "backend", "frontend", "full stack"],
            "safety": ["safety", "compliance", "hazard", "dependability", "reliability", "safe"],
            "leadership": ["leadership", "executive", "director", "cxo", "management", "senior leadership"],
            "administration": ["admin", "administrative", "admin assistant", "office", "clerical"],
            "graduate": ["graduate", "final year", "recent graduate", "trainee", "internship"],
        }

        # --------------------------------------------------
        # Assessment Types
        # --------------------------------------------------
        self.assessment_type_dictionary = {
            "coding": ["coding", "programming", "developer", "software engineer", "backend", "frontend", "full stack"],
            "technical": ["technical", "knowledge", "stack"],
            "personality": ["personality", "opq", "behavior", "behaviour"],
            "leadership": ["leadership", "manager", "executive"],
            "simulation": ["simulation", "simulator"],
            "numerical": ["numerical", "math", "statistics", "quantitative"],
            "verbal": ["verbal", "english", "communication"],
            "cognitive": ["cognitive", "aptitude", "reasoning", "general ability"],
            "situational": ["situational", "judgement", "scenario", "biodata"],
            "knowledge": ["knowledge", "domain knowledge"],
            "skills": ["skills", "competency", "competencies"],
        }

        # --------------------------------------------------
        # Languages
        # --------------------------------------------------
        self.languages = ["english", "spanish", "french", "german", "latin american spanish"]

        # --------------------------------------------------
        # Job Levels (mapped to standardized values)
        # --------------------------------------------------
        self.job_levels = {
            "entry": [
                "graduate", "entry level", "entry", "fresher", "intern", "internship",
                "junior", "associate", "beginner", "trainee", "final year", "recent graduate"
            ],
            "mid": [
                "mid", "mid level", "intermediate", "medium", "regular", "professional"
            ],
            "senior": [
                "senior", "staff", "lead", "principal", "senior level", "expert",
                "advanced", "manager", "engineering manager", "director", "head",
                "executive", "vp", "vice president", "cxo", "senior leadership"
            ],
        }

        # --------------------------------------------------
        # Experience to Level Mapping (years of experience -> seniority)
        # --------------------------------------------------
        self.experience_to_level = [
            (0, 2, "entry"),
            (3, 6, "mid"),
            (7, 100, "senior")
        ]

        # --------------------------------------------------
        # Role/Context Detection (for queries without explicit skills)
        # --------------------------------------------------
        self.role_contexts = {
            "leadership": ["senior leadership", "executive", "cxo", "director", "management"],
            "sales": ["sales", "sales organization", "sales team", "selling"],
            "contact_centre": ["contact centre", "contact center", "call center", "call centre", "customer service"],
            "graduate": ["graduate scheme", "graduate trainee", "management trainee", "recent graduates"],
            "admin": ["admin assistant", "administrative", "office staff"],
            "plant_operator": ["plant operator", "chemical facility", "manufacturing"],
            "healthcare": ["healthcare", "medical", "hospital", "patient records"],
            "engineer": ["engineer", "developer", "programmer", "full-stack", "backend", "frontend"],
        }

    def parse(self, query: str) -> Dict[str, Any]:
        q = query.lower()

        intent = {
            "skills": [],
            "domains": [],
            "assessment_types": [],
            "job_level": None,
            "experience": None,
            "duration": None,
            "languages": [],
            "remote": None,
            "adaptive": None,
            "keywords": [],
            "role_context": None,
            "has_skills_or_domain": False,
        }

        # --------------------------------------------------
        # Skills (exact word boundary matching)
        # --------------------------------------------------
        for skill, aliases in self.skills.items():
            for alias in aliases:
                pattern = r"\b" + re.escape(alias.lower()) + r"\b"
                if re.search(pattern, q):
                    intent["skills"].append(skill)
                    break

        # --------------------------------------------------
        # Domains (exact word boundary matching)
        # --------------------------------------------------
        for domain, aliases in self.domain_dictionary.items():
            for alias in aliases:
                pattern = r"\b" + re.escape(alias.lower()) + r"\b"
                if re.search(pattern, q):
                    intent["domains"].append(domain)
                    break

        # --------------------------------------------------
        # Domains FALLBACK (simple substring matching for multi-word phrases)
        # --------------------------------------------------
        if not intent["domains"]:
            for domain, aliases in self.domain_dictionary.items():
                for alias in aliases:
                    if alias in q:
                        intent["domains"].append(domain)
                        break
                if intent["domains"]:
                    break

        # --------------------------------------------------
        # Assessment Types
        # --------------------------------------------------
        for assessment_type, aliases in self.assessment_type_dictionary.items():
            for alias in aliases:
                pattern = r"\b" + re.escape(alias.lower()) + r"\b"
                if re.search(pattern, q):
                    intent["assessment_types"].append(assessment_type)
                    break

        # --------------------------------------------------
        # Assessment Types FALLBACK (simple substring matching)
        # --------------------------------------------------
        if not intent["assessment_types"]:
            for assessment_type, aliases in self.assessment_type_dictionary.items():
                for alias in aliases:
                    if alias in q:
                        intent["assessment_types"].append(assessment_type)
                        break
                if intent["assessment_types"]:
                    break

        # --------------------------------------------------
        # Job Level (from explicit mentions)
        # --------------------------------------------------
        for level, aliases in self.job_levels.items():
            for alias in aliases:
                pattern = r"\b" + re.escape(alias.lower()) + r"\b"
                if re.search(pattern, q):
                    intent["job_level"] = level
                    break
            if intent["job_level"] is not None:
                break

        # --------------------------------------------------
        # Job Level FALLBACK (simple substring matching)
        # --------------------------------------------------
        if intent["job_level"] is None:
            for level, aliases in self.job_levels.items():
                for alias in aliases:
                    if alias in q:
                        intent["job_level"] = level
                        break
                if intent["job_level"] is not None:
                    break

        # --------------------------------------------------
        # Experience (extract years of experience)
        # --------------------------------------------------
        years_match = re.search(
            r"(\d+)\s*(?:\+)?\s*(?:years?|yrs?|year|yr)",
            q,
        )

        if years_match:
            intent["experience"] = int(years_match.group(1))
            
            # If no job level explicitly mentioned, derive from experience
            if intent["job_level"] is None and intent["experience"] is not None:
                exp = intent["experience"]
                for low, high, level in self.experience_to_level:
                    if low <= exp <= high:
                        intent["job_level"] = level
                        break

        # --------------------------------------------------
        # Role Context Detection
        # --------------------------------------------------
        for context, patterns in self.role_contexts.items():
            for pattern in patterns:
                if pattern in q:
                    intent["role_context"] = context
                    break
            if intent["role_context"] is not None:
                break

        # --------------------------------------------------
        # Special Case: Leadership with no other context
        # --------------------------------------------------
        if "leadership" in q and not intent["skills"] and not intent["domains"]:
            intent["domains"].append("leadership")
            intent["role_context"] = "leadership"

        # --------------------------------------------------
        # Special Case: Contact Centre detection
        # --------------------------------------------------
        contact_centre_keywords = ["contact centre", "contact center", "call center", "call centre"]
        for keyword in contact_centre_keywords:
            if keyword in q:
                if "customer service" not in intent["domains"]:
                    intent["domains"].append("customer service")
                intent["role_context"] = "contact_centre"
                break

        # --------------------------------------------------
        # Special Case: Graduate detection
        # --------------------------------------------------
        graduate_keywords = ["graduate", "final year", "trainee", "internship"]
        for keyword in graduate_keywords:
            if keyword in q and "graduate" not in intent["domains"]:
                intent["domains"].append("graduate")
                if intent["role_context"] is None:
                    intent["role_context"] = "graduate"
                break

        # --------------------------------------------------
        # Duration (extract maximum duration in minutes)
        # --------------------------------------------------
        duration_match = re.search(
            r"(?:under|less than)?\s*(\d+)\s*(?:minutes?|mins?|min)",
            q,
        )

        if duration_match:
            intent["duration"] = int(duration_match.group(1))

        # --------------------------------------------------
        # Languages
        # --------------------------------------------------
        for language in self.languages:
            if language in q:
                intent["languages"].append(language)

        # --------------------------------------------------
        # Remote / Adaptive
        # --------------------------------------------------
        remote_keywords = ["remote", "work from home", "wfh"]
        onsite_keywords = ["onsite", "in office"]
        
        if any(word in q for word in remote_keywords):
            intent["remote"] = True
        elif any(word in q for word in onsite_keywords):
            intent["remote"] = False
        elif "hybrid" in q:
            intent["remote"] = None

        if "adaptive" in q:
            intent["adaptive"] = True

        # --------------------------------------------------
        # Keywords
        # --------------------------------------------------
        intent["keywords"] = re.findall(
            r"[A-Za-z0-9+#.-]+",
            query,
        )

        # --------------------------------------------------
        # Check if we have any skills or domains or role context
        # --------------------------------------------------
        intent["has_skills_or_domain"] = (
            len(intent["skills"]) > 0 or 
            len(intent["domains"]) > 0 or 
            intent["role_context"] is not None
        )

        # --------------------------------------------------
        # Special: Leadership roles should NOT be treated as skills
        # that trigger immediate recommendations without context
        # --------------------------------------------------
        if intent["role_context"] == "leadership" and len(intent["skills"]) == 1 and "leadership" in intent["skills"]:
            # Keep leadership as a domain but don't treat it as a skill for retrieval
            # This ensures the system asks clarifying questions first
            if "leadership" in intent["skills"]:
                intent["skills"].remove("leadership")

        # --------------------------------------------------
        # Remove Duplicates
        # --------------------------------------------------
        intent["skills"] = sorted(set(intent["skills"]))
        intent["domains"] = sorted(set(intent["domains"]))
        intent["assessment_types"] = sorted(set(intent["assessment_types"]))
        intent["languages"] = sorted(set(intent["languages"]))

        return intent
