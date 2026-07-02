import re


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
            "javascript": [
                "javascript",
                "ecmascript"
            ],
            "typescript": [
                "typescript"
            ],
            "mongodb": ["mongodb", "mongo"],
            "linux": ["linux", "unix", "ubuntu"],
            "networking": ["network", "networking", "tcp", "udp"],
            "selenium": ["selenium"],
            "cypress": ["cypress"],
            "tensorflow": ["tensorflow"],
            "pytorch": ["pytorch"],
            "hipaa": ["hipaa"],
            "excel": ["excel"],
            "word": ["word"],
            "statistics": ["statistics"],
            "finance": ["financial", "accounting"],
            "leadership": ["leadership", "executive", "director", "cxo"],
        }

        # --------------------------------------------------
        # Domain Dictionary
        # --------------------------------------------------
        self.domain_dictionary = {
            "finance": [
                "finance",
                "financial",
                "accounting",
                "banking",
            ],
            "healthcare": [
                "healthcare",
                "medical",
                "hospital",
                "hipaa",
            ],
            "manufacturing": [
                "manufacturing",
                "industrial",
                "plant",
                "factory",
                "chemical",
            ],
            "customer service": [
                "customer service",
                "contact centre",
                "contact center",
                "call center",
                "call centre",
            ],
            "sales": [
                "sales",
                "selling",
                "retail",
            ],
            "software": [
                "software",
                "developer",
                "engineer",
                "programmer",
                "backend",
                "frontend",
                "full stack",
            ],
        }

        # --------------------------------------------------
        # Assessment Types
        # --------------------------------------------------
        self.assessment_type_dictionary = {
            "coding": [
                "coding",
                "programming",
                "developer",
                "software engineer",
                "backend",
                "frontend",
                "full stack",
            ],
            "technical": [
                "technical",
                "knowledge",
            ],
            "personality": [
                "personality",
                "opq",
                "behavior",
                "behaviour",
            ],
            "leadership": [
                "leadership",
                "manager",
                "executive",
            ],
            "simulation": [
                "simulation",
                "simulator",
            ],
            "numerical": [
                "numerical",
                "math",
                "statistics",
            ],
            "verbal": [
                "verbal",
                "english",
                "communication",
            ],
        }

        # --------------------------------------------------
        # Languages
        # --------------------------------------------------
        self.languages = [
            "english",
            "spanish",
            "french",
            "german",
        ]

        # --------------------------------------------------
        # Job Levels
        # --------------------------------------------------
        self.job_levels = {
            "Graduate": [
                "graduate",
                "entry level",
                "fresher",
                "intern",
                "internship",
            ],
            "Junior": [
                "junior",
                "associate",
            ],
            "Mid": [
                "mid",
                "mid level",
            ],
            "Senior": [
                "senior",
                "staff",
            ],
            "Lead": [
                "lead",
                "principal",
            ],
            "Manager": [
                "manager",
                "engineering manager",
            ],
            "Director": [
                "director",
                "head",
            ],
            "Executive": [
                "executive",
                "vp",
                "vice president",
                "cxo",
            ],
        }

    def parse(self, query):
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
        }

        # --------------------------------------------------
        # Skills
        # --------------------------------------------------
        for skill, aliases in self.skills.items():
            for alias in aliases:
                # Exact word matching
                pattern = r"\b" + re.escape(alias.lower()) + r"\b"
                if re.search(pattern, q):
                    intent["skills"].append(skill)
                    break

        # --------------------------------------------------
        # Domains
        # --------------------------------------------------
        for domain, aliases in self.domain_dictionary.items():
            for alias in aliases:
                # Exact word matching
                pattern = r"\b" + re.escape(alias.lower()) + r"\b"
                if re.search(pattern, q):
                    intent["domains"].append(domain)
                    break

        # --------------------------------------------------
        # Assessment Types
        # --------------------------------------------------
        for assessment_type, aliases in self.assessment_type_dictionary.items():
            for alias in aliases:
                # Exact word matching
                pattern = r"\b" + re.escape(alias.lower()) + r"\b"
                if re.search(pattern, q):
                    intent["assessment_types"].append(assessment_type)
                    break

        # --------------------------------------------------
        # Job Level
        # --------------------------------------------------
        for level, aliases in self.job_levels.items():
            for alias in aliases:
                # Exact word matching
                pattern = r"\b" + re.escape(alias.lower()) + r"\b"
                if re.search(pattern, q):
                    intent["job_level"] = level
                    break

            if intent["job_level"] is not None:
                break

        # --------------------------------------------------
        # Experience
        # --------------------------------------------------
        years = re.search(
            r"(\d+)\s*(?:\+)?\s*(?:years?|yrs?|year|yr)",
            q,
        )

        if years:
            intent["experience"] = int(years.group(1))

        # --------------------------------------------------
        # Duration
        # --------------------------------------------------
        duration = re.search(
            r"(?:under|less than)?\s*(\d+)\s*(?:minutes?|mins?|min)",
            q,
        )

        if duration:
            intent["duration"] = int(duration.group(1))

        # --------------------------------------------------
        # Languages
        # --------------------------------------------------
        for language in self.languages:
            if language in q:
                intent["languages"].append(language)

        # --------------------------------------------------
        # Remote / Adaptive
        # --------------------------------------------------
        if any(
            word in q
            for word in [
                "remote",
                "work from home",
                "wfh",
            ]
        ):
            intent["remote"] = True
        elif "onsite" in q:
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
        # Remove Duplicates
        # --------------------------------------------------
        intent["skills"] = sorted(set(intent["skills"]))
        intent["domains"] = sorted(set(intent["domains"]))
        intent["assessment_types"] = sorted(set(intent["assessment_types"]))
        intent["languages"] = sorted(set(intent["languages"]))

        return intent
