class RecommendationExplainer:

    def explain(self, assessment, intent, source):

        reasons = []

        text = (
            assessment.get("name", "")
            + " "
            + assessment.get("description", "")
        ).lower()

        # --------------------------------------------------
        # Skills
        # --------------------------------------------------

        matched = []

        if "matched_skills" in assessment:
            matched = assessment["matched_skills"]
        else:
            for skill in intent.get("skills", []):
                if skill.lower() in text:
                    matched.append(skill)

        if matched:
            reasons.append(
                "Matched skills: " +
                ", ".join(sorted(set(matched)))
            )

        # --------------------------------------------------
        # Assessment Types
        # --------------------------------------------------
        
        assessment_types = intent.get("assessment_types", [])
        if assessment_types:
            categories = " ".join(
                assessment.get("category", [])
            ).lower()
            
            matched_types = []
            for atype in assessment_types:
                atype_lower = atype.lower()
                # Check if assessment type appears in categories
                if atype_lower in categories:
                    matched_types.append(atype)
                # Also check for related terms
                elif atype_lower == "cognitive" and ("aptitude" in categories or "reasoning" in categories):
                    matched_types.append(atype)
                elif atype_lower == "personality" and ("behavior" in categories or "behaviour" in categories):
                    matched_types.append(atype)
                elif atype_lower == "technical" and ("knowledge" in categories or "skill" in categories):
                    matched_types.append(atype)
                elif atype_lower == "situational" and ("judgment" in categories or "judgement" in categories or "scenario" in categories):
                    matched_types.append(atype)
                elif atype_lower == "numerical" and ("numerical" in categories or "math" in categories or "quantitative" in categories):
                    matched_types.append(atype)
                elif atype_lower == "verbal" and ("verbal" in categories or "communication" in categories):
                    matched_types.append(atype)
            
            if matched_types:
                reasons.append(
                    "Matches assessment type: " +
                    ", ".join(sorted(set(matched_types)))
                )

        # --------------------------------------------------
        # Domain / Role Context
        # --------------------------------------------------

        categories = " ".join(
            assessment.get("category", [])
        ).lower()

        # Check domains from intent
        for domain in intent.get("domains", []):
            if domain.lower() in categories:
                reasons.append(
                    f"Relevant for {domain}"
                )

        # Check role context
        role_context = intent.get("role_context")
        if role_context:
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
            
            # Check if the assessment is relevant to the role context
            context_keywords = {
                "leadership": ["leadership", "executive", "director", "manager", "management"],
                "sales": ["sales", "selling", "retail", "commercial"],
                "contact_centre": ["customer service", "contact", "call center", "service"],
                "graduate": ["graduate", "entry", "junior"],
                "admin": ["administrative", "office", "clerical", "secretarial"],
                "plant_operator": ["manufacturing", "industrial", "plant", "safety", "mechanical"],
                "healthcare": ["healthcare", "medical", "hospital", "clinical"],
                "engineer": ["engineering", "technical", "developer", "programmer"]
            }
            
            keywords = context_keywords.get(role_context, [])
            for keyword in keywords:
                if keyword in categories or keyword in text:
                    reasons.append(
                        f"Relevant for {context_display} role"
                    )
                    break

        # --------------------------------------------------
        # Job Level
        # --------------------------------------------------

        job_level = intent.get("job_level")
        
        # Get assessment level from name (if not in job_levels field)
        assessment_levels = [
            level.lower()
            for level in assessment.get("job_levels", [])
        ]
        
        # If no job_levels field, try to detect from name
        if not assessment_levels:
            name = assessment.get("name", "").lower()
            if "entry" in name or "junior" in name or "beginner" in name:
                assessment_levels.append("entry")
            elif "advanced" in name or "senior" in name or "expert" in name:
                assessment_levels.append("senior")
            elif "mid" in name or "intermediate" in name:
                assessment_levels.append("mid")

        if job_level is not None and assessment_levels:
            # Check if the intent level matches any assessment level
            job_level_lower = job_level.lower()
            if job_level_lower in assessment_levels:
                level_display = {
                    "entry": "Entry-level",
                    "mid": "Mid-level",
                    "senior": "Senior-level"
                }.get(job_level_lower, job_level_lower.capitalize())
                reasons.append(
                    f"Suitable for {level_display} candidates"
                )
        elif assessment_levels and not job_level:
            # If no job level in intent but assessment has level, don't add anything
            # (this prevents showing "Mid" by default)
            pass

        # --------------------------------------------------
        # Remote
        # --------------------------------------------------

        if (
            intent.get("remote") is True
            and assessment.get("remote") is True
        ):
            reasons.append(
                "Remote testing supported"
            )

        # --------------------------------------------------
        # Adaptive
        # --------------------------------------------------

        if (
            intent.get("adaptive") is True
            and assessment.get("adaptive") is True
        ):
            reasons.append(
                "Adaptive assessment"
            )

        # --------------------------------------------------
        # Retrieval Source
        # --------------------------------------------------

        source_messages = {
            "semantic": "High semantic similarity",
            "bm25": "Strong keyword match",
            "graph": "Related through knowledge graph",
            "metadata": "Matched structured metadata",
            "rrf": "Recommended by hybrid retrieval",
        }

        if source in source_messages:
            reasons.append(source_messages[source])

        # --------------------------------------------------
        # Remove duplicates
        # --------------------------------------------------

        unique = []

        for reason in reasons:
            if reason not in unique:
                unique.append(reason)

        return unique
