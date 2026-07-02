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

        # Prefer matched skills produced by RecommendationRules
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
        # Domain
        # --------------------------------------------------

        categories = " ".join(
            assessment.get("category", [])
        ).lower()

        for domain in intent.get("domains", []):

            if domain.lower() in categories:
                reasons.append(
                    f"Relevant for {domain}"
                )

        # --------------------------------------------------
        # Job Level
        # --------------------------------------------------

        if intent.get("job_level"):

            levels = " ".join(
                assessment.get("job_levels", [])
            ).lower()

            if intent["job_level"].lower() in levels:

                reasons.append(
                    f"Suitable for {intent['job_level']} candidates"
                )

        # --------------------------------------------------
        # Remote
        # --------------------------------------------------

        if (
            intent.get("remote") is True
            and assessment.get("remote")
        ):
            reasons.append(
                "Remote testing supported"
            )

        # --------------------------------------------------
        # Adaptive
        # --------------------------------------------------

        if (
            intent.get("adaptive") is True
            and assessment.get("adaptive")
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