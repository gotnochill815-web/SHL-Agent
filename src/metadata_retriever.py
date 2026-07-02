class MetadataRetriever:

    def __init__(self, catalog):
        self.catalog = catalog

    def retrieve(self, intent, top_k=20):

        results = []

        for assessment in self.catalog:

            score = 0.0

            text = (
                assessment.get("name", "")
                + " "
                + assessment.get("description", "")
            ).lower()

            matched = []

            # -----------------------------------
            # Skill Matching
            # -----------------------------------

            for skill in intent.get("skills", []):

                if skill.lower() in text:
                    score += 4
                    matched.append(skill)

            # -----------------------------------
            # Assessment Type
            # -----------------------------------

            categories = " ".join(
                assessment.get("category", [])
            ).lower()

            for assessment_type in intent.get(
                "assessment_types",
                [],
            ):

                if assessment_type.lower() in categories:
                    score += 3

            # -----------------------------------
            # Job Level
            # -----------------------------------

            levels = " ".join(
                assessment.get("job_levels", [])
            ).lower()

            if (
                intent.get("job_level")
                and intent["job_level"].lower() in levels
            ):
                score += 2

            # -----------------------------------
            # Language
            # -----------------------------------

            langs = " ".join(
                assessment.get("languages", [])
            ).lower()

            for language in intent.get(
                "languages",
                [],
            ):

                if language.lower() in langs:
                    score += 2

            # -----------------------------------
            # Remote
            # -----------------------------------

            if (
                intent.get("remote") is True
                and assessment.get("remote")
            ):
                score += 1

            # -----------------------------------
            # Adaptive
            # -----------------------------------

            if (
                intent.get("adaptive") is True
                and assessment.get("adaptive")
            ):
                score += 1

            if score > 0:

                assessment["matched_skills"] = matched

                results.append(
                    {
                        "assessment": assessment,
                        "score": score,
                        "source": "metadata",
                    }
                )

        results.sort(
            key=lambda x: x["score"],
            reverse=True,
        )

        return results[:top_k]