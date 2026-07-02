class RecommendationRules:

    def apply(self, intent, results):

        skills = set(intent.get("skills", []))
        domains = set(intent.get("domains", []))

        boosted = []

        for result in results:

            assessment = result["assessment"]
            score = result["score"]

            name = assessment["name"].lower()

            # -----------------------------------
            # Java
            # -----------------------------------

            if "java" in skills:

                if "java" in name:
                    score += 0.35

            # -----------------------------------
            # Spring
            # -----------------------------------

            if "spring" in skills:

                if "spring" in name:
                    score += 0.30

            # -----------------------------------
            # SQL
            # -----------------------------------

            if "sql" in skills:

                if "sql" in name:
                    score += 0.25

            # -----------------------------------
            # AWS
            # -----------------------------------

            if "aws" in skills:

                if "aws" in name:
                    score += 0.25

            # -----------------------------------
            # Docker
            # -----------------------------------

            if "docker" in skills:

                if "docker" in name:
                    score += 0.25

            # -----------------------------------
            # Leadership
            # -----------------------------------

            if "leadership" in skills:

                if "leadership" in name:
                    score += 0.40

            # -----------------------------------
            # Healthcare
            # -----------------------------------

            if "healthcare" in domains:

                if (
                    "medical" in name
                    or "hipaa" in name
                ):
                    score += 0.40

            boosted.append(
                {
                    "assessment": assessment,
                    "score": score,
                }
            )

        boosted.sort(
            key=lambda x: x["score"],
            reverse=True,
        )

        return boosted