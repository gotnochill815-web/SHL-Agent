class BusinessRanker:

    def rank(
        self,
        intent,
        candidates,
    ):
        """
        Final business-aware ranking layer.
        Runs AFTER cross encoder reranking.
        """

        for candidate in candidates:

            assessment = candidate["assessment"]

            score = candidate.get(
                "cross_encoder_score",
                candidate.get("rerank_score", 0.0),
            )

            bonus = 0.0

            ####################################################
            # Exact skill matches
            ####################################################

            matched = assessment.get(
                "matched_skills",
                [],
            )

            bonus += 0.25 * len(matched)

            ####################################################
            # Exact assessment name contains skill
            ####################################################

            name = assessment.get(
                "name",
                "",
            ).lower()

            for skill in intent.get(
                "skills",
                [],
            ):

                if skill.lower() in name:
                    bonus += 0.40

            ####################################################
            # Job level
            ####################################################

            if intent.get("job_level"):

                levels = " ".join(
                    assessment.get(
                        "job_levels",
                        [],
                    )
                ).lower()

                if intent["job_level"].lower() in levels:
                    bonus += 0.15

            ####################################################
            # Duration
            ####################################################

            if intent.get("duration") is not None:

                duration = assessment.get(
                    "duration_minutes"
                )

                if (
                    duration is not None
                    and duration <= intent["duration"]
                ):
                    bonus += 0.10

            ####################################################
            # Remote
            ####################################################

            if (
                intent.get("remote") is True
                and assessment.get("remote")
            ):
                bonus += 0.05

            ####################################################
            # Adaptive
            ####################################################

            if (
                intent.get("adaptive") is True
                and assessment.get("adaptive")
            ):
                bonus += 0.05

            ####################################################
            # Final score
            ####################################################

            candidate["business_score"] = bonus

            candidate["final_score"] = (
                score + bonus
            )

        candidates.sort(
            key=lambda x: x["final_score"],
            reverse=True,
        )

        return candidates