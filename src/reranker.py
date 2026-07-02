from sentence_transformers import CrossEncoder
import numpy as np


class CrossEncoderReranker:

    def __init__(self):
        self.model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

    # --------------------------------------------------
    # Normalize Scores
    # --------------------------------------------------

    def _normalize(self, scores):

        scores = np.asarray(scores, dtype=float)

        if len(scores) == 0:
            return scores

        mn = scores.min()
        mx = scores.max()

        if mx - mn < 1e-8:
            return np.ones_like(scores)

        return (scores - mn) / (mx - mn)

    # --------------------------------------------------
    # Rerank
    # --------------------------------------------------

    def rerank(
        self,
        query,
        candidates,
        top_k=5,
    ):

        # ------------------------------------------
        # Remove duplicates
        # ------------------------------------------

        unique = {}

        for candidate in candidates:

            aid = candidate["assessment"]["assessment_id"]

            if (
                aid not in unique
                or candidate.get("score", 0)
                > unique[aid].get("score", 0)
            ):
                unique[aid] = candidate

        candidates = list(unique.values())

        if len(candidates) == 0:
            return []

        # ------------------------------------------
        # Build CrossEncoder input
        # ------------------------------------------

        pairs = []
        retrieval_scores = []

        for candidate in candidates:

            assessment = candidate["assessment"]

            text = f"""
Assessment:
{assessment.get("name","")}

Category:
{' '.join(assessment.get("category", []))}

Job Levels:
{' '.join(assessment.get("job_levels", []))}

Duration:
{assessment.get("duration_minutes")}

Description:
{assessment.get("description","")}
"""

            pairs.append((query, text))

            retrieval_scores.append(
                candidate.get(
                    "retrieval_score",
                    candidate.get("score", 0.0),
                )
            )

        # ------------------------------------------
        # CrossEncoder
        # ------------------------------------------

        ce_scores = self.model.predict(pairs)

        ce_norm = self._normalize(ce_scores)
        retrieval_norm = self._normalize(retrieval_scores)

        query_lower = query.lower()

        ranked = []

        # ------------------------------------------
        # Hybrid Ranking
        # ------------------------------------------

        for ce, rt, candidate in zip(
            ce_norm,
            retrieval_norm,
            candidates,
        ):

            assessment = candidate["assessment"]

            name = assessment.get(
                "name",
                "",
            )

            name_lower = name.lower()

            matched_skills = assessment.get(
                "matched_skills",
                [],
            )

            business_score = candidate.get(
                "business_score",
                0.0,
            )

            # ----------------------------------
            # Skill Bonus
            # ----------------------------------

            skill_bonus = min(
                0.05 * len(matched_skills),
                0.20,
            )

            # ----------------------------------
            # Exact Name Match Bonus
            # ----------------------------------

            phrase_bonus = 0.0

            for skill in matched_skills:
                if skill.lower() in name_lower:
                    phrase_bonus += 0.08

            # Explicit query phrases
            important_terms = [
                "spring",
                "python",
                "docker",
                "sql",
                "aws",
                "java",
                "react",
                "angular",
                "excel",
                "linux",
            ]

            for term in important_terms:
                if (
                    term in query_lower
                    and term in name_lower
                ):
                    phrase_bonus += 0.12

            phrase_bonus = min(
                phrase_bonus,
                0.30,
            )

            # ----------------------------------
            # Final Hybrid Score
            # ----------------------------------

            final_score = (
                0.55 * rt
                + 0.25 * ce
                + business_score
                + skill_bonus
                + phrase_bonus
            )

            candidate["retrieval_score"] = round(
                float(rt),
                4,
            )

            candidate["cross_encoder_score"] = round(
                float(ce),
                4,
            )

            candidate["business_score"] = round(
                float(business_score),
                4,
            )

            candidate["final_score"] = round(
                float(final_score),
                4,
            )

            ranked.append(candidate)

        ranked.sort(
            key=lambda x: x["final_score"],
            reverse=True,
        )

        return ranked[:top_k]