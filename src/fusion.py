from collections import defaultdict


class ReciprocalRankFusion:

    def __init__(self, k=60):
        self.k = k

    def fuse(self, ranked_lists):

        scores = defaultdict(float)
        assessments = {}

        for ranked in ranked_lists:

            for rank, result in enumerate(ranked):

                aid = result["assessment"]["assessment_id"]

                assessments[aid] = result["assessment"]

                scores[aid] += 1.0 / (self.k + rank + 1)

        results = []

        for aid, score in scores.items():

            results.append(
                {
                    "assessment": assessments[aid],
                    "score": score,
                    "source": "rrf",
                }
            )

        results.sort(
            key=lambda x: x["score"],
            reverse=True,
        )

        return results