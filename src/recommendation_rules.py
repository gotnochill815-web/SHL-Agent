class RecommendationRules:

    def __init__(self):

        self.skill_mapping = {

            "java": [
                "java",
                "core java",
                "java ee",
                "j2ee",
            ],

            "spring": [
                "spring",
                "spring boot",
                "java frameworks",
            ],

            "sql": [
                "sql",
                "postgres",
                "mysql",
                "oracle",
                "database",
            ],

            "aws": [
                "aws",
                "amazon web services",
                "cloud",
            ],

            "docker": [
                "docker",
                "container",
                "kubernetes",
            ],

            "python": [
                "python",
            ],

            "linux": [
                "linux",
            ],

            "networking": [
                "network",
                "tcp",
                "udp",
            ],

            "excel": [
                "excel",
            ],

            "word": [
                "word",
            ],

            "hipaa": [
                "hipaa",
            ],
        }

    def apply(self, intent, results):

        requested = intent.get("skills", [])

        for result in results:

            assessment = result["assessment"]

            text = (
                assessment.get("name", "")
                + " "
                + assessment.get("description", "")
            ).lower()

            bonus = 0

            matched = []

            for skill in requested:

                aliases = self.skill_mapping.get(skill, [skill])

                for alias in aliases:

                    if alias in text:
                        bonus += 3
                        matched.append(skill)
                        break

            # Senior boost
            if (
                intent.get("job_level") == "Senior"
                and "professional"
                in " ".join(
                    assessment.get("job_levels", [])
                ).lower()
            ):
                bonus += 2

            result["retrieval_score"] = (
                result.get("retrieval_score", result.get("score", 0))
                + bonus
            )

            result["matched_skills"] = matched

        results.sort(
            key=lambda x: x["retrieval_score"],
            reverse=True,
        )

        return results