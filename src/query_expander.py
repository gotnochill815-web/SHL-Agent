class QueryExpander:

    def __init__(self):

        self.synonyms = {

            "java": [
                "core java",
                "java 8",
                "java ee",
                "j2ee",
            ],

            "spring": [
                "spring boot",
                "spring framework",
            ],

            "sql": [
                "mysql",
                "postgresql",
                "oracle sql",
                "database",
            ],

            "aws": [
                "amazon web services",
                "ec2",
                "s3",
                "cloud",
            ],

            "docker": [
                "container",
                "containers",
                "containerization",
            ],

            "python": [
                "django",
                "flask",
                "fastapi",
            ],

            "linux": [
                "unix",
                "ubuntu",
            ],

            "networking": [
                "tcp",
                "udp",
                "network protocols",
            ],

            "excel": [
                "spreadsheet",
                "microsoft excel",
            ],

            "word": [
                "microsoft word",
            ],

            "finance": [
                "financial accounting",
                "banking",
            ],

            "leadership": [
                "executive",
                "director",
                "cxo",
            ],
        }

    def expand(self, intent):

        terms = []

        for skill in intent.get("skills", []):

            terms.append(skill)

            terms.extend(
                self.synonyms.get(skill, [])
            )

        terms.extend(
            intent.get("domains", [])
        )

        terms.extend(
            intent.get("assessment_types", [])
        )

        if intent.get("job_level"):
            terms.append(intent["job_level"])

        return " ".join(dict.fromkeys(terms))