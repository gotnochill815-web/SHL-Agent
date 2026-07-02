class ConversationMemory:

    def __init__(self):
        self.memory = {}

    def add(
        self,
        session_id,
        query,
        intent=None,
        recommendations=None,
    ):

        if session_id not in self.memory:
            self.memory[session_id] = []

        self.memory[session_id].append(
            {
                "query": query,
                "intent": intent,
                "recommendations": recommendations,
            }
        )

        self.memory[session_id] = self.memory[session_id][-5:]

    def get_context(self, session_id):

        if session_id not in self.memory:
            return ""

        return "\n".join(
            item["query"]
            for item in self.memory[session_id]
        )

    def get_history(self, session_id):

        return self.memory.get(session_id, [])

    def get_last_intent(self, session_id):

        history = self.memory.get(session_id, [])

        if not history:
            return None

        return history[-1]["intent"]

    def get_last_recommendations(self, session_id):

        history = self.memory.get(session_id, [])

        if not history:
            return None

        return history[-1]["recommendations"]

    def clear(self, session_id):

        self.memory.pop(session_id, None)