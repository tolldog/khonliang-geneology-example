"""
Genealogy router — maps queries to the right research role.
"""

from khonliang.roles.router import BaseRouter


class GenealogyRouter(BaseRouter):
    """
    Routes genealogy queries to specialist roles.

    Priority:
      1. Fact checking: validation, contradiction, verification requests
      2. Narrative: story, biography, history writing requests
      3. Researcher: general questions (fallback)
    """

    def __init__(self):
        super().__init__(fallback_role="researcher")

        # Fact checking / validation
        self.register_keywords(
            [
                "check",
                "validate",
                "verify",
                "contradiction",
                "wrong",
                "incorrect",
                "error",
                "mistake",
                "impossible",
                "suspicious",
                "anomaly",
                "accurate",
                "fact check",
                "plausible",
            ],
            "fact_checker",
        )

        # Narrative / storytelling
        self.register_keywords(
            [
                "story",
                "narrative",
                "tell me about",
                "biography",
                "life of",
                "history of",
                "write about",
                "describe",
                "what was life like",
                "journey",
                "migration",
            ],
            "narrator",
        )
