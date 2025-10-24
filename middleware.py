# middleware.py

import hashlib

class FatigueController:
    """
    Middleware that tracks fatigue based on how many times a prompt has been used.
    Each prompt has its own tiredness limit based on its length.
    """

    def __init__(self):
        self.task_counts = {}
        self.prompt_limits = {}

    def _key(self, prompt: str) -> str:
        return hashlib.sha256(prompt.strip().lower().encode()).hexdigest()

    def get_limit(self, prompt: str) -> int:
        key = self._key(prompt)
        if key not in self.prompt_limits:
            word_count = len(prompt.strip().split())
            limit = max(1, min(5, word_count // 30 or 1))  # 1 try per 30 words, bounded [1–5]
            self.prompt_limits[key] = limit
        return self.prompt_limits[key]

    def increment(self, prompt: str) -> int:
        key = self._key(prompt)
        self.task_counts[key] = self.task_counts.get(key, 0) + 1
        return self.task_counts[key]

    def is_tired(self, prompt: str) -> bool:
        key = self._key(prompt)
        count = self.task_counts.get(key, 0)
        limit = self.get_limit(prompt)
        return count >= limit

    def reset(self):
        self.task_counts.clear()
        self.prompt_limits.clear()
