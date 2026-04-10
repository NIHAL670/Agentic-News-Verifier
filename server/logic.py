from typing import Dict, Any, Tuple
from pydantic import BaseModel

MIN_VALID_SCORE = 0.01
MAX_VALID_SCORE = 0.99


def _safe_score(value: float) -> float:
    """Clamp score to strictly (0, 1) — never exactly 0.0 or 1.0."""
    if value is None:
        return MIN_VALID_SCORE
    value = float(value)
    if value <= 0.0:
        return MIN_VALID_SCORE
    if value >= 1.0:
        return MAX_VALID_SCORE
    return round(value, 4)


class NewsAction(BaseModel):
    action_type: str
    query_or_label: str


class NewsObservation(BaseModel):
    headline: str
    evidence: str
    steps_left: int


class FakeNewsLogic:
    def __init__(self):
        self.task_data = {
            "task-1": {
                "headline": "NASA confirms Moon is Swiss Cheese.",
                "label": "false",
                "base_evidence": "Basic science contradicts this claim. No credible source found.",
                "search_results": "Scientific journals confirm the Moon is composed of rock and metal, not cheese. NASA has published extensive research on lunar geology.",
            },
            "task-2": {
                "headline": "New policy: 1000 units for all tomorrow.",
                "label": "true",
                "base_evidence": "Social media rumors suggest a new government policy.",
                "search_results": "Official Gazette Vol 42 confirms the Stimulus Act providing 1000 units has been signed into law.",
            },
            "task-3": {
                "headline": "Coffee leads to 20% IQ increase.",
                "label": "false",
                "base_evidence": "Viral blog post claim with no credible citation.",
                "search_results": "A peer-reviewed study found that coffee increases alertness temporarily but has no effect on IQ scores.",
            },
        }

        self.reset("task-1")

    def reset(self, task_id: str = None) -> NewsObservation:
        if not task_id or task_id not in self.task_data:
            task_id = "task-1"

        self.current_task_id = task_id
        self.current_task = self.task_data[task_id]
        self.collected_evidence = self.current_task["base_evidence"]
        self.steps_left = 5
        self.done = False
        self.total_reward = 0.0
        return self._get_obs()

    def step(self, action: NewsAction) -> Tuple[NewsObservation, float, bool, Dict[str, Any]]:
        if self.done:
            return self._get_obs(), _safe_score(0.05), True, {"score": _safe_score(self.total_reward)}

        self.steps_left -= 1
        reward = 0.05  # base reward for any action

        if action.action_type == "search":
            if self.collected_evidence == self.current_task["base_evidence"]:
                reward = 0.15  # first search is more valuable
                self.collected_evidence = self.current_task["search_results"]
            else:
                reward = 0.08  # diminishing returns for repeated search

        elif action.action_type == "verify":
            self.done = True
            if action.query_or_label.strip().lower() == self.current_task["label"].lower():
                reward = 0.79  # correct verdict
            else:
                reward = 0.05  # wrong verdict

        if self.steps_left <= 0:
            self.done = True

        safe_reward = _safe_score(reward)
        self.total_reward += safe_reward

        return self._get_obs(), safe_reward, self.done, {"score": _safe_score(self.total_reward)}

    def _get_obs(self) -> NewsObservation:
        return NewsObservation(
            headline=self.current_task["headline"],
            evidence=self.collected_evidence,
            steps_left=self.steps_left,
        )

    def get_expected_output(self) -> Dict[str, Any]:
        return {
            "label": self.current_task["label"],
            "evidence": self.current_task["search_results"],
        }

    def state(self) -> Dict[str, Any]:
        return {
            "task_id": self.current_task_id,
            "steps_left": self.steps_left,
            "done": self.done,
        }
