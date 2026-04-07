from typing import Dict, Any
from pydantic import BaseModel

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
                "base_evidence": "Basic science contradicts this.",
                "search_results": "Scientific journals confirm rock/metal structure."
            },
            "task-2": {
                "headline": "New policy: 1000 units for all tomorrow.",
                "label": "true",
                "base_evidence": "Social media rumors.",
                "search_results": "Official Gazette Vol 42 confirms Stimulus Act."
            },
            "task-3": {
                "headline": "Coffee leads to 20% IQ increase.",
                "label": "false",
                "base_evidence": "Viral blog post claim.",
                "search_results": "Study found alertness increase, not IQ."
            }
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
        return self._get_obs()

    def step(self, action: NewsAction):
        if self.done:
            return self._get_obs(), 0.05, True, {"score": 0.05}

        self.steps_left -= 1
        reward = 0.05

        if action.action_type == "search":
            if self.collected_evidence == self.current_task["base_evidence"]:
                reward = 0.15
                self.collected_evidence = self.current_task["search_results"]
            else:
                reward = 0.08

        elif action.action_type == "verify":
            self.done = True
            if action.query_or_label.strip().lower() == self.current_task["label"].lower():
                reward = 0.95
            else:
                reward = 0.05

        if self.steps_left <= 0:
            self.done = True

        return self._get_obs(), float(reward), self.done, {"score": float(reward)}

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


# server/tasks.py

from typing import Dict, Any, List

def fake_news_grader(output: Dict[str, Any], expected: Dict[str, Any]) -> bool:
    if not isinstance(output, dict):
        return False

    out_label = str(output.get("label", "")).strip().lower()
    exp_label = str(expected.get("label", "")).strip().lower()

    if out_label != exp_label:
        return False

    expected_keyword = str(expected.get("required_keyword", "")).strip().lower()
    if expected_keyword:
        out_evidence = str(output.get("evidence", "")).strip().lower()
        if expected_keyword not in out_evidence:
            return False

    return True


tasks: List[Dict[str, Any]] = [
    {
        "id": "task-1",
        "input": {
            "headline": "NASA confirms Moon is Swiss Cheese."
        },
        "expected_output": {
            "label": "false",
            "required_keyword": "science",
        },
        "grader": fake_news_grader,
    },
    {
        "id": "task-2",
        "input": {
            "headline": "New policy: 1000 units for all tomorrow."
        },
        "expected_output": {
            "label": "true",
            "required_keyword": "gazette",
        },
        "grader": fake_news_grader,
    },
    {
        "id": "task-3",
        "input": {
            "headline": "Coffee leads to 20% IQ increase."
        },
        "expected_output": {
            "label": "false",
            "required_keyword": "study",
        },
        "grader": fake_news_grader,
    },
]
