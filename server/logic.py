import random
from typing import Optional, Dict, Any, List
from pydantic import BaseModel

# Standard Models (OpenEnv compatible)
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

        # 🔥 IMPORTANT
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
        if self.done: return self._get_obs(), 0.05, True, {"score": 0.05}
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
        
        if self.steps_left <= 0: self.done = True
        return self._get_obs(), float(reward), self.done, {"score": float(reward)}

    def _get_obs(self) -> NewsObservation:
        return NewsObservation(
            headline=self.current_task["headline"], 
            evidence=self.collected_evidence, 
            steps_left=self.steps_left
        )
    
    def state(self) -> Dict[str, Any]:
        return {"task_id": self.current_task_id, "steps_left": self.steps_left, "done": self.done}
 
