import random
from typing import Optional, Dict, Any
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
        # Real-world scenarios: Easy (Obvious), Medium (Requires 1 search), Hard (Conflicting info)
        self.task_data = {
            "easy": {
                "headline": "NASA confirms the Moon is made of 100% Swiss Cheese.",
                "label": "False",
                "base_evidence": "Basic planetary science contradicts this.",
                "search_results": "Scientific journals confirm Moon is made of rock and metal."
            },
            "medium": {
                "headline": "New government policy: All citizens to receive 1000 units of currency tomorrow.",
                "label": "True",
                "base_evidence": "Social media rumors are circulating.",
                "search_results": "Official Government Gazette Vol 42 confirms the 'Economic Stimulus Act'."
            },
            "hard": {
                "headline": "Study shows drinking coffee leads to immediate 20% increase in IQ.",
                "label": "False",
                "base_evidence": "A viral blog post claims this study is revolutionary.",
                "search_results": "Original study found only temporary alertness, not IQ increase. Parody site misquoted the results."
            }
        }
        self.reset("easy")

    def reset(self, task_id: str = "easy") -> NewsObservation:
        # Task ID handle karna zaroori hai
        if task_id not in self.task_data:
            task_id = "easy"
            
        self.current_task_id = task_id
        self.current_task = self.task_data[task_id]
        self.collected_evidence = self.current_task["base_evidence"]
        self.steps_left = 5
        self.done = False
        
        return NewsObservation(
            headline=self.current_task["headline"],
            evidence=self.collected_evidence,
            steps_left=self.steps_left
        )

    def step(self, action: NewsAction):
        if self.done:
            return self._get_obs(), 0.0, True

        self.steps_left -= 1
        reward = 0.0
        
        # Action Logic
        if action.action_type == "search":
            # Partial Reward (0.1) for gathering more information
            if self.collected_evidence == self.current_task["base_evidence"]:
                reward = 0.10
                self.collected_evidence = self.current_task["search_results"]
            else:
                reward = 0.02 # Diminishing returns for repeat search
            
        elif action.action_type == "verify":
            self.done = True
            # Final Reward: 1.0 if correct, 0.0 if wrong
            if action.query_or_label.strip().lower() == self.current_task["label"].lower():
                reward = 1.0
            else:
                reward = 0.0
        
        # Max steps reached check
        if self.steps_left <= 0:
            self.done = True
            
        return self._get_obs(), reward, self.done

    def _get_obs(self) -> NewsObservation:
        return NewsObservation(
            headline=self.current_task["headline"],
            evidence=self.collected_evidence,
            steps_left=self.steps_left
        )