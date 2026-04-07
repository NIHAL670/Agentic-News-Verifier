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
            "task-1": {
                "headline": "NASA confirms the Moon is made of 100% Swiss Cheese.",
                "label": "False",
                "base_evidence": "Basic planetary science contradicts this.",
                "search_results": "Scientific journals confirm Moon is made of rock and metal."
            },
            "task-2": {
                "headline": "New government policy: All citizens to receive 1000 units of currency tomorrow.",
                "label": "True",
                "base_evidence": "Social media rumors are circulating.",
                "search_results": "Official Government Gazette Vol 42 confirms the 'Economic Stimulus Act'."
            },
            "task-3": {
                "headline": "Study shows drinking coffee leads to immediate 20% increase in IQ.",
                "label": "False",
                "base_evidence": "A viral blog post claims this study is revolutionary.",
                "search_results": "Original study found only temporary alertness, not IQ increase. Parody site misquoted the results."
            }
        }
        self.reset("task-1")

    def reset(self, task_id: str = "task-1") -> NewsObservation:
        # Task ID handle karna zaroori hai
        if task_id not in self.task_data:
            task_id = "task-1"
            
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
            # Done hone ke baad bhi reward 0.0 nahi, 0.01 rakho (safe side)
            return self._get_obs(),float(reward), self.done, {"score": float(reward)}

        self.steps_left -= 1
        reward = 0.05  # Default small positive reward (0 nahi hona chahiye)
        
        # Action Logic
        if action.action_type == "search":
            # Partial Reward for gathering information
            if self.collected_evidence == self.current_task["base_evidence"]:
                reward = 0.15 # 0.10 se thoda badha diya
                self.collected_evidence = self.current_task["search_results"]
            else:
                reward = 0.08 # Repeat search par bhi 0 se bada reward
            
        elif action.action_type == "verify":
            self.done = True
            # FINAL SCORE FIX: 0 aur 1 ke beech hona chahiye
            if action.query_or_label.strip().lower() == self.current_task["label"].lower():
                # Correct Answer: 1.0 ki jagah 0.95 do
                reward = 0.95 
            else:
                # Wrong Answer: 0.0 ki jagah 0.05 do
                reward = 0.05
        
        # Max steps reached check
        if self.steps_left <= 0:
            self.done = True
            if not self.done: # Agar bina verify kiye steps khatam hue
                reward = 0.02
            
        return self._get_obs(), float(reward), self.done

    def _get_obs(self) -> NewsObservation:
        return NewsObservation(
            headline=self.current_task["headline"],
            evidence=self.collected_evidence,
            steps_left=self.steps_left
        )
