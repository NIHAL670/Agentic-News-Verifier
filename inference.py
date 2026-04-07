import asyncio
import os
import textwrap
import re
from typing import List, Optional
from openai import OpenAI

# ISSUE 1 FIXED: Echo env ki jagah tumhara Fake News logic import ho raha hai
from server.logic import NewsAction, FakeNewsLogic 

# Environment Variables
API_KEY = os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
TASK_NAME = os.getenv("TASK_ID", "easy") 
BENCHMARK = "fake-news-verifier"
MAX_STEPS = 5

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a Fact-Checking Agent. Your goal is to verify if a news headline is True or False.
    Available Actions:
    1. search(keywords): Use this to find evidence.
    2. verify(True) or verify(False): Use this for your final verdict.
    
    Rules:
    - Reply ONLY with the action call, e.g., search(moon cheese) or verify(False).
    """
).strip()

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)

def parse_action(model_response: str) -> NewsAction:
    match = re.search(r"(\w+)\((.*)\)", model_response.strip())
    if match:
        action_type = match.group(1).lower()
        content = match.group(2).strip().replace("'", "").replace('"', "")
        return NewsAction(action_type=action_type, query_or_label=content)
    return NewsAction(action_type="verify", query_or_label="False")

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = FakeNewsLogic() 
    
    rewards: List[float] = []
    steps_taken = 0
    
    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Task ID handle karne ke liye (task-1 format)
        task_id = TASK_NAME if "task-" in TASK_NAME else f"task-1"
        obs = env.reset(task_id)
        done = False

        for step in range(1, MAX_STEPS + 1):
            prompt = f"Headline: {obs.headline}\nEvidence: {obs.evidence}\nSteps Left: {obs.steps_left}\nAction:"
            
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=50
            )
            
            model_raw_response = completion.choices[0].message.content.strip()
            action_obj = parse_action(model_raw_response)
            
            # Logic.py se 4 values aani chahiye: obs, reward, done, info
            obs, reward, done, info = env.step(action_obj)
            
            rewards.append(reward)
            steps_taken = step
            
            log_step(step=step, action=model_raw_response, reward=reward, done=done, error=None)

            if done:
                break

        # Meta Score Fix: 1.00 bilkul nahi hona chahiye
        success = any(r >= 0.90 for r in rewards)
        final_score = 0.95 if success else 0.05 # Strictly between 0 and 1

    except Exception as e:
        print(f"Error: {e}")
        success = False
        final_score = 0.01 # 0.00 se bachne ke liye
    finally:
        log_end(success=success, steps=steps_taken, score=final_score, rewards=rewards)

if __name__ == "__main__":
    asyncio.run(main())
 
