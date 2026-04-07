import asyncio
import os
import re
import sys
import textwrap
from typing import List, Optional

from openai import OpenAI
from server.logic import NewsAction, FakeNewsLogic

API_KEY = os.getenv("HF_TOKEN")
if API_KEY is None:
    raise ValueError("HF_TOKEN environment variable is required")

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
TASK_ID = os.getenv("TASK_ID", "task-1")
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
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True,
    )


def parse_action(model_response: str) -> NewsAction:
    match = re.search(r"(\w+)$$(.*)$$", model_response.strip())
    if match:
        action_type = match.group(1).lower()
        content = match.group(2).strip().replace("'", "").replace('"', "")
        return NewsAction(action_type=action_type, query_or_label=content)
    return NewsAction(action_type="verify", query_or_label="false")


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = FakeNewsLogic()

    rewards: List[float] = []
    steps_taken = 0
    success = False

    log_start(task=TASK_ID, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = env.reset(TASK_ID)
        done = False

        for step in range(1, MAX_STEPS + 1):
            prompt = (
                f"Headline: {obs.headline}\n"
                f"Evidence: {obs.evidence}\n"
                f"Steps Left: {obs.steps_left}\n"
                f"Action:"
            )

            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=50,
            )

            model_raw_response = completion.choices[0].message.content.strip()
            action_obj = parse_action(model_raw_response)

            obs, reward, done, info = env.step(action_obj)

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=model_raw_response, reward=reward, done=done, error=None)

            if done:
                success = reward >= 0.90
                break

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
    finally:
        log_end(success=success, steps=steps_taken, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())
