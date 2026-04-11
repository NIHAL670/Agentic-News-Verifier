"""
Inference Script for Fake News Verifier Environment
====================================================
Runs ALL tasks (task-1, task-2, task-3) sequentially.
Emits structured [START]/[STEP]/[END] logs per the OpenEnv spec.

Required env vars:
  API_BASE_URL - LLM endpoint
  MODEL_NAME   - Model identifier
  HF_TOKEN     - API key
"""

import asyncio
import os
import re
import sys
import textwrap
from typing import List, Optional

from openai import OpenAI
from server.logic import FakeNewsLogic, NewsAction, _safe_score

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
if not API_KEY:
    print("[ERROR] HF_TOKEN or API_KEY environment variable is required", flush=True)
    sys.exit(1)

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK = "fake-news-verifier"
MAX_STEPS = 5
TEMPERATURE = 0.0
MAX_TOKENS = 60

# All 3 tasks to run
ALL_TASK_IDS = ["task-1", "task-2", "task-3"]

SYSTEM_PROMPT = textwrap.dedent("""
You are a Fact-Checking Agent. Your goal is to verify if a news headline is True or False.

Available Actions:
1. search(keywords) - Search for evidence related to the headline.
2. verify(true) or verify(false) - Submit your final verdict.

Rules:
- First, use search() to gather evidence.
- Then, use verify(true) or verify(false) for your final answer.
- Reply with ONLY the action call. No extra text.
- Examples: search(moon cheese NASA) or verify(false)
""").strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    # Sanitize action string: remove newlines and limit length
    clean_action = action.replace("\n", " ").replace("\r", " ").strip()[:200]
    print(
        f"[STEP] step={step} action={clean_action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def parse_action(model_response: str) -> NewsAction:
    """Parse model output like 'search(moon cheese)' or 'verify(false)' into a NewsAction."""
    text = model_response.strip()

    # Match pattern: word(content)
    match = re.search(r"(\w+)\(([^)]*)\)", text)
    if match:
        action_type = match.group(1).lower()
        content = match.group(2).strip().replace("'", "").replace('"', "")
        if action_type in ("search", "verify"):
            return NewsAction(action_type=action_type, query_or_label=content)

    # Fallback: check if the response contains true/false directly
    lower = text.lower()
    if "true" in lower:
        return NewsAction(action_type="verify", query_or_label="true")
    if "false" in lower:
        return NewsAction(action_type="verify", query_or_label="false")

    # Default: search with whatever the model said
    return NewsAction(action_type="search", query_or_label=text[:100])


def get_model_action(client: OpenAI, headline: str, evidence: str, steps_left: int) -> str:
    """Call the LLM to get the next action."""
    user_prompt = (
        f"Headline: {headline}\n"
        f"Evidence so far: {evidence}\n"
        f"Steps remaining: {steps_left}\n"
        f"What is your next action?"
    )

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return text if text else "search(evidence)"
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return "search(evidence)"


def run_single_task(client: OpenAI, env: FakeNewsLogic, task_id: str) -> float:
    """
    Run a single task episode. Returns the final score strictly in (0, 1).
    Emits [START], [STEP]*, [END] logs.
    """
    rewards: List[float] = []
    steps_taken = 0
    success = False
    final_score = 0.01

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = env.reset(task_id)
        done = False

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            # Get action from LLM
            model_raw = get_model_action(
                client,
                headline=obs.headline,
                evidence=obs.evidence,
                steps_left=obs.steps_left,
            )
            action_obj = parse_action(model_raw)

            # Step the environment
            obs, reward, done, info = env.step(action_obj)

            rewards.append(reward)
            steps_taken = step

            log_step(
                step=step,
                action=model_raw,
                reward=reward,
                done=done,
                error=None,
            )

            if done:
                break

        # Calculate final score: average reward normalized to (0, 1)
        if rewards:
            total = sum(rewards)
            # Normalize: max possible is ~0.95 per step * MAX_STEPS
            final_score = total / (MAX_STEPS * 0.99)
            final_score = min(max(final_score, 0.0), 1.0)

        # Clamp to strictly (0, 1)
        final_score = _safe_score(final_score)
        success = final_score > 0.1

    except Exception as e:
        print(f"[DEBUG] Error during task {task_id}: {e}", flush=True)
        final_score = _safe_score(0.01)

    finally:
        log_end(success=success, steps=steps_taken, score=final_score, rewards=rewards)

    return final_score


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = FakeNewsLogic()

    all_scores = []

    for task_id in ALL_TASK_IDS:
        score = run_single_task(client, env, task_id)
        all_scores.append(score)

    # Summary
    print(f"\n[SUMMARY] tasks={len(ALL_TASK_IDS)} scores={','.join(f'{s:.2f}' for s in all_scores)}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
