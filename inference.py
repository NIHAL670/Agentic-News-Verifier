"""
Inference Script for Fake News Verifier Environment
====================================================
Runs ALL tasks (task-1, task-2, task-3) sequentially.
Emits structured [START]/[STEP]/[END] logs per the OpenEnv spec.

Required env vars:
  API_BASE_URL - LLM endpoint (has default)
  MODEL_NAME   - Model identifier (has default)
  HF_TOKEN     - API key (mandatory, no default)
"""

import os
import re
import sys
import textwrap
from typing import List, Optional

from openai import OpenAI
from server.logic import FakeNewsLogic, NewsAction, _safe_score

# --- Environment variables ---
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

BENCHMARK = "fake-news-verifier"
MAX_STEPS = 5
TEMPERATURE = 0.0
MAX_TOKENS = 80

ALL_TASK_IDS = ["task-1", "task-2", "task-3"]

SYSTEM_PROMPT = textwrap.dedent("""
You are a Fact-Checking Agent. Verify if a news headline is true or false.

Available actions (reply with ONLY one action per turn, no extra text):
  search(keywords)  — search for evidence. Use relevant scientific or official keywords.
  verify(true)      — submit verdict: headline is TRUE
  verify(false)     — submit verdict: headline is FALSE

Strategy:
1. Always search first to gather evidence. Include specific keywords like "science", "study", "gazette", "research", "official" in your search.
2. After seeing evidence, call verify(true) or verify(false).

Examples:
  search(NASA moon composition science geology)
  verify(false)
""").strip()


# --- Logging helpers (exact OpenEnv spec format) ---

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    clean_action = action.replace("\n", " ").replace("\r", " ").strip()[:200]
    print(
        f"[STEP] step={step} action={clean_action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True,
    )


# --- Action parser ---

def parse_action(model_response: str) -> NewsAction:
    text = model_response.strip()

    match = re.search(r"(\w+)\(([^)]*)\)", text)
    if match:
        action_type = match.group(1).lower()
        content = match.group(2).strip().replace("'", "").replace('"', "")
        if action_type in ("search", "verify"):
            return NewsAction(action_type=action_type, query_or_label=content)

    lower = text.lower()
    if "true" in lower:
        return NewsAction(action_type="verify", query_or_label="true")
    if "false" in lower:
        return NewsAction(action_type="verify", query_or_label="false")

    return NewsAction(action_type="search", query_or_label=text[:100])


# --- LLM call ---

def get_model_action(
    client: OpenAI,
    headline: str,
    evidence: str,
    steps_left: int,
    conversation_history: List[dict],
) -> str:
    user_msg = (
        f"Headline: {headline}\n"
        f"Evidence: {evidence}\n"
        f"Steps remaining: {steps_left}\n"
        f"What is your next action?"
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_msg})

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return text if text else "search(evidence)"
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return "search(evidence)"


# --- Single task runner ---

def run_single_task(client: OpenAI, env: FakeNewsLogic, task_id: str) -> float:
    rewards: List[float] = []
    steps_taken = 0
    success = False
    conversation_history: List[dict] = []

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = env.reset(task_id)
        done = False

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            model_raw = get_model_action(
                client,
                headline=obs.headline,
                evidence=obs.evidence,
                steps_left=obs.steps_left,
                conversation_history=conversation_history,
            )

            # Track conversation so agent remembers what it searched
            conversation_history.append({"role": "assistant", "content": model_raw})

            action_obj = parse_action(model_raw)

            last_error = None
            try:
                obs, reward, done, info = env.step(action_obj)
            except Exception as e:
                last_error = str(e)
                reward = 0.01
                done = True

            rewards.append(reward)
            steps_taken = step

            log_step(
                step=step,
                action=model_raw,
                reward=reward,
                done=done,
                error=last_error,
            )

            if done:
                break

        # Success = agent got a high reward (correct verify = 0.80)
        success = any(r >= 0.5 for r in rewards)

    except Exception as e:
        print(f"[DEBUG] Error during task {task_id}: {e}", flush=True)
        if not rewards:
            rewards = [0.01]

    finally:
        log_end(success=success, steps=steps_taken, rewards=rewards)

    return sum(rewards)


# --- Main ---

def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    env = FakeNewsLogic()

    all_scores = []
    for task_id in ALL_TASK_IDS:
        score = run_single_task(client, env, task_id)
        all_scores.append(score)

    print(
        f"\n[SUMMARY] tasks={len(ALL_TASK_IDS)} scores={','.join(f'{s:.2f}' for s in all_scores)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
