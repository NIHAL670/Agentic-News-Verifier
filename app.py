import asyncio
import os
import threading
import uvicorn
import time
import textwrap
import re
import sys
from openai import OpenAI
from server.main import app  
from server.logic import NewsAction, NewsObservation

# Environment Variables
API_KEY = os.getenv("HF_TOKEN")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

SYSTEM_PROMPT = """
You are a Fact-Checking Agent. Verify if the news is True or False.
Actions:
1. search(keywords): To find evidence.
2. verify(True) or verify(False): For final verdict.
Reply ONLY with action call, e.g., search(moon) or verify(False).
"""

def parse_action(response):
    match = re.search(r"(\w+)\((.*)\)", response)
    if match:
        return match.group(1).lower(), match.group(2).strip().replace("'", "").replace('"', "")
    return "verify", "False"

async def run_agent():
    # Server ko fully load hone ka time dete hain
    await asyncio.sleep(12) 
    
    print("\n" + "="*30, flush=True)
    print("[AGENT] Starting Full Evaluation (3 Tasks)", flush=True)
    print("="*30 + "\n", flush=True)
    
    from server.main import env_logic
    client_llm = OpenAI(base_url="https://router.huggingface.co/v1", api_key=API_KEY)
    
    # 1. Teeno tasks define karo
    tasks = ["easy", "medium", "hard"]

    for task_id in tasks:
        print(f"\n>>> RUNNING TASK: {task_id.upper()}", flush=True)
        
        # 2. Environment ko current task_id ke saath reset karo
        obs = env_logic.reset(task_id)
        print(f"[START] Headline: {obs.headline}", flush=True)
        sys.stdout.flush()

        # Har task ke liye maximum 5 steps
        for step in range(1, 6):
            prompt = f"Headline: {obs.headline}\nEvidence: {obs.evidence}\nAction:"
            
            try:
                completion = client_llm.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=60,
                    temperature=0 # Consistency ke liye 0 rakhein
                )
                
                raw_action = completion.choices[0].message.content.strip()
                action_type, content = parse_action(raw_action)
                
                # logic.py mein step execute karo
                action_obj = NewsAction(action_type=action_type, query_or_label=content)
                obs, reward, done = env_logic.step(action_obj)
                
                print(f"[{task_id.upper()} - STEP {step}] Action: {raw_action} | Reward: {reward:.2f}", flush=True)
                sys.stdout.flush()
                
                if done:
                    status = "SUCCESS" if reward >= 1.0 else "FAILED"
                    print(f"[RESULT] {task_id.upper()} Task {status} | Total Score: {reward:.2f}", flush=True)
                    break
            except Exception as e:
                print(f"[ERROR] Task {task_id} failed at step {step}: {e}", flush=True)
                break
        
        # Task ke beech mein chhota gap (optional)
        await asyncio.sleep(2)

    print("\n" + "="*30, flush=True)
    print("[FINISH] All Meta Tasks Attempted!", flush=True)
    print("="*30, flush=True)
    sys.stdout.flush()

# --- MAIN RUNNER ---
if __name__ == "__main__":
    # 1. Server ko separate thread mein start karo
    config = uvicorn.Config(app, host="0.0.0.0", port=7860, log_level="info")
    server = uvicorn.Server(config)
    
    thread = threading.Thread(target=server.run)
    thread.start()

    # 2. Agent ko main loop mein chalao
    try:
        asyncio.run(run_agent())
    except KeyboardInterrupt:
        pass
    finally:
        # Space ko zinda rakhne ke liye server ko join karo
        thread.join()