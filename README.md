title: Fake News Verifier
emoji: 🤖
colorFrom: blue
colorTo: red
sdk: docker
app_file: app.py
pinned: false

🤖 Fake News Verifier Agent (Meta Hackathon)

This project is an AI-powered Fact-Checking Environment built for the Meta Hackathon. It uses a custom Reinforcement Learning (RL) style environment to test an LLM agent's ability to verify news headlines using search actions and logical reasoning.
🚀 Features

    Multi-Level Tasks: Includes Easy, Medium, and Hard verification scenarios.
    Reasoning-First: The agent is incentivized to search() for evidence before giving a final verify() verdict.
    FastAPI Backend: Fully compliant with Meta's OpenEnv specifications for automated evaluation.
    LLM Integration: Powered by Qwen/Qwen2.5-72B-Instruct via Hugging Face Inference API.

🛠️ Project Structure

    server/logic.py: The core environment logic and task data.
    server/main.py: FastAPI server exposing /reset and /step endpoints.
    app.py: Main entry point that runs the server and the autonomous agent.
    inference.py: Standard inference script for benchmarking.
    openenv.yaml: Configuration for the OpenEnv benchmark.

📊 How it Works

    Reset: The environment loads a task (e.g., "Medium").
    Action: The Agent can either call search(keywords) to get more evidence or verify(True/False) to finish.
    Reward: - +0.10 for the first search.
        +1.00 for a correct verification.
        0.00 for incorrect verification.

📝 Sample Execution Logs

[AGENT] Starting Full Evaluation (3 Tasks)
>>> RUNNING TASK: EASY -> SUCCESS (Reward: 1.00)
>>> RUNNING TASK: MEDIUM -> SUCCESS (Reward: 1.10)
>>> RUNNING TASK: HARD -> SUCCESS (Reward: 1.10)
[FINISH] All Meta Tasks Attempted!
