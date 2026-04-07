itle: Fake News Verifier
emoji: 🤖
colorFrom: blue
colorTo: red
sdk: docker
app_file: app.py
pinned: false

🤖 Agentic Fake News Verifier (Meta Hackathon)

This project is a custom Fact-Checking Environment built for the Meta PyTorch Hackathon x Scaler School of Technology. It is designed as an OpenAI Gym-style environment (OpenEnv) to evaluate how effectively an LLM agent can verify news headlines using search actions.
🚀 Phase 2 Optimized Features

    Strictly Compliant Rewards: Scores are balanced between 0.05 and 0.95 to meet Meta's strict (0, 1) range requirement.
    Triple Task Validation: Includes three distinct tasks (task-1, task-2, task-3) with built-in graders.
    FastAPI Backend: Fully compliant with OpenEnv specifications for /reset and /step endpoints.
    Dockerized Deployment: Ready for seamless evaluation on Hugging Face Spaces.

🛠️ Project Structure

    server/logic.py: The core environment engine containing the news database, reward logic, and task graders.
    server/app.py: The FastAPI server exposing the environment to the Meta Validator.
    Dockerfile: Container configuration for Hugging Face Spaces deployment.
    pyproject.toml: Project metadata and dependency management.

📊 Environment Logic & Rewards

To ensure high-quality verification, the environment follows a specific reward structure:
Action 	Reward 	Description
Search 	+0.15 	Incentivizes gathering evidence before a verdict.
Correct Verify 	+0.95 	High reward for accurate Fact-Checking.
Incorrect Verify 	+0.05 	Minimal reward to maintain the (0, 1) range.
Default Step 	+0.05 	Base reward for taking an action.
📝 How to Test Locally

    Clone the repository.
    Install dependencies: pip install fastapi uvicorn pydantic.
    Run the server:

    python -m uvicorn server.app:app --host 0.0.0.0 --port 7860


