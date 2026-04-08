---
title: Fake News Verifier
emoji: 🤖
colorFrom: blue
colorTo: red
sdk: docker
app_file: server/app.py
pinned: false
---

# 🤖 Agentic Fake News Verifier

An AI-powered fact-checking environment built for the **Meta PyTorch OpenEnv Hackathon**. This environment tests how effectively an LLM agent can verify news headlines through a multi-step reasoning workflow.

## Environment Description

The Fake News Verifier simulates a fact-checking workflow where an AI agent must:

1. **Analyze** a news headline presented as an observation
2. **Search** for supporting/contradicting evidence
3. **Verify** the headline as True or False based on evidence

The environment rewards agents for gathering evidence before making judgments, encouraging thorough investigation over hasty conclusions.

## Action Space

The agent can take two types of actions:

| Action | Format | Description |
|--------|--------|-------------|
| `search` | `search(keywords)` | Search for evidence related to the headline |
| `verify` | `verify(true/false)` | Submit final verdict on the headline's truthfulness |

### Action Schema (Pydantic)
```python
class NewsAction(BaseModel):
    action_type: str   # "search" or "verify"
    query_or_label: str  # search keywords or "true"/"false"
```

## Observation Space

Each observation contains:

| Field | Type | Description |
|-------|------|-------------|
| `headline` | `str` | The news headline to verify |
| `evidence` | `str` | Currently available evidence (updates after search) |
| `steps_left` | `int` | Remaining steps in the episode |

## Tasks

| Task ID | Difficulty | Headline | Expected Label |
|---------|-----------|----------|----------------|
| `task-1` | Easy | "NASA confirms Moon is Swiss Cheese." | `false` |
| `task-2` | Medium | "New policy: 1000 units for all tomorrow." | `true` |
| `task-3` | Hard | "Coffee leads to 20% IQ increase." | `false` |

## Reward Structure

| Action | Reward | Condition |
|--------|--------|-----------|
| Search (first) | `0.15` | First search action provides new evidence |
| Search (repeated) | `0.08` | Subsequent searches (diminishing returns) |
| Verify (correct) | `0.80` | Correct true/false label |
| Verify (incorrect) | `0.05` | Wrong label |
| Any other step | `0.05` | Base reward |

All rewards are clamped to the range `(0.01, 0.99)` — never exactly 0 or 1.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/reset` | POST | Reset environment (optional `task_id` in body) |
| `/step` | POST | Execute agent action |
| `/state` | GET | Get current environment state |
| `/tasks` | GET | List all tasks with grader info |
| `/grade` | POST | Grade agent output for a specific task |

## Setup & Local Testing

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Server
```bash
python -m uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Run Inference
```bash
export HF_TOKEN="your-token-here"
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
python inference.py
```

### Docker
```bash
docker build -t fake-news-verifier .
docker run -p 7860:7860 fake-news-verifier
```

## Project Structure

```
├── server/
│   ├── __init__.py      # Package init
│   ├── app.py           # FastAPI server with OpenEnv endpoints
│   ├── logic.py         # Core environment engine + reward logic
│   └── tasks.py         # Task definitions + grader functions
├── inference.py          # Baseline inference script (runs all 3 tasks)
├── openenv.yaml          # OpenEnv environment specification
├── Dockerfile            # Container config for HF Spaces
├── pyproject.toml        # Project metadata
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## License

MIT
