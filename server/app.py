from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Any, Dict, Optional
import uvicorn

from server.logic import FakeNewsLogic, NewsAction, _safe_score
from server.tasks import tasks, fake_news_grader

app = FastAPI(
    title="Fake News Verifier Environment",
    description="An OpenEnv-compatible environment for AI agent fact-checking",
    version="1.0.0",
)

env_logic = FakeNewsLogic()


class GradeRequest(BaseModel):
    task_id: str
    output: Dict[str, Any]


class ResetRequest(BaseModel):
    task_id: Optional[str] = "task-1"


class ActionRequest(BaseModel):
    action: Dict[str, Any]


@app.get("/")
def read_root():
    return {"status": "Success", "message": "Environment is Up and Running!"}


@app.get("/tasks")
@app.get("/tasks/")
async def get_tasks():
    """Return all tasks with grader information."""
    tasks_list = []
    for task in tasks:
        tasks_list.append({
            "id": task["id"],
            "name": task.get("name", task["id"]),
            "difficulty": task.get("difficulty", "medium"),
            "max_steps": task.get("max_steps", 5),
            "input": task.get("input", {}),
            "expected_output": task.get("expected_output", {}),
            "grader": {
                "type": "score",
                "criteria": task.get("expected_output", {})
            }
        })
    return JSONResponse(content=tasks_list)


@app.post("/grade")
@app.post("/grade/")
async def grade(req: GradeRequest):
    """Grade agent output for a specific task. Returns score strictly in (0, 1)."""
    try:
        task = next((t for t in tasks if t["id"] == req.task_id), None)
        if task is None:
            return JSONResponse(content={
                "score": 0.01,
                "feedback": f"Unknown task_id: {req.task_id}"
            })

        expected = task.get("expected_output", {})
        grader_fn = task.get("grader", fake_news_grader)
        raw_score = grader_fn(req.output, expected)
        score = _safe_score(raw_score)

        return JSONResponse(content={
            "score": float(round(max(0.01, min(0.99, score)), 4)),
            "feedback": f"Graded task {req.task_id}"
        })
    except Exception as e:
        return JSONResponse(content={
            "score": 0.01,
            "feedback": f"Grading error: {str(e)}"
        })


@app.post("/reset")
@app.post("/reset/")
async def reset(req: ResetRequest = None):
    """Reset the environment, optionally for a specific task."""
    if req is None or req.task_id is None:
        task_id = "task-1"
    else:
        task_id = req.task_id

    observation = env_logic.reset(task_id)
    return JSONResponse(content={
        "observation": {
            "headline": str(observation.headline),
            "evidence": str(observation.evidence),
            "steps_left": int(observation.steps_left)
        },
        "info": {"task_id": str(env_logic.current_task_id)}
    })


@app.post("/step")
@app.post("/step/")
async def step(req: ActionRequest):
    """Execute one step in the environment."""
    try:
        action = NewsAction(**req.action)
        observation, reward, done, info = env_logic.step(action)
        safe_reward = _safe_score(reward)
        return JSONResponse(content={
            "observation": {
                "headline": str(observation.headline),
                "evidence": str(observation.evidence),
                "steps_left": int(observation.steps_left)
            },
            "reward": float(round(max(0.01, min(0.99, safe_reward)), 4)),
            "done": bool(done),
            "info": {"score": safe_reward}
        })
    except Exception as e:
        return JSONResponse(content={
            "observation": {"headline": "", "evidence": "", "steps_left": 0},
            "reward": 0.01,
            "done": True,
            "info": {"error": str(e)}
        }, status_code=200)


@app.get("/state")
@app.get("/state/")
async def state():
    """Return the current environment state."""
    return JSONResponse(content=env_logic.state())


def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
