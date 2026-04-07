from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

from server.logic import FakeNewsLogic, NewsAction
from server.tasks import tasks

app = FastAPI()
env_logic = FakeNewsLogic()

@app.get("/")
def read_root():
    return {"status": "Success", "message": "Environment is Up and Running!"}

@app.get("/tasks/")
async def get_tasks():
    # Return serializable task metadata only
    tasks_list = []
    for task in tasks:
        tasks_list.append({
            "id": task["id"],
            "input": task.get("input", {}),
            "expected_output": task.get("expected_output", {})
        })
    return JSONResponse(content=tasks_list)

@app.post("/reset")
async def reset(request: Request):
    try:
        data = await request.json()
    except Exception:
        data = {}

    task_id = data.get("task_id", "task-1")
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
async def step(request: Request):
    try:
        data = await request.json()
        action_data = data.get("action", data)
        action = NewsAction(**action_data)

        observation, reward, done, info = env_logic.step(action)

        return JSONResponse(content={
            "observation": {
                "headline": str(observation.headline),
                "evidence": str(observation.evidence),
                "steps_left": int(observation.steps_left)
            },
            "reward": float(reward),
            "done": bool(done),
            "info": info
        })
    except Exception as e:
        return JSONResponse(content={
            "observation": {
                "headline": "",
                "evidence": "",
                "steps_left": 0
            },
            "reward": 0.05,
            "done": True,
            "info": {"error": str(e)}
        }, status_code=200)

def main():
    """Main entry point for the server."""
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
