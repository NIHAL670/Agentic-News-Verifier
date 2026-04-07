from fastapi import FastAPI, Request
from .logic import FakeNewsLogic, NewsAction
from fastapi.responses import JSONResponse
import uvicorn
from .tasks import tasks

app = FastAPI()
env_logic = FakeNewsLogic()

@app.get("/")
def read_root():
    return {"status": "Success", "message": "Environment is Up and Running!"}

# SUCCESS FIX: Validator yahan se 3 tasks confirm karega
@app.get("/tasks/")
async def get_tasks():
    tasks_list = [
        {"id": task["input"], "name": task["name"]}
        for task in tasks
    ]
    return JSONResponse(content=tasks_list)

@app.post("/reset")
async def reset(request: Request):
    try:
        data = await request.json()
    except:
        data = {}
        
    task_id = data.get("task_id", "task-1")
    observation = env_logic.reset(task_id)
    
    # Observation ko explicitly dictionary mein badalna zaroori hai
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
        # Handle cases where action might be nested or direct
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
        return JSONResponse(content={"error": str(e), "done": True, "reward": 0.05}, status_code=200)

def main():
    """Main entry point for the server."""
    uvicorn.run(app, host="0.0.0.0", port=7861)

if __name__ == "__main__":
    main()
