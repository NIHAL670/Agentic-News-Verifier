from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn
from server.logic import FakeNewsLogic, NewsAction

app = FastAPI(title="Fake News Verification Environment")
env_logic = FakeNewsLogic()

# --- New: Isse 404 Error hat jayega ---
@app.get("/")
async def root():
    return {"status": "Online", "message": "Environment is Live"}

class ResetRequest(BaseModel):
    task_id: str = "easy"

class StepRequest(BaseModel):
    action_type: str 
    query_or_label: str

@app.post("/reset")
async def reset(request: ResetRequest):
    try:
        # Meta Rule: Respond with observation
        return env_logic.reset(request.task_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/step")
async def step(request: StepRequest):
    try:
        action = NewsAction(action_type=request.action_type, query_or_label=request.query_or_label)
        observation, reward, done = env_logic.step(action)
        return {
            "observation": observation,
            "reward": float(reward),
            "done": bool(done),
            "info": {}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/state")
async def state():
    return {
        "task_id": env_logic.current_task_id,
        "steps_left": env_logic.steps_left,
        "done": env_logic.done
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)