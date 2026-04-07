from fastapi import FastAPI, Request
from server.logic import FakeNewsLogic, NewsAction
import uvicorn

app = FastAPI()
env_logic = FakeNewsLogic()

@app.get("/")
def read_root():
    return {"status": "Success", "message": "Environment is Up and Running!"}

@app.post("/reset")
async def reset(request: Request):
    data = await request.json()
    task_id = data.get("task_id")
    observation = env_logic.reset(task_id)
    return {"observation": observation, "info": {"task_id": env_logic.current_task_id}}

@app.post("/step")
async def step(request: Request):
    data = await request.json()
    action = NewsAction(**data["action"])
    observation, reward, done, info = env_logic.step(action)
    return {"observation": observation, "reward": float(reward), "done": bool(done), "info": info}
    
def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
