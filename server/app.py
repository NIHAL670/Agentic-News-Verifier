from fastapi import FastAPI, Request
from server.logic import FakeNewsLogic, NewsAction
import uvicorn

app = FastAPI()
env_logic = FakeNewsLogic()

@app.get("/")
def read_root():
    return {"status": "Success", "message": "Environment is Up and Running!"}

# NEW: Validator isi endpoint se check karta hai ki 3 tasks hain ya nahi
@app.get("/tasks")
async def get_tasks():
    return [
        {"id": "task-1", "name": "Easy: Historical Fact"},
        {"id": "task-2", "name": "Medium: Current Events"},
        {"id": "task-3", "name": "Hard: Contextual Misinformation"}
    ]

@app.post("/reset")
async def reset(request: Request):
    try:
        data = await request.json()
    except:
        data = {}
        
    task_id = data.get("task_id")
    observation = env_logic.reset(task_id)
    
    return {
        "observation": {
            "headline": observation.headline,
            "evidence": observation.evidence,
            "steps_left": observation.steps_left
        },
        "info": {"task_id": env_logic.current_task_id}
    }

@app.post("/step")
async def step(request: Request):
    try:
        data = await request.json()
        # "action" key ke andar se data nikalna
        action_data = data.get("action", {})
        action = NewsAction(**action_data)
        
        observation, reward, done, info = env_logic.step(action)
        
        return {
            "observation": {
                "headline": observation.headline,
                "evidence": observation.evidence,
                "steps_left": observation.steps_left
            },
            "reward": float(reward),
            "done": bool(done),
            "info": info
        }
    except Exception as e:
        # Crash se bachne ke liye safe return
        return {"error": str(e), "done": True, "reward": 0.05}

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
