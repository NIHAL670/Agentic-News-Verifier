from fastapi import FastAPI, Request
from server.logic import FakeNewsLogic, NewsAction
import uvicorn

# Environment initialization
app = FastAPI()
env_logic = FakeNewsLogic()

@app.get("/")
def read_root():
    return {"status": "running", "message": "Meta PyTorch Hackathon Environment Ready"}

@app.post("/reset")
async def reset(request: Request):
    try:
        data = await request.json()
    except:
        data = {}
    
    task_id = data.get("task_id")
    # Humne logic.py mein random choice rakha hai, toh task_id None hone par bhi 
    # ye 3 alag tasks mein se koi ek pick karega.
    observation = env_logic.reset(task_id)
    
    return {
        "observation": observation,
        "info": {"task_id": env_logic.current_task_id}
    }

@app.post("/step")
async def step(request: Request):
    try:
        data = await request.json()
    except:
        return {"error": "Invalid JSON"}

    action_data = data.get("action", {})
    # Action object creation
    action = NewsAction(**action_data)
    
    # logic.py se results lena
    observation, reward, done, info = env_logic.step(action)
    
    # Meta ko return karna
    return {
        "observation": observation,
        "reward": float(reward),
        "done": bool(done),
        "info": info  # Is info mein hamara {"score": float} ja raha hai
    }

def main():
    # Port 7860 Hugging Face Spaces ke liye standard hai
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
