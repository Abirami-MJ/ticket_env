from fastapi import FastAPI
from pydantic import BaseModel
from env import TicketEnv
from tasks.easy import get_task as easy_task
import uvicorn

from fastapi import FastAPI

app = FastAPI(
    title="Ticket API",
    docs_url="/docs",       # enable Swagger
    redoc_url="/redoc"
)
env = TicketEnv()
state = None

class Action(BaseModel):
    action_type: str
    user_id: str
    seats: int

# Root route - This tells Hugging Face your app is alive
@app.get("/")
def read_root():
    return {
        "status": "active",
        "message": "Ticket Environment API is running",
        "endpoints": ["/docs", "/state", "/reset", "/step"]
    }

@app.post("/reset")
def reset():
    global state
    state = env.reset(easy_task())
    return state

@app.post("/step")
def step(action: Action):
    global state
    # model_dump() is preferred for Pydantic v2
    state, reward, done, info = env.step(action.model_dump())
    return {"state": state, "reward": reward, "done": done}

@app.get("/state")
def get_state():
    return state

# This must be at the very bottom with no trailing symbols or brackets
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)