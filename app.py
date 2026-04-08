# -------------------- IMPORTS --------------------
from fastapi import FastAPI
from pydantic import BaseModel
from env import TicketEnv
from tasks.easy import get_task as easy_task
import os
import gradio as gr

# -------------------- FASTAPI --------------------
app = FastAPI()

# Fix for Hugging Face Spaces
if "SPACE_ID" in os.environ:
    app.root_path = ""

env = TicketEnv()
state = None

# -------------------- MODEL --------------------
class Action(BaseModel):
    action_type: str
    user_id: str
    seats: int

# -------------------- API ROUTES --------------------

@app.post("/reset")
def reset_env():
    global state
    state = env.reset(easy_task())
    return state

@app.post("/step")
def step_env(action: Action):
    global state
    state, reward, done, info = env.step(action.model_dump())
    return {
        "state": state,
        "reward": reward,
        "done": done
    }

@app.get("/state")
def get_state():
    return state

# -------------------- GRADIO FUNCTIONS --------------------

def ui_state():
    return state if state is not None else {"message": "Click Reset first"}

def ui_reset():
    return reset_env()


def ui_step(action_type, user_id, seats):
    try:
        action = Action(
            action_type=action_type,
            user_id=user_id,
            seats=int(seats)
        )
        return step_env(action)
    except Exception as e:
        return {"error": str(e)}    

# -------------------- GRADIO UI --------------------

with gr.Blocks() as demo:
    gr.Markdown("# 🎟️ Ticket Environment UI")

    with gr.Row():
        state_btn = gr.Button("Get State")
        reset_btn = gr.Button("Reset")

    output = gr.JSON()

    state_btn.click(ui_state, outputs=output)
    reset_btn.click(ui_reset, outputs=output)

    gr.Markdown("## Take Action")

    action_type = gr.Textbox(label="Action Type")
    user_id = gr.Textbox(label="User ID")
    seats = gr.Number(label="Seats")

    step_btn = gr.Button("Step")
    step_output = gr.JSON()

    step_btn.click(
        ui_step,
        inputs=[action_type, user_id, seats],
        outputs=step_output
    )

# -------------------- MOUNT GRADIO --------------------

app = gr.mount_gradio_app(app, demo, path="/")
