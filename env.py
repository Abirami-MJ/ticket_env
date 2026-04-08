import copy
from typing import Dict, Any


class TicketEnv:
    def __init__(self):
        self.state = {}
        self.done = False
        self.max_steps = 5
        self.current_step = 0

    def reset(self, task: Dict[str, Any] = None) -> Dict[str, Any]:
        self.current_step = 0
        self.done = False

        if task:
            self.state = copy.deepcopy(task)
            self.state["waitlist"] = []
            self.state["history"] = []
        else:
            self.state = {
                "available_seats": 2,
                "users": [],
                "waitlist": [],
                "history": []
            }

        return self.state

    def step(self, action: Dict[str, Any]):
        if self.done:
            return self.state, 0.0, True, {}

        self.current_step += 1
        reward = 0.0
        reason = ""

        action_type = action.get("action_type")
        user_id = action.get("user_id")
        seats = action.get("seats", 0)

        user = next((u for u in self.state["users"] if u["id"] == user_id), None)

        if not user:
            reward -= 0.2
            reason = "Invalid user"

        else:
            if action_type == "allocate":
                if user["payment_status"] == "success":
                    if seats <= self.state["available_seats"]:
                        self.state["available_seats"] -= seats
                        reward += 0.5
                        reason = "Seats allocated successfully"
                    else:
                        reward -= 0.3
                        reason = "Not enough seats"
                else:
                    reward -= 0.4
                    reason = "Payment not successful"

            elif action_type == "partial_allocate":
                if seats < user["requested_seats"] and seats <= self.state["available_seats"]:
                    self.state["available_seats"] -= seats
                    reward += 0.4
                    reason = "Partial allocation done"
                else:
                    reward -= 0.2
                    reason = "Invalid partial allocation"

            elif action_type == "waitlist":
                if user_id not in self.state["waitlist"]:
                    self.state["waitlist"].append(user_id)
                    reward += 0.2
                    reason = "Added to waitlist"
                else:
                    reason = "Already in waitlist"

            elif action_type == "retry_payment":
                if user["payment_status"] in ["failed", "deducted_failed"]:
                    user["payment_status"] = "success"
                    reward += 0.4
                    reason = "Payment retried successfully"
                else:
                    reward -= 0.1
                    reason = "Retry not needed"

            elif action_type == "refund":
                if user["payment_status"] == "deducted_failed":
                    user["payment_status"] = "refunded"
                    reward += 0.4
                    reason = "Refund processed"
                else:
                    reward -= 0.1
                    reason = "Refund not applicable"

            elif action_type == "cancel":
                self.state["available_seats"] += user.get("requested_seats", 1)
                reward += 0.3
                reason = "Booking cancelled, seats released"

                if self.state["waitlist"]:
                    self.state["waitlist"].pop(0)
                    reward += 0.2
                    reason += " and reallocated"

            else:
                reward -= 0.1
                reason = "Invalid action"

        if self.state["available_seats"] > 0:
            reward -= 0.05

        self.state["history"].append({
            "step": self.current_step,
            "action": action,
            "reward": reward,
            "reason": reason
        })

        if self.current_step >= self.max_steps:
            self.done = True

        return self.state, reward, self.done, {}
