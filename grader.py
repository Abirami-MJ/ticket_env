from typing import List, Dict, Any


def grade(history: List[Dict[str, Any]]) -> float:
    if not history:
        return 0.0

    allocation = 0
    payment = 0
    efficiency = 0
    cancellation = 0

    total = len(history)

    for step in history:
        action = step["action"]
        reward = step["reward"]
        action_type = action["action_type"]

        if action_type in ["allocate", "partial_allocate"] and reward > 0:
            allocation += 1

        if action_type in ["retry_payment", "refund"] and reward > 0:
            payment += 1

        if action_type == "cancel" and reward > 0:
            cancellation += 1

        if reward > 0:
            efficiency += 1

    score = (
        0.3 * (allocation / total) +
        0.3 * (payment / total) +
        0.2 * (efficiency / total) +
        0.2 * (cancellation / total)
    )

    return max(0.0, min(1.0, score))