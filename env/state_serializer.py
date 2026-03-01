DIR_NAMES = {0: "east", 1: "south", 2: "west", 3: "north"}


def serialize_state(state: dict) -> str:
    lines = []

    agent_pos = state["agent_pos"]
    agent_dir = DIR_NAMES.get(state.get("agent_dir", 0), "unknown")
    lines.append(f"Agent at {agent_pos}, facing {agent_dir}.")

    goal_pos = state.get("goal_pos")
    if goal_pos:
        lines.append(f"Goal at {goal_pos}.")
    else:
        lines.append("Goal position unknown.")

    objects = state.get("visible_objects", [])
    if objects:
        for obj in objects:
            color = obj.get("color") or ""
            desc = f"{color} {obj['type']}".strip()
            lines.append(f"{desc.capitalize()} at {obj['pos']}.")
    else:
        lines.append("No obstacles visible.")

    step = state.get("step", 0)
    max_steps = state.get("max_steps", 50)
    lines.append(f"Step {step} of {max_steps}.")
    lines.append("Instruction: navigate to the goal.")
    lines.append("What is the next action?")

    return "\n".join(lines)
