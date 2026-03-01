DIR_NAMES = {0: "east", 1: "south", 2: "west", 3: "north"}


def serialize_state(state: dict) -> str:
    lines = []
    agent_dir = DIR_NAMES.get(state.get("agent_dir", 0), "unknown")
    lines.append(f"Agent at {state['agent_pos']}, facing {agent_dir}.")

    goal = state.get("goal_pos")
    lines.append(f"Goal at {goal}." if goal else "Goal position unknown.")

    for obj in state.get("visible_objects", []):
        color = obj.get("color") or ""
        desc = f"{color} {obj['type']}".strip()
        lines.append(f"{desc.capitalize()} at {obj['pos']}.")

    if not state.get("visible_objects"):
        lines.append("No obstacles visible.")

    lines.append(f"Step {state.get('step', 0)} of {state.get('max_steps', 50)}.")
    lines.append("Instruction: navigate to the goal.")
    lines.append("What is the next action?")
    return "\n".join(lines)
