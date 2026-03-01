import time
import wandb
from PIL import Image
from env.desktop_env import DesktopEnv
from models.planner import NOVAPlanner


SYSTEM_PROMPT = (
    "You are a desktop agent controlling a computer with keyboard and mouse. "
    "Look at the screenshot and output the next action in this exact format:\n"
    "  click X Y          - left click at pixel (X, Y)\n"
    "  double_click X Y   - double click at (X, Y)\n"
    "  right_click X Y    - right click at (X, Y)\n"
    "  move X Y           - move mouse to (X, Y)\n"
    "  scroll X Y N       - scroll N clicks at (X, Y), negative = down\n"
    "  key KEY            - press a key (enter, escape, tab, space, etc.)\n"
    "  hotkey COMBO       - e.g. hotkey ctrl+c\n"
    "  type TEXT          - type the given text\n"
    "  wait               - do nothing this step\n"
    "Output only one action line, nothing else."
)


class DesktopAgent:
    def __init__(
        self,
        planner: NOVAPlanner,
        monitor: int = 1,
        region: dict = None,
        max_steps: int = 50,
        step_delay: float = 0.5,
    ):
        self.planner = planner
        self.env = DesktopEnv(monitor=monitor, region=region)
        self.max_steps = max_steps
        self.step_delay = step_delay

    def run(self, task: str, wandb_log: bool = False) -> list[dict]:
        """
        Run the agent on a free-form desktop task.

        task: natural language description, e.g. "Open a terminal and type ls"
        Returns the full action history.
        """
        full_prompt = f"Task: {task}\n\n{SYSTEM_PROMPT}"
        history = []
        screenshot = self.env.reset()

        for step in range(self.max_steps):
            action_str = self.planner.predict_from_image(screenshot, full_prompt)
            print(f"Step {step + 1}: {action_str}")

            try:
                screenshot = self.env.step(action_str)
                error = None
            except Exception as e:
                error = str(e)
                screenshot = self.env.screenshot()

            record = {
                "step": step + 1,
                "action": action_str,
                "error": error,
            }
            history.append(record)

            if wandb_log and wandb.run is not None:
                wandb.log({
                    "desktop/step": step + 1,
                    "desktop/action": action_str,
                    "desktop/screenshot": wandb.Image(screenshot, caption=f"step {step + 1}"),
                })

            time.sleep(self.step_delay)

        self.env.close()
        return history

    def close(self):
        self.env.close()
