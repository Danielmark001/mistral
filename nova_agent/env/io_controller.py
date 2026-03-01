import time
import pyautogui
from dataclasses import dataclass
from typing import Optional

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.05


@dataclass
class IOAction:
    type: str
    x: Optional[int] = None
    y: Optional[int] = None
    key: Optional[str] = None
    text: Optional[str] = None
    button: str = "left"
    clicks: int = 1
    scroll_amount: int = 3


class IOController:
    def __init__(self, move_duration=0.15):
        self.move_duration = move_duration
        self.screen_width, self.screen_height = pyautogui.size()

    def execute(self, action: IOAction):
        if action.type == "move":
            pyautogui.moveTo(action.x, action.y, duration=self.move_duration)

        elif action.type == "click":
            pyautogui.click(action.x, action.y, button=action.button, clicks=action.clicks)

        elif action.type == "double_click":
            pyautogui.doubleClick(action.x, action.y)

        elif action.type == "right_click":
            pyautogui.rightClick(action.x, action.y)

        elif action.type == "drag":
            # move to start position first, then drag to destination
            start_x = action.x
            start_y = action.y
            pyautogui.moveTo(start_x, start_y, duration=self.move_duration)
            pyautogui.mouseDown(button="left")
            pyautogui.moveTo(start_x, start_y, duration=self.move_duration)
            pyautogui.mouseUp(button="left")

        elif action.type == "scroll":
            pyautogui.scroll(action.scroll_amount, x=action.x, y=action.y)

        elif action.type == "key":
            pyautogui.press(action.key)

        elif action.type == "hotkey":
            keys = action.key.split("+")
            pyautogui.hotkey(*keys)

        elif action.type == "type":
            # use pyperclip + paste for reliable unicode support
            try:
                import pyperclip
                pyperclip.copy(action.text)
                pyautogui.hotkey("ctrl", "v")
            except ImportError:
                # fallback: type character by character
                for char in action.text:
                    pyautogui.press(char) if len(char) == 1 else pyautogui.typewrite(char, interval=0.02)

        elif action.type == "wait":
            time.sleep(0.5)

        else:
            raise ValueError(f"Unknown action type: {action.type}")

    def parse_action(self, action_str: str) -> IOAction:
        parts = action_str.strip().split()
        if not parts:
            return IOAction(type="wait")

        action_type = parts[0].lower()

        try:
            if action_type in ("click", "move", "right_click", "double_click") and len(parts) >= 3:
                return IOAction(type=action_type, x=int(parts[1]), y=int(parts[2]))

            if action_type == "scroll" and len(parts) >= 4:
                return IOAction(type="scroll", x=int(parts[1]), y=int(parts[2]), scroll_amount=int(parts[3]))

            if action_type == "key" and len(parts) >= 2:
                return IOAction(type="key", key=parts[1])

            if action_type == "hotkey" and len(parts) >= 2:
                return IOAction(type="hotkey", key=parts[1])

            if action_type == "type" and len(parts) >= 2:
                return IOAction(type="type", text=" ".join(parts[1:]))

        except (ValueError, IndexError):
            pass

        return IOAction(type="wait")

    @property
    def screen_size(self):
        return self.screen_width, self.screen_height
