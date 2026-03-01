import mss
import numpy as np
from PIL import Image
from env.io_controller import IOController, IOAction


class DesktopEnv:
    """
    Captures the real screen and exposes it as a PIL Image.
    Actions are executed via IOController (keyboard/mouse).
    """

    def __init__(self, monitor: int = 1, region: dict = None):
        """
        monitor: which monitor to capture (1 = primary)
        region: optional dict with keys top, left, width, height to capture a sub-region
        """
        self.monitor_idx = monitor
        self.region = region
        self.controller = IOController()
        self._sct = mss.mss()
        self.step_count = 0

    def screenshot(self) -> Image.Image:
        if self.region:
            raw = self._sct.grab(self.region)
        else:
            raw = self._sct.grab(self._sct.monitors[self.monitor_idx])
        return Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")

    def step(self, action_str: str) -> Image.Image:
        action = self.controller.parse_action(action_str)
        self.controller.execute(action)
        self.step_count += 1
        return self.screenshot()

    def execute(self, action: IOAction) -> Image.Image:
        self.controller.execute(action)
        self.step_count += 1
        return self.screenshot()

    def reset(self) -> Image.Image:
        self.step_count = 0
        return self.screenshot()

    @property
    def screen_size(self):
        return self.controller.screen_size

    def close(self):
        self._sct.close()
