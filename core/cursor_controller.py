import pyautogui
import numpy as np
from utils.screen import get_screen_size

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0

class CursorController:
    def __init__(self, smoothing=0.72, frame_reduction=0.15, dead_zone=4):
        """
        smoothing: higher = smoother but slower. 0.72 is noticeably calmer.
        dead_zone: pixels of movement to ignore — filters micro-tremor.
        """
        self.screen_w, self.screen_h = get_screen_size()
        self.smoothing = smoothing
        self.frame_reduction = frame_reduction
        self.dead_zone = dead_zone
        self.prev_x = self.screen_w // 2
        self.prev_y = self.screen_h // 2
        self.is_dragging = False
        self._frozen = False        # when True, cursor ignores hand movement
        self._freeze_x = self.screen_w // 2
        self._freeze_y = self.screen_h // 2

    def freeze(self):
        """Freeze cursor at current position — call when pinch starts."""
        self._frozen = True
        self._freeze_x = self.prev_x
        self._freeze_y = self.prev_y

    def unfreeze(self):
        """Resume normal cursor movement."""
        self._frozen = False

    def map_to_screen(self, cam_x, cam_y, frame_w, frame_h):
        margin_x = int(frame_w * self.frame_reduction)
        margin_y = int(frame_h * self.frame_reduction)
        cam_x = np.clip(cam_x, margin_x, frame_w - margin_x)
        cam_y = np.clip(cam_y, margin_y, frame_h - margin_y)
        norm_x = (cam_x - margin_x) / (frame_w - 2 * margin_x)
        norm_y = (cam_y - margin_y) / (frame_h - 2 * margin_y)
        return int(norm_x * self.screen_w), int(norm_y * self.screen_h)

    def move(self, cam_x, cam_y, frame_w, frame_h):
        if self._frozen and not self.is_dragging:
            return  # frozen + not dragging = click incoming, don't move

        target_x, target_y = self.map_to_screen(cam_x, cam_y, frame_w, frame_h)
        smooth_x = int(self.prev_x + (target_x - self.prev_x) * (1 - self.smoothing))
        smooth_y = int(self.prev_y + (target_y - self.prev_y) * (1 - self.smoothing))

        if abs(smooth_x - self.prev_x) > self.dead_zone or \
        abs(smooth_y - self.prev_y) > self.dead_zone:
            # During drag: just move the mouse — button is already held down via mouseDown()
            pyautogui.moveTo(smooth_x, smooth_y)
            self.prev_x, self.prev_y = smooth_x, smooth_y

    def left_click(self):
        pyautogui.click(button='left')

    def right_click(self):
        pyautogui.click(button='right')

    def start_drag(self):
        self.is_dragging = True
        pyautogui.mouseDown(button='left')

    def end_drag(self):
        self.is_dragging = False
        pyautogui.mouseUp(button='left')

    def get_current_position(self):
        return self.prev_x, self.prev_y