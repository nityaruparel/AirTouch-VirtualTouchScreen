import time
import math

IDLE       = "IDLE"
PINCH_DOWN = "PINCH_DOWN"
DRAGGING   = "DRAGGING"

CALIBRATION_FRAMES = 20

class GestureDetector:
    def __init__(self, click_drop=0.35, hold_ms=600, debounce_ms=350):
        self.click_drop  = click_drop
        self.hold_ms     = hold_ms
        self.debounce_ms = debounce_ms
        self._reset_calibration()

    def _reset_calibration(self):
        self._baseline      = None
        self._calib_samples = []
        self._calibrated    = False
        self.drag_state     = IDLE
        self.curl_start_time  = None
        self._dropout_ms      = 0
        self.last_left_click_time  = 0
        self.last_right_click_time = 0
        print("[AirTouch] Calibrating — hold your natural cursor pose...")

    def recalibrate(self):
        """Call this when user presses C."""
        self._reset_calibration()

    # ── Geometry ──────────────────────────────────────────────────────────────

    def _dist(self, a, b):
        return math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2 + (a.z-b.z)**2)

    def _hand_size(self, lms):
        return self._dist(lms[0], lms[9])

    def _pinch_ratio(self, lms, a=4, b=8):
        d    = self._dist(lms[a], lms[b])
        size = self._hand_size(lms)
        return d / size if size > 0 else 1.0

    def _update_baseline(self, lms):
        if self._calibrated:
            return
        ratio = self._pinch_ratio(lms, 4, 8)
        self._calib_samples.append(ratio)
        if len(self._calib_samples) >= CALIBRATION_FRAMES:
            sorted_s       = sorted(self._calib_samples)
            self._baseline = sorted_s[len(sorted_s) // 2]
            self._calibrated = True
            print(f"[AirTouch] Calibrated. Baseline={self._baseline:.3f}  "
                  f"Click threshold={self._baseline - self.click_drop:.3f}")

    def _drift_correct(self, lms):
        """
        Slowly nudge baseline toward current neutral when hand is clearly
        open (not clicking). Handles camera distance changes mid-session.
        Correction rate is tiny (1%) so it never overrides a real click.
        """
        if not self._calibrated:
            return
        current_ratio = self._pinch_ratio(lms, 4, 8)
        # Only drift-correct when hand is clearly NOT clicking
        if current_ratio > self._baseline - (self.click_drop * 0.5):
            self._baseline = self._baseline * 0.99 + current_ratio * 0.01

    def _is_left_pinching(self, lms):
        if not self._calibrated:
            return False
        return self._pinch_ratio(lms, 4, 8) < (self._baseline - self.click_drop)

    def _is_right_pinching(self, lms):
        if not self._calibrated:
            return False
        return self._pinch_ratio(lms, 4, 12) < (self._baseline - self.click_drop)

    def _debounced(self, last_time):
        return (time.time() * 1000 - last_time) > self.debounce_ms

    # ── Public API ────────────────────────────────────────────────────────────

    def is_currently_clicking(self, lms):
        return self._is_left_pinching(lms)

    def is_calibrated(self):
        return self._calibrated

    def get_debug_values(self, lms):
        ratio_left  = self._pinch_ratio(lms, 4, 8)
        ratio_right = self._pinch_ratio(lms, 4, 12)
        threshold   = (self._baseline - self.click_drop) if self._calibrated else None
        return ratio_left, ratio_right, self._baseline, threshold

    def detect(self, lms):
        now_ms = time.time() * 1000
        self._update_baseline(lms)
        self._drift_correct(lms)

        result = {
            "left_click":  False,
            "right_click": False,
            "drag_start":  False,
            "drag_end":    False,
            "dragging":    False,
            "calibrating": not self._calibrated,
        }

        if not self._calibrated:
            return result

        left_pinch  = self._is_left_pinching(lms)
        right_pinch = self._is_right_pinching(lms)

        # Right click — thumb + middle
        if right_pinch and not left_pinch:
            if self._debounced(self.last_right_click_time):
                result["right_click"] = True
                self.last_right_click_time = now_ms
            self.drag_state = IDLE
            return result

        # Left click / drag state machine
        if self.drag_state == IDLE:
            if left_pinch:
                self.drag_state      = PINCH_DOWN
                self.curl_start_time = now_ms
                self._dropout_ms     = 0

        elif self.drag_state == PINCH_DOWN:
            if left_pinch:
                self._dropout_ms = 0
                if (now_ms - self.curl_start_time) >= self.hold_ms:
                    self.drag_state = DRAGGING
                    result["drag_start"] = True
            else:
                self._dropout_ms += 16
                if self._dropout_ms > 100:
                    if self._debounced(self.last_left_click_time):
                        result["left_click"]      = True
                        self.last_left_click_time = now_ms
                    self.drag_state      = IDLE
                    self.curl_start_time = None

        elif self.drag_state == DRAGGING:
            if left_pinch:
                result["dragging"] = True
            else:
                result["drag_end"] = True
                self.drag_state      = IDLE
                self.curl_start_time = None

        return result