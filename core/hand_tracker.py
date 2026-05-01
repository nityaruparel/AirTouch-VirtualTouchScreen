import mediapipe as mp
import cv2
import urllib.request
import os

MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand landmark model (~25MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded.")

class HandTracker:
    def __init__(self, max_hands=1, detection_confidence=0.7, tracking_confidence=0.7):
        download_model()

        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self.landmarker = HandLandmarker.create_from_options(options)
        self.INDEX_FINGER_TIP = 8

    def process(self, frame, timestamp_ms):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        return self.landmarker.detect_for_video(mp_image, timestamp_ms)

    def get_index_fingertip(self, results, frame_w, frame_h):
        if not results.hand_landmarks:
            return None

        hand = results.hand_landmarks[0]
        lm = hand[self.INDEX_FINGER_TIP]

        x = int(lm.x * frame_w)
        y = int(lm.y * frame_h)
        return (x, y)

    def draw_landmarks(self, frame, results):
        if not results.hand_landmarks:
            return frame

        # Manual drawing since mp.solutions.drawing_utils is gone
        for hand in results.hand_landmarks:
            for lm in hand:
                x = int(lm.x * frame.shape[1])
                y = int(lm.y * frame.shape[0])
                cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

            # Draw basic connections for index finger for visual feedback
            tips = [hand[i] for i in [0, 5, 6, 7, 8]]
            pts = [(int(l.x * frame.shape[1]), int(l.y * frame.shape[0])) for l in tips]
            for i in range(len(pts) - 1):
                cv2.line(frame, pts[i], pts[i+1], (0, 200, 255), 2)

        return frame

    def close(self):
        self.landmarker.close()