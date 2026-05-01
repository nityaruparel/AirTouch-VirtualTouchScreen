import cv2
import time
import math
from core.hand_tracker import HandTracker
from core.cursor_controller import CursorController
from core.gesture_detector import GestureDetector

CAMERA_INDEX    = 0
SHOW_PREVIEW    = True
SMOOTHING       = 0.72
FRAME_REDUCTION = 0.15

def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {CAMERA_INDEX}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    tracker    = HandTracker(max_hands=1)
    controller = CursorController(smoothing=SMOOTHING, frame_reduction=FRAME_REDUCTION)
    gesture    = GestureDetector(click_drop=0.28, hold_ms=600, debounce_ms=350)

    print("AirTouch running.")
    print("IMPORTANT: Hold your hand in your NATURAL cursor-moving pose for 2 seconds to calibrate.")
    print("Top-left corner = emergency exit.")

    start_time  = time.time()
    prev_time   = start_time
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame            = cv2.flip(frame, 1)
            frame_h, frame_w = frame.shape[:2]
            timestamp_ms     = int((time.time() - start_time) * 1000)
            frame_count     += 1

            results   = tracker.process(frame, timestamp_ms)
            fingertip = tracker.get_index_fingertip(results, frame_w, frame_h)

            if results.hand_landmarks:
                hand_lms = results.hand_landmarks[0]
                actions  = gesture.detect(hand_lms)

                # Debug every ~1 second
                if frame_count % 30 == 0 and gesture.is_calibrated():
                    r_left, r_right, baseline, threshold = gesture.get_debug_values(hand_lms)
                    print(f"left={r_left:.3f}  right={r_right:.3f}  "
                          f"baseline={baseline:.3f}  threshold={threshold:.3f}  "
                          f"pinching={'YES' if r_left < threshold else 'no'}")

                # Freeze on pinch
                if gesture.is_currently_clicking(hand_lms):
                    if not controller._frozen and not controller.is_dragging:
                        controller.freeze()
                else:
                    if controller._frozen:
                        controller.unfreeze()

                if fingertip:
                    controller.move(fingertip[0], fingertip[1], frame_w, frame_h)

                if actions["left_click"]:
                    controller.unfreeze()
                    controller.left_click()
                    print("LEFT CLICK")
                elif actions["right_click"]:
                    controller.unfreeze()
                    controller.right_click()
                    print("RIGHT CLICK")
                elif actions["drag_start"]:
                    controller.start_drag()
                    print("DRAG START")
                elif actions["drag_end"]:
                    controller.end_drag()
                    print("DRAG END")

            else:
                controller.unfreeze()
                if controller.is_dragging:
                    controller.end_drag()
                    print("DRAG RELEASED — hand lost")

            if SHOW_PREVIEW:
                frame = tracker.draw_landmarks(frame, results)

                # Show calibration progress
                if not gesture.is_calibrated():
                    cv2.putText(frame, "CALIBRATING — hold natural pose",
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                else:
                    if controller.is_dragging:
                        status, color = "DRAGGING", (255, 100,   0)
                    elif controller._frozen:
                        status, color = "CLICKING", (0,   165, 255)
                    else:
                        status, color = "READY",    (0,   255,   0)
                    cv2.putText(frame, status, (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                curr_time = time.time()
                fps       = 1 / (curr_time - prev_time + 1e-9)
                prev_time = curr_time
                cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                dot_color = (0, 255, 0) if fingertip else (0, 0, 255)
                cv2.circle(frame, (frame_w - 20, 20), 10, dot_color, -1)
                cv2.imshow("AirTouch", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                gesture.recalibrate()
                controller.unfreeze()

    finally:
        if controller.is_dragging:
            controller.end_drag()
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()
        print("AirTouch stopped.")

if __name__ == "__main__":
    main()