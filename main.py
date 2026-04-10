"""
main.py — Multi-Hand Robotic Arm Simulation Orchestrator
=========================================================
Pipeline:
    Webcam → MediaPipe (max 2 hands) → per-hand gesture recognition
    → two independent RoboticArmVisualizer instances
    → dark canvas render  (NO webcam feed displayed)

Left hand  → left arm  (mount at x = 30 % of canvas width)
Right hand → right arm (mount at x = 70 % of canvas width)

Controls:
    ESC — quit
"""

import cv2
import numpy as np
import time
from hand_tracking import HandTracker
from gesture_recognition import GestureRecognizer
from visualization import RoboticArmVisualizer, BG_COLOR


# Canvas X positions for each hand's arm mount
ARM_X_FRACTION = {"Left": 0.28, "Right": 0.72}

# Display label shown above each mount
ARM_LABEL = {"Left": "LEFT HAND", "Right": "RIGHT HAND"}


def make_canvas(w, h):
    """Create a fresh dark background canvas for each frame."""
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:] = BG_COLOR
    return canvas


def main():
    print("Using NEW visualization module  [HIGH-PERFORMANCE MODE]")
    print("RoboticArmVisualizer active     [DUAL-HAND MODE]")

    # ── Camera setup ────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    canvas_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    canvas_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ── Module initialisation — one set per hand ──────────────────────
    tracker = HandTracker(max_hands=2, smoothing_factor=0.6)

    # Independent gesture recognisers so Left and Right history never mix
    recognizers = {
        "Left":  GestureRecognizer(history_length=5),
        "Right": GestureRecognizer(history_length=5),
    }

    arm_vis = {
        "Left":  RoboticArmVisualizer(),
        "Right": RoboticArmVisualizer(),
    }

    # Tracking & State Caches
    last_state = {
        "Left":  {"gesture": "No Hand", "fingers": [0, 0, 0, 0, 0]},
        "Right": {"gesture": "No Hand", "fingers": [0, 0, 0, 0, 0]},
    }
    last_valid_landmarks = {"Left": None, "Right": None}
    no_hand_frames = {"Left": 0, "Right": 0}

    # Temporal persistence config
    LOST_HAND_THRESHOLD = 8
    
    # ── FPS tracking ─────────────────────────────────────────────────
    prev_time = time.time()
    fps       = 0.0
    frame_count = 0
    
    # Init first active canvas so frame-skipping has something to show
    canvas = make_canvas(canvas_w, canvas_h)

    print(f"Canvas: {canvas_w}×{canvas_h} — press ESC to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        frame_count += 1

        # Mirror for natural selfie-view interaction
        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        # ── Fast Tracker Downscaling ──
        # MediaPipe is faster on smaller inputs, and coordinates remain normalized
        small_frame = cv2.resize(frame, (640, 360))
        results    = tracker.process_frame(small_frame)
        hands_data = tracker.get_all_hands_data(results)

        detected_this_frame = set()

        # Update cache for successfully detected hands
        for hand in hands_data:
            hand_type = hand["type"]
            raw_landmarks = hand["landmarks"]
            
            # Smooth structural jitter at the 3D landmark level
            smoothed = tracker.smooth_normalized_landmarks(raw_landmarks, hand_type)
            last_valid_landmarks[hand_type] = smoothed
            
            detected_this_frame.add(hand_type)
            no_hand_frames[hand_type] = 0

        # Process logic for all hands
        for hand_type in ("Left", "Right"):
            if hand_type not in detected_this_frame:
                no_hand_frames[hand_type] += 1
                
                # Grace period: reuse last valid landmarks to prevent 1-frame flickers
                if no_hand_frames[hand_type] < LOST_HAND_THRESHOLD and last_valid_landmarks[hand_type]:
                    smoothed_lms = last_valid_landmarks[hand_type]
                    gesture, finger_states = recognizers[hand_type].recognize_gesture(smoothed_lms, hand_type)
                    arm_vis[hand_type].update(smoothed_lms, finger_states, gesture, w, h, ARM_X_FRACTION[hand_type])
                    last_state[hand_type] = {"gesture": gesture, "fingers": finger_states}
                else:
                    # Hand definitively lost -> trigger reset
                    if no_hand_frames[hand_type] >= LOST_HAND_THRESHOLD:
                        last_valid_landmarks[hand_type] = None
                        tracker.reset_smoothing(hand_type)
                        
                        arm_vis[hand_type].update(None, [0,0,0,0,0], "Resetting...", w, h, ARM_X_FRACTION[hand_type])
                        last_state[hand_type] = {"gesture": "Resetting...", "fingers": [0,0,0,0,0]}
            else:
                # Normal live update
                smoothed_lms = last_valid_landmarks[hand_type]
                gesture, finger_states = recognizers[hand_type].recognize_gesture(smoothed_lms, hand_type)
                arm_vis[hand_type].update(smoothed_lms, finger_states, gesture, w, h, ARM_X_FRACTION[hand_type])
                last_state[hand_type] = {"gesture": gesture, "fingers": finger_states}

        # ── Rendering Loop (Skipped occasionally for FPS) ───────────
        if frame_count % 2 == 0:
            canvas = make_canvas(canvas_w, canvas_h)
            
            mid = canvas_w // 2
            cv2.line(canvas, (mid, 0), (mid, canvas_h), (35, 35, 50), 1, cv2.LINE_AA)

            for hand_type in ("Left", "Right"):
                arm_vis[hand_type].render(
                    canvas,
                    x_fraction=ARM_X_FRACTION[hand_type],
                    hand_label=ARM_LABEL[hand_type],
                )

            _draw_dual_hud(canvas, last_state, fps, canvas_w, canvas_h)

        # ── FPS Tracking Math ─────────────────────────────────────────
        now = time.time()
        dt  = now - prev_time
        if dt > 0:
            fps = 0.8 * fps + 0.2 * (1.0 / dt)
        prev_time = now

        # Always show the canvas window (fast)
        cv2.imshow("Robotic Arm Simulation  [Dual Hand]", canvas)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Shut down cleanly.")


# ══════════════════════════════════════════════════════════════════════
# Dual-hand HUD helper
# ══════════════════════════════════════════════════════════════════════

GESTURE_BADGE_COLORS = {
    "Move (Open)":  (80,  255, 100),
    "Stop (Fist)":  (50,   50, 240),
    "Grab (Pinch)": (50,  230, 230),
    "Point":        (240, 200,  50),
    "Victory":      (220,  80, 240),
    "No Hand":      (80,   80,  80),
    "Resetting...": (50,  150, 200),
}
FINGER_COLORS_LIST = [
    (80, 200, 255),   # Thumb  — amber
    (80, 255,  80),   # Index  — green
    (255, 200, 60),   # Middle — blue
    (180,  80, 255),  # Ring   — pink
    (60,  220, 255),  # Pinky  — yellow
]


def _draw_dual_hud(canvas, last_state, fps, cw, ch):
    """Draw two compact info panels — bottom-left and bottom-right."""
    panel_w, panel_h = 310, 130
    padding          = 14

    positions = {
        "Left":  (padding,          ch - panel_h - padding),
        "Right": (cw - panel_w - padding, ch - panel_h - padding),
    }

    for hand_type, (px, py) in positions.items():
        state   = last_state[hand_type]
        gesture = state["gesture"]
        fingers = state["fingers"]
        badge   = GESTURE_BADGE_COLORS.get(gesture, (140, 140, 140))

        # Panel background
        overlay = canvas.copy()
        cv2.rectangle(overlay, (px, py), (px + panel_w, py + panel_h),
                      (10, 12, 22), -1)
        cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0, canvas)
        cv2.rectangle(canvas, (px, py), (px + panel_w, py + panel_h),
                      badge, 1, cv2.LINE_AA)

        # Hand type title
        cv2.putText(canvas, f"{hand_type.upper()} HAND",
                    (px + 10, py + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160),
                    1, cv2.LINE_AA)

        # Gesture
        cv2.putText(canvas, gesture,
                    (px + 10, py + 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, badge,
                    2, cv2.LINE_AA)

        # Finger state dots  [T  I  M  R  P]
        labels = ["T", "I", "M", "R", "P"]
        for i, (lbl, state_bit) in enumerate(zip(labels, fingers)):
            cx = px + 14 + i * 54
            cy = py + 88
            color = FINGER_COLORS_LIST[i] if state_bit else (40, 40, 40)
            cv2.circle(canvas, (cx, cy), 13,
                       color, -1 if state_bit else 2, cv2.LINE_AA)
            cv2.putText(canvas, lbl, (cx - 4, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        (0, 0, 0) if state_bit else (70, 70, 70),
                        1, cv2.LINE_AA)

    # Central FPS indicator
    fps_color = (80, 255, 80) if fps > 20 else (50, 50, 230)
    fps_text  = f"FPS  {int(fps)}"
    (tw, _), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.putText(canvas, fps_text, ((cw - tw) // 2, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, fps_color, 2, cv2.LINE_AA)

    # Exit hint
    cv2.putText(canvas, "ESC — quit",
                (cw // 2 - 35, ch - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (60, 60, 60), 1, cv2.LINE_AA)


if __name__ == "__main__":
    main()
