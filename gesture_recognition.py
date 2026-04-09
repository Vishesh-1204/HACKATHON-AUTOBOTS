"""
gesture_recognition.py — Agent 2: Gesture Recognition Engineer
==============================================================
Robust per-finger state detection using structural landmark comparisons
(tip vs PIP for fingers, lateral X for thumb) and temporal smoothing
for flicker-free gesture output.
"""

import math
from collections import deque


class GestureRecognizer:
    """Detects individual finger states and classifies hand gestures."""

    def __init__(self, history_length=5):
        """
        Args:
            history_length: Number of past frames to consider for temporal
                            majority-vote smoothing of the gesture label and states.
        """
        self._history = deque(maxlen=history_length)
        self._state_history = deque(maxlen=history_length)

    # ------------------------------------------------------------------
    # Per-finger detection
    # ------------------------------------------------------------------
    def is_finger_up(self, landmarks, finger_id, handedness):
        """
        Returns 1 if the given finger is extended and 0 otherwise.

        Uses anatomically meaningful comparisons:
        - Thumb:  lateral X displacement of TIP (4) vs IP (3),
                  adjusted for left/right hand.
        - Others: vertical Y of TIP vs PIP joint. A tip that is
                  higher than its PIP means the finger is extended.

        finger_id: 0=Thumb, 1=Index, 2=Middle, 3=Ring, 4=Pinky
        """
        if finger_id == 0:
            # Thumb: Use X-axis comparison depending on hand orientation.
            # Detect orientation using wrist (0) and index MCP (5).
            # If the index MCP (5) x is less than wrist (0) x, the hand is facing leftwards 
            # (which means the thumb should be on the left/lower x).
            # We determine the direction the thumb should point based on this palm orientation.
            is_right_oriented = landmarks[5].x < landmarks[0].x
            
            if is_right_oriented:
                return 1 if landmarks[4].x < landmarks[3].x else 0
            else:
                return 1 if landmarks[4].x > landmarks[3].x else 0
        else:
            tip_ids = [8, 12, 16, 20]
            pip_ids = [6, 10, 14, 18]
            tip = tip_ids[finger_id - 1]
            pip = pip_ids[finger_id - 1]
            # In screen space, lower Y = higher position
            return 1 if landmarks[tip].y < landmarks[pip].y else 0

    def get_finger_states(self, landmarks, handedness):
        """
        Returns [thumb, index, middle, ring, pinky] as a list of 0/1.
        Includes temporal smoothing (majority voting) to prevent flickering.
        """
        raw_states = [self.is_finger_up(landmarks, i, handedness) for i in range(5)]
        self._state_history.append(raw_states)
        
        smoothed_states = []
        for i in range(5):
            # Majority vote count for this specific finger across history
            vote_count = sum(state_array[i] for state_array in self._state_history)
            smoothed_states.append(1 if vote_count > len(self._state_history) / 2 else 0)
            
        return smoothed_states

    # ------------------------------------------------------------------
    # Gesture classification
    # ------------------------------------------------------------------
    def recognize_gesture(self, landmarks, handedness):
        """
        Classifies the current hand pose into a named gesture using
        finger states + normalised distance checks (for pinch).

        Returns:
            (gesture_name: str, finger_states: list[int])
        """
        states = self.get_finger_states(landmarks, handedness)

        # Palm-size normalised pinch distance
        wrist = landmarks[0]
        mcp_index = landmarks[5]
        palm_size = math.hypot(mcp_index.x - wrist.x, mcp_index.y - wrist.y)

        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        pinch_dist = math.hypot(index_tip.x - thumb_tip.x,
                                index_tip.y - thumb_tip.y)

        # --- Classification rules (ordered by specificity) ---

        # 1. Pinch — thumb and index physically close regardless of state
        if pinch_dist < palm_size * 0.8:
            raw = "Grab (Pinch)"

        # 2. Fist — all fingers curled (thumb may flicker)
        elif states[1:] == [0, 0, 0, 0]:
            raw = "Stop (Fist)"

        # 3. Point — only index up
        elif states[1] == 1 and sum(states[2:]) == 0:
            raw = "Point"

        # 4. Victory — index + middle up, others down
        elif states[1:3] == [1, 1] and sum(states[3:]) == 0:
            raw = "Victory"

        # 5. Open hand — at least 4 fingers up
        elif sum(states) >= 4:
            raw = "Move (Open)"

        # 6. Fallback heuristic
        else:
            up_count = sum(states[1:])  # ignore thumb for fallback
            if up_count >= 3:
                raw = "Move (Open)"
            elif up_count == 0:
                raw = "Stop (Fist)"
            else:
                raw = "Move (Open)"

        # --- Temporal majority-vote smoothing ---
        self._history.append(raw)
        counts = {}
        for g in self._history:
            counts[g] = counts.get(g, 0) + 1
        smoothed = max(counts, key=counts.get)

        return smoothed, states
