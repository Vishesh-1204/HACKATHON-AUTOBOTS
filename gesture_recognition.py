"""
gesture_recognition.py — Agent 2: Gesture Recognition Engineer
==============================================================
Robust multi-condition gesture classifier operating on MediaPipe 3D geometries.
Eliminates naive binary flickers by computing continuous confidence thresholds
based on finger states, anatomical joint comparisons, and rotational-invariant
vector distances (compactness / pinch radii).
"""

import math
from collections import deque


class GestureRecognizer:
    """Detects individual finger states and classifies hand gestures via Confidence Scoring."""

    def __init__(self, history_length=5):
        """
        Args:
            history_length: Length of temporal history buffer to stabilize gesture outputs.
        """
        self._history = deque(maxlen=history_length)
        self._state_history = deque(maxlen=history_length)

    # ------------------------------------------------------------------
    # Per-finger detection (Robust Euclidean Distance logic)
    # ------------------------------------------------------------------
    def is_finger_up(self, landmarks, finger_id, handedness):
        """
        Calculates strict mechanical openness using spatial rotation-invariant distances.
        Eliminates perspective flipping errors by assessing radius from Wrist instead of local Y heights.
        """
        wrist = landmarks[0]
        
        if finger_id == 0:
            # Thumb: Open means resting outwardly opposed to the Pinky base.
            # Closed means folded directly over the palm.
            tip = landmarks[4]
            ip = landmarks[3]
            pinky_mcp = landmarks[17]
            
            dist_tip = math.hypot(tip.x - pinky_mcp.x, tip.y - pinky_mcp.y)
            dist_ip  = math.hypot(ip.x - pinky_mcp.x, ip.y - pinky_mcp.y)
            
            return 1 if dist_tip > dist_ip else 0
        else:
            # Fingers: Compare absolute 3D tip extension vs PIP joint radius from the wrist. 
            tip_ids = [8, 12, 16, 20]
            pip_ids = [6, 10, 14, 18]
            
            tip = landmarks[tip_ids[finger_id - 1]]
            pip = landmarks[pip_ids[finger_id - 1]]
            
            dist_tip = math.hypot(tip.x - wrist.x, tip.y - wrist.y)
            dist_pip = math.hypot(pip.x - wrist.x, pip.y - wrist.y)
            
            return 1 if dist_tip > dist_pip else 0

    def get_finger_states(self, landmarks, handedness):
        """
        Computes boolean finger extensions through a temporal majority voting buffer.
        """
        raw_states = [self.is_finger_up(landmarks, i, handedness) for i in range(5)]
        self._state_history.append(raw_states)
        
        smoothed_states = []
        for i in range(5):
            vote_count = sum(state_array[i] for state_array in self._state_history)
            smoothed_states.append(1 if vote_count > len(self._state_history) / 2 else 0)
            
        return smoothed_states

    # ------------------------------------------------------------------
    # Confidence-based Gesture Architecture
    # ------------------------------------------------------------------
    def recognize_gesture(self, landmarks, handedness):
        """
        Core physics engine computing anatomical vector shapes to prevent overlapping 
        classification logic (e.g. Fists accidentally triggering Pinch).
        """
        states = self.get_finger_states(landmarks, handedness)

        # Baseline Anatomical Indexing
        wrist = landmarks[0]
        mcp_index = landmarks[5]
        
        # Scaling magnitude (distance across palm base) to normalize hand-to-camera distance 
        palm_size = math.hypot(mcp_index.x - wrist.x, mcp_index.y - wrist.y)
        if palm_size < 1e-6: palm_size = 1e-6

        # Key Effector Coordinates
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]
        tips = [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]

        # 1. PINCH MATH
        pinch_dist = math.hypot(index_tip.x - thumb_tip.x, index_tip.y - thumb_tip.y)
        pinch_ratio_raw = pinch_dist / palm_size
        other_fingers_open = sum(states[2:]) # Evaluates Middle, Ring, Pinky
        
        # 2. OVERALL COMPACTNESS (FIST MATH)
        avg_tip_dist = sum(math.hypot(t.x - wrist.x, t.y - wrist.y) for t in tips) / 5.0
        compactness = avg_tip_dist / palm_size
        closed_fingers_count = 5 - sum(states)

        # 3. COMPETITIVE CONFIDENCE MATRIX
        confidences = {
            "Stop (Fist)": 0.0,
            "Grab (Pinch)": 0.0,
            "Point": 0.0,
            "Victory": 0.0,
            "Move (Open)": 0.0
        }

        # --- FIST --- (Max compactness tightly curling)
        fist_score = (closed_fingers_count / 5.0) + max(0.0, 1.5 - compactness)
        # Validation override to aggressively clamp when mathematically curled
        if closed_fingers_count >= 4 and compactness < 1.4:
            fist_score += 2.0
        confidences["Stop (Fist)"] = fist_score

        # --- PINCH --- (Index & Thumb touching tightly, BUT hand itself remains un-curled)
        pinch_score = max(0.0, 1.0 - pinch_ratio_raw) + (other_fingers_open / 3.0)
        # Structural Validation: Demands external fingers be somewhat open to count positively
        if pinch_ratio_raw < 0.8 and other_fingers_open >= 1:
            pinch_score += 1.5
        confidences["Grab (Pinch)"] = pinch_score

        # --- POINT ---
        if states[1] == 1 and sum(states[2:]) == 0:
            confidences["Point"] = 2.0 + max(0.0, compactness - 1.0)

        # --- VICTORY ---
        if states[1:3] == [1, 1] and sum(states[3:]) == 0:
            confidences["Victory"] = 2.0 + max(0.0, compactness - 1.0)

        # --- OPEN HAND ---
        open_score = (sum(states) / 5.0) + max(0.0, compactness - 1.5)
        if sum(states) >= 4 and compactness > 1.8:
            open_score += 1.5
        confidences["Move (Open)"] = open_score

        # 4. GESTURE PRIORITY LOGIC (Explicit override resolution)
        # A fully wrapped fist physically brings the thumb and index finger together.
        # This violently wipes out the Pinch score if the whole hand is compressed down.
        if confidences["Stop (Fist)"] > 1.8 and closed_fingers_count >= 4:
            confidences["Grab (Pinch)"] *= 0.1

        # Evaluate Victor
        winner_raw = max(confidences.items(), key=lambda x: x[1])[0]

        # Debug Readout Module (Optional Logging for fine-tuning weights)
        # print(f"[{handedness}] States: {states} | PinchR: {pinch_ratio_raw:.2f} | C-Pact: {compactness:.2f} -> WINNER: {winner_raw}")

        # 5. TEMPORAL STABILITY
        self._history.append(winner_raw)
        counts = {}
        for g in self._history:
            counts[g] = counts.get(g, 0) + 1
        smoothed_winner = max(counts, key=counts.get)

        return smoothed_winner, states
