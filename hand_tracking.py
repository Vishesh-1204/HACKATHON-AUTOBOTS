"""
hand_tracking.py — Multi-Hand Computer Vision Specialist
=========================================================
Supports up to 2 simultaneous hands with independent per-hand EMA
smoothing buffers so Left and Right hands never interfere.

Public API
----------
process_frame(frame)         → raw MediaPipe results
get_all_hands_data(results)  → list of hand dicts (see below)
get_smoothed_landmarks(landmarks, hand_type, w, h)  → np.ndarray (21,2)
reset_smoothing(hand_type=None)  → clear one or all buffers

Hand dict schema
----------------
{
    "type":      "Left" | "Right",   # MediaPipe handedness
    "landmarks": <list of 21 NormalizedLandmark>
}
"""

import cv2
import numpy as np
import mediapipe as mp
import os
import urllib.request
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

NUM_LANDMARKS = 21


class HandTracker:
    """
    Wraps MediaPipe HandLandmarker with support for up to 2 simultaneous
    hands and independent per-hand EMA coordinate smoothing.
    """

    def __init__(self,
                 model_path='hand_landmarker.task',
                 max_hands=2,
                 smoothing_factor=0.45):
        """
        Args:
            model_path:       Path to the .task model file.
            max_hands:        Maximum number of hands to detect (default 2).
            smoothing_factor: EMA alpha — 0 = very smooth/laggy,
                              1 = no smoothing/jittery. 0.45 is a good default.
        """
        self.model_path       = model_path
        self.max_hands        = max_hands
        self.smoothing_factor = smoothing_factor
        self._ensure_model_exists()

        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_hands,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.7,
            min_tracking_confidence=0.7,
            running_mode=vision.RunningMode.IMAGE
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

        # Independent EMA buffers keyed by hand type: "Left" / "Right"
        # Shape of each value: np.ndarray (NUM_LANDMARKS, 2) float64
        self._smoothed_px: dict[str, np.ndarray | None] = {
            "Left":  None,
            "Right": None,
        }

    # ──────────────────────────────────────────────────────────────────
    # Model management
    # ──────────────────────────────────────────────────────────────────

    def _ensure_model_exists(self):
        """Downloads the MediaPipe hand landmarker model if missing."""
        if not os.path.exists(self.model_path):
            print("Downloading hand_landmarker model…")
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/mediapipe-models/"
                "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
                self.model_path
            )

    # ──────────────────────────────────────────────────────────────────
    # Detection
    # ──────────────────────────────────────────────────────────────────

    def process_frame(self, frame):
        """Run MediaPipe detection on a BGR OpenCV frame. Returns raw results."""
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        return self.detector.detect(mp_image)

    def get_all_hands_data(self, results, frame_is_flipped=True):
        """
        Extract data for ALL detected hands.  Each entry in the returned
        list is independent — Left and Right are never merged or overwritten.

        Args:
            results:          MediaPipe HandLandmarker results.
            frame_is_flipped: Boolean indicating if the input frame was horizontally
                              flipped. If True, reverses MediaPipe's handedness labels
                              to match the user's real physical hands.

        Returns:
            List of dicts, one per detected hand:
            [
              { "type": "Left",  "landmarks": <21-lm list> },
              { "type": "Right", "landmarks": <21-lm list> },
              ...
            ]
            Returns an empty list when no hands are detected.
        """
        hands_data = []
        if not (results and results.hand_landmarks):
            return hands_data

        for i, lm_list in enumerate(results.hand_landmarks):
            hand_type = results.handedness[i][0].category_name
            
            # MediaPipe predicts handedness based on visual appearance.
            # If the user mirrors the camera feed for natural interaction,
            # a real Left hand looks like a Right hand to the model.
            if frame_is_flipped:
                hand_type = "Right" if hand_type == "Left" else "Left"

            hands_data.append({
                "type":      hand_type,
                "landmarks": lm_list,
            })

        return hands_data

    # ──────────────────────────────────────────────────────────────────
    # Per-hand smoothed pixel coordinates
    # ──────────────────────────────────────────────────────────────────

    def get_smoothed_landmarks(self, landmarks, hand_type, width, height):
        """
        Convert normalised landmarks to pixel coordinates and apply
        per-hand EMA smoothing so the two hands never interfere.

        Args:
            landmarks:  List of 21 NormalizedLandmark objects.
            hand_type:  "Left" or "Right" — selects the correct buffer.
            width:      Frame pixel width.
            height:     Frame pixel height.

        Returns:
            np.ndarray of shape (21, 2) — smoothed int32 pixel (x, y).
        """
        raw = np.array(
            [[lm.x * width, lm.y * height] for lm in landmarks],
            dtype=np.float64
        )

        buf = self._smoothed_px.get(hand_type)
        if buf is None:
            self._smoothed_px[hand_type] = raw.copy()
        else:
            a = self.smoothing_factor
            self._smoothed_px[hand_type] = a * raw + (1 - a) * buf

        return self._smoothed_px[hand_type].astype(np.int32)

    def reset_smoothing(self, hand_type=None):
        """
        Reset the EMA smoothing buffer.

        Args:
            hand_type: "Left", "Right", or None (resets both).
        """
        if hand_type is None:
            self._smoothed_px = {"Left": None, "Right": None}
        elif hand_type in self._smoothed_px:
            self._smoothed_px[hand_type] = None

    # ──────────────────────────────────────────────────────────────────
    # Backward-compatible single-hand helper (kept for legacy callers)
    # ──────────────────────────────────────────────────────────────────

    def get_first_hand_data(self, results):
        """
        Returns (landmarks, handedness_str) for the first detected hand
        only.  Prefer get_all_hands_data() for multi-hand use.
        """
        data = self.get_all_hands_data(results)
        if data:
            return data[0]["landmarks"], data[0]["type"]
        return None, None
