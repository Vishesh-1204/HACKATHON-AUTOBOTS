"""
hand_tracking.py — Multi-Hand Computer Vision Specialist
=========================================================
Supports up to 2 simultaneous hands with independent per-hand EMA
smoothing buffers so Left and Right hands never interfere.

Public API
----------
process_frame(frame)         → raw MediaPipe results
get_all_hands_data(results)  → list of hand dicts (see below)
smooth_normalized_landmarks(landmarks, hand_type)  → smooth 3D landmarks
reset_smoothing(hand_type=None)  → clear one or all buffers
"""

import cv2
import numpy as np
import mediapipe as mp
import os
import urllib.request
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

NUM_LANDMARKS = 21

class SmoothLandmark:
    """Wrapper to mimic MediaPipe's NormalizedLandmark with smooth mutable states."""
    __slots__ = ['x', 'y', 'z']
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

class HandTracker:
    """
    Wraps MediaPipe HandLandmarker with support for up to 2 simultaneous
    hands and independent per-hand EMA coordinate smoothing.
    """

    def __init__(self,
                 model_path='hand_landmarker.task',
                 max_hands=2,
                 smoothing_factor=0.6): # 0.6 for responsive but stable performance
        self.model_path       = model_path
        self.max_hands        = max_hands
        self.smoothing_factor = smoothing_factor
        self._ensure_model_exists()

        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_hands,
            min_hand_detection_confidence=0.5, # Lowered for fewer lost frames
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=vision.RunningMode.IMAGE
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

        # Independent EMA buffers keyed by hand type
        self._smoothed_norm: dict[str, np.ndarray | None] = {
            "Left":  None,
            "Right": None,
        }

    def _ensure_model_exists(self):
        if not os.path.exists(self.model_path):
            print("Downloading hand_landmarker model…")
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/mediapipe-models/"
                "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
                self.model_path
            )

    def process_frame(self, frame):
        """Run MediaPipe detection on a BGR OpenCV frame."""
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        return self.detector.detect(mp_image)

    def get_all_hands_data(self, results, frame_is_flipped=True):
        hands_data = []
        if not (results and results.hand_landmarks):
            return hands_data

        for i, lm_list in enumerate(results.hand_landmarks):
            hand_type = results.handedness[i][0].category_name
            if frame_is_flipped:
                hand_type = "Right" if hand_type == "Left" else "Left"

            hands_data.append({
                "type":      hand_type,
                "landmarks": lm_list,
            })
        return hands_data

    def smooth_normalized_landmarks(self, landmarks, hand_type):
        """Applies EMA smoothing directly to the 3D normalized coordinates to fix structural jitter."""
        raw = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float64)
        buf = self._smoothed_norm.get(hand_type)
        
        if buf is None:
            buf = raw.copy()
        else:
            alpha = self.smoothing_factor
            buf = alpha * raw + (1.0 - alpha) * buf
            
        self._smoothed_norm[hand_type] = buf
        return [SmoothLandmark(float(r[0]), float(r[1]), float(r[2])) for r in buf]

    def reset_smoothing(self, hand_type=None):
        if hand_type is None:
            self._smoothed_norm = {"Left": None, "Right": None}
        elif hand_type in self._smoothed_norm:
            self._smoothed_norm[hand_type] = None
