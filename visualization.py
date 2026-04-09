"""
visualization.py — Hybrid Robotic Arm & AR Ghost Hand Renderer
==============================================================
Combines the mechanics of a robotic arm simulation with a glowing,
semi-transparent "ghost hand" overlay.

Features:
- Ghost Hand: Translucent skeleton drawn on the dark background.
- Finger Influence: Wrist → Fingertip neon lines for active fingers.
- Robotic Arm: Metallic structure, grounded physically, with neon glow.
- Connection Logic: Index finger distance maps to elbow bend, tilt to shoulder.
"""

import cv2
import numpy as np
import math

# ══════════════════════════════════════════════════════════════════════
# CONSTANTS & TOPOLOGY
# ══════════════════════════════════════════════════════════════════════

BG_COLOR          = (18, 18, 28)

# Robotic Arm Colors - Metallic Core
COLOR_UPPER_ARM   = (170, 180, 190)    # Metallic Silver
COLOR_FOREARM     = (140, 150, 160)    # Outer Gunmetal
COLOR_WRIST_LINK  = (110, 120, 130)    # Dark Steel
COLOR_GRIPPER_L   = (160, 170, 180)
COLOR_GRIPPER_R   = (160, 170, 180)
COLOR_JOINT       = (230, 240, 250)

# Robotic Arm Colors - Neon Glows
GLOW_COLOR_UPPER  = (20, 200, 255)     # Cyan
GLOW_COLOR_FORE   = (100, 255, 150)    # Neon Green
GLOW_COLOR_WRIST  = (255, 180,  60)    # Amber Glow
GLOW_COLOR_GRIPL  = (200,  80, 255)
GLOW_COLOR_GRIPR  = (255,  80, 160)
COLOR_JOINT_HL    = (100, 255, 255)
COLOR_GUIDE_LINE  = (35,   45,  55)

GLOW_ALPHA        = 0.50               # Transparent bleed for the glow

# Ghost Hand Colors
FINGER_COLORS = {
    'thumb':  (80, 200, 255),   # Yellow / Amber
    'index':  (80, 255, 80),    # Green
    'middle': (255, 200, 60),   # Blue
    'ring':   (180, 80, 255),   # Pink
    'pinky':  (60, 220, 255),   # Cyan
}
FINGER_NAMES = list(FINGER_COLORS.keys())

FINGER_CONNECTIONS = {
    'thumb':  [(0, 1), (1, 2), (2, 3), (3, 4)],
    'index':  [(0, 5), (5, 6), (6, 7), (7, 8)],
    'middle': [(0, 9), (9, 10), (10, 11), (11, 12)],
    'ring':   [(0, 13), (13, 14), (14, 15), (15, 16)],
    'pinky':  [(0, 17), (17, 18), (18, 19), (19, 20)],
}
PALM_CONNECTIONS = [(5, 9), (9, 13), (13, 17), (0, 17), (0, 5)]
FINGERTIP_IDS = [4, 8, 12, 16, 20]
PALM_POLYGON_IDS = [0, 1, 5, 9, 13, 17]


GESTURE_BADGE_COLORS = {
    'Move (Open)':   (80,  255, 100),
    'Stop (Fist)':   (50,   50, 240),
    'Grab (Pinch)':  (50,  230, 230),
    'Point':         (240, 200,  50),
    'Victory':       (220,  80, 240),
    'No Hand':       (100, 100, 100),
}


# ══════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════

def _lerp(a, b, t):
    return a + (b - a) * t

def _lerp_angle(a, b, t):
    diff = (b - a + math.pi) % (2 * math.pi) - math.pi
    return a + diff * t

def _polar(origin, angle, length):
    ox, oy = origin
    return (int(ox + length * math.cos(angle)),
            int(oy + length * math.sin(angle)))

def _draw_joint(canvas, pt, outer_r, inner_r, outer_color, inner_color, ring_color):
    cv2.circle(canvas, pt, outer_r, outer_color, -1, cv2.LINE_AA)
    cv2.circle(canvas, pt, inner_r, inner_color, -1, cv2.LINE_AA)
    cv2.circle(canvas, pt, outer_r, ring_color,   2, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════
# VISUALIZER
# ══════════════════════════════════════════════════════════════════════

class RoboticArmVisualizer:
    SEG_UPPER  = 190
    SEG_FORE   = 150
    SEG_WRIST  = 65
    GRIP_LEN   = 60

    SMOOTH     = 0.08   # Heavy smoothing for physical mass realism

    def __init__(self):
        self._shoulder_ang  = -math.pi / 2
        self._elbow_bend    = math.pi / 4
        self._wrist_bend    = 0.0
        self._gripper_open  = 1.0

        self._pulse = 0.0
        self._gesture = "No Hand"

        # Ghost hand state
        self._hand_pts = None
        self._finger_states = [0, 0, 0, 0, 0]
        
    def update(self, landmarks, finger_states, gesture, w, h):
        self._gesture = gesture
        self._pulse   = (self._pulse + 0.12) % (2 * math.pi)

        if landmarks is None:
            self._hand_pts = None
            self._finger_states = [0, 0, 0, 0, 0]
            return

        self._finger_states = finger_states

        # Scale raw landmarks to screen coordinates
        self._hand_pts = np.array([
            [int(lm.x * w), int(lm.y * h)] for lm in landmarks
        ])

        if gesture == "Stop (Fist)":
            # Physically lock the arm joints in place, ignore landmarks
            pass
        else:
            # Kinematics mapping
            wrist     = landmarks[0]
            index_mcp = landmarks[5]
            index_tip = landmarks[8]
            middle_mcp = landmarks[9]

            # 1. SHOULDER (Orientation) -- mapped to hand tilt (wrist to index MCP)
            dx_arm = (index_mcp.x - wrist.x)
            dy_arm = (index_mcp.y - wrist.y)
            raw_shoulder = math.atan2(-dy_arm, dx_arm)
            
            # Constrain to upper hemisphere (to avoid arm dropping through the floor)
            raw_shoulder = max(-math.pi * 0.95, min(-math.pi * 0.05, raw_shoulder))

            # 2. ELBOW BEND -- mapped to the stretch of the index finger from the wrist
            # (Provides very clear, physically intuitive mapping to the arm extension)
            dist = math.hypot(index_tip.x - wrist.x, index_tip.y - wrist.y)
            # Normalised span is typically 0.1 (fist) to 0.5 (open hand)
            dist_clamped = max(0.1, min(0.6, dist))
            normalized_bend = 1.0 - ((dist_clamped - 0.1) / 0.5)
            
            # Angle bending limits (0 = fully extended, 0.8*pi = heavily curled)
            raw_elbow_bend = normalized_bend * (math.pi * 0.8)

            # 3. WRIST LINK
            dx_wr = (middle_mcp.x - index_mcp.x)
            dy_wr = (middle_mcp.y - index_mcp.y)
            raw_wrist_bend = math.atan2(-dy_wr, dx_wr) * 0.4
            
            # Constrain wrist sideways wobble
            raw_wrist_bend = max(-0.4, min(0.4, raw_wrist_bend))

            # Stabilize tiny jitters (deadzone)
            if abs(raw_shoulder - self._shoulder_ang) < 0.02: raw_shoulder = self._shoulder_ang
            if abs(raw_elbow_bend - self._elbow_bend) < 0.03: raw_elbow_bend = self._elbow_bend
            if abs(raw_wrist_bend - self._wrist_bend) < 0.03: raw_wrist_bend = self._wrist_bend

            # EMA smoothing
            s = self.SMOOTH
            self._shoulder_ang = _lerp_angle(self._shoulder_ang, raw_shoulder,    s)
            self._elbow_bend   = _lerp_angle(self._elbow_bend,   raw_elbow_bend,  s)
            self._wrist_bend   = _lerp_angle(self._wrist_bend,   raw_wrist_bend,  s)

        # Update Gripper State
        if gesture == "Grab (Pinch)":
            raw_gripper = 0.0
        elif gesture == "Stop (Fist)":
            raw_gripper = 0.15 # Gripper locks semi-closed
        elif gesture in ("Move (Open)", "Victory"):
            raw_gripper = 1.0
        else:
            raw_gripper = 0.5   
            
        self._gripper_open = _lerp(self._gripper_open, raw_gripper, self.SMOOTH * 1.5)

    def render(self, canvas, x_fraction=0.5, hand_label=""):
        h, w = canvas.shape[:2]
        # Ground anchor placed fixed at the bottom 
        mount = (int(w * x_fraction), int(h * 0.88))

        self._draw_grid(canvas, mount)
        self._draw_shadow(canvas, mount)
        
        # 1. Background layer: Ghost Hand
        self._draw_ghost_hand(canvas)
        
        # 2. Foreground layer: Robotic Arm
        self._draw_arm(canvas, mount)
        self._draw_mount_base(canvas, mount)

        if hand_label:
            badge = GESTURE_BADGE_COLORS.get(self._gesture, (180, 180, 180))
            lx = mount[0] - 28
            ly = mount[1] - 40
            cv2.putText(canvas, hand_label, (lx, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, badge, 2, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # Hybrid Drawing Helpers
    # ------------------------------------------------------------------

    def _draw_shadow(self, canvas, mount):
        """Draws an elliptical floor shadow under the base to ground it."""
        mx, my = mount
        cv2.ellipse(canvas, (mx, my+30), (110, 20), 0, 0, 360, (10, 10, 15), -1, cv2.LINE_AA)

    def _draw_ghost_hand(self, canvas):
        """Translucent ghost hand outline and finger influence lines."""
        if self._hand_pts is None:
            return

        pts = self._hand_pts
        overlay = canvas.copy()

        # Palm Mesh
        palm_pts = np.array([pts[i] for i in PALM_POLYGON_IDS], dtype=np.int32)
        cv2.fillConvexPoly(overlay, palm_pts, (30, 40, 50), cv2.LINE_AA)

        # Hand bones with dimming
        for fi, (name, conns) in enumerate(FINGER_CONNECTIONS.items()):
            color = FINGER_COLORS[name]
            if not self._finger_states[fi]:
                color = tuple(max(0, int(c * 0.25)) for c in color)
            for (a, b) in conns:
                cv2.line(overlay, tuple(pts[a]), tuple(pts[b]), color, 3, cv2.LINE_AA)

        for (a, b) in PALM_CONNECTIONS:
            cv2.line(overlay, tuple(pts[a]), tuple(pts[b]), (100, 100, 100), 2, cv2.LINE_AA)

        # Blend ghost skeleton
        cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0, canvas)

        # ── Draw Influence Lines (Wrist → Fingertips) ──
        wrist = tuple(pts[0])
        for fi, tip_id in enumerate(FINGERTIP_IDS):
            is_up = self._finger_states[fi]
            color = FINGER_COLORS[FINGER_NAMES[fi]]
            tip = tuple(pts[tip_id])
            if is_up:
                cv2.line(canvas, wrist, tip, color, 1, cv2.LINE_AA)
                cv2.circle(canvas, tip, 5, color, -1, cv2.LINE_AA)
            else:
                dim_color = tuple(max(0, int(c * 0.3)) for c in color)
                cv2.line(canvas, wrist, tip, dim_color, 1, cv2.LINE_AA)
                
        # Index visual arm tracker vector
        if self._finger_states[1]: # Index is out
            idx_mcp = tuple(pts[5])
            idx_tip = tuple(pts[8])
            vec_x = idx_tip[0] - idx_mcp[0]
            vec_y = idx_tip[1] - idx_mcp[1]
            vec_len = math.hypot(vec_x, vec_y)
            if vec_len > 0:
                extend_x = int(idx_tip[0] + (vec_x / vec_len) * 70)
                extend_y = int(idx_tip[1] + (vec_y / vec_len) * 70)
                cv2.line(canvas, idx_tip, (extend_x, extend_y), FINGER_COLORS['index'], 2, cv2.LINE_AA)

    def _draw_grid(self, canvas, mount):
        h, w = canvas.shape[:2]
        mx, my = mount
        for ang_deg in range(0, 180, 20):
            ang = math.radians(ang_deg)
            end = _polar(mount, -ang, max(w, h))
            cv2.line(canvas, mount, end, COLOR_GUIDE_LINE, 1, cv2.LINE_AA)
        cv2.line(canvas, (0, my), (w, my), COLOR_GUIDE_LINE, 1, cv2.LINE_AA)

    def _draw_mount_base(self, canvas, mount):
        """Reinforced, heavy physical anchor for the robotic operations."""
        mx, my = mount
        # Heavy trapezoid base
        pts = np.array([
            [mx - 70, my + 6],
            [mx + 70, my + 6],
            [mx + 50, my + 32],
            [mx - 50, my + 32],
        ], np.int32)
        cv2.fillPoly(canvas, [pts], (40, 45, 50), cv2.LINE_AA)
        cv2.polylines(canvas, [pts], True, (90, 100, 110), 2, cv2.LINE_AA)
        
        # Bolts / Rivets
        for bx in [mx - 45, mx - 15, mx + 15, mx + 45]:
            cv2.circle(canvas, (bx, my + 19), 3, (20, 20, 25), -1, cv2.LINE_AA)
            cv2.circle(canvas, (bx, my + 19), 4, (100, 110, 120), 1, cv2.LINE_AA)
            
        # Main Axle Joint Anchor
        cv2.circle(canvas, mount, 24, (30, 35, 40), -1, cv2.LINE_AA)
        cv2.circle(canvas, mount, 24, (180, 190, 200),  3, cv2.LINE_AA)
        cv2.circle(canvas, mount,  8, GLOW_COLOR_UPPER, -1, cv2.LINE_AA)

    def _draw_arm(self, canvas, mount):
        pulse_val = 0.5 + 0.5 * math.sin(self._pulse)

        shoulder = mount
        shoulder_ang = self._shoulder_ang
        elbow = _polar(shoulder, -shoulder_ang, self.SEG_UPPER)
        elbow_ang = shoulder_ang + self._elbow_bend
        wrist_ang = elbow_ang + self._wrist_bend
        wrist_pt  = _polar(elbow, -elbow_ang, self.SEG_FORE)
        grip_base = _polar(wrist_pt, -wrist_ang, self.SEG_WRIST)

        badge = GESTURE_BADGE_COLORS.get(self._gesture, (180, 180, 180))

        # ── Vivid Neon Power Core Glows ──
        # Drawn extremely soft and translucent under the metallic struts
        glow_layer = canvas.copy()
        cv2.line(glow_layer, shoulder, elbow, GLOW_COLOR_UPPER, 34, cv2.LINE_AA)
        cv2.line(glow_layer, elbow, wrist_pt, GLOW_COLOR_FORE, 26, cv2.LINE_AA)
        cv2.line(glow_layer, wrist_pt, grip_base, GLOW_COLOR_WRIST, 18, cv2.LINE_AA)
        cv2.addWeighted(glow_layer, GLOW_ALPHA, canvas, 1 - GLOW_ALPHA, 0, canvas)

        # ── Metallic Solid Geometry ──
        # Very thick, heavy core structure
        cv2.line(canvas, shoulder, elbow,    COLOR_UPPER_ARM,  16, cv2.LINE_AA)
        cv2.line(canvas, elbow,    wrist_pt, COLOR_FOREARM,    12, cv2.LINE_AA)
        cv2.line(canvas, wrist_pt, grip_base,COLOR_WRIST_LINK, 9,  cv2.LINE_AA)

        self._draw_angle_arc(canvas, elbow, shoulder_ang, elbow_ang)

        # ── Mechanical Joints ──
        _draw_joint(canvas, elbow,    18, 9, (60,70,80),  COLOR_UPPER_ARM, COLOR_JOINT)
        _draw_joint(canvas, wrist_pt, 14, 7, (50,65,75),  COLOR_FOREARM,   COLOR_JOINT)
        _draw_joint(canvas, grip_base, 12, 6, (45,60,70), COLOR_WRIST_LINK,COLOR_JOINT)

        self._draw_gripper(canvas, grip_base, wrist_ang, pulse_val, badge)

        self._label_midpoint(canvas, shoulder, elbow, "UPPER ARM")
        self._label_midpoint(canvas, elbow, wrist_pt,  "FOREARM")
        self._label_midpoint(canvas, wrist_pt, grip_base, "WRIST")

        # ── Render active finger states floating near forearm ──
        self._draw_arm_finger_influences(canvas, elbow, wrist_pt)

    def _draw_arm_finger_influences(self, canvas, elbow, wrist_pt):
        """Displays small, glowing visual state nodes along the forearm."""
        mid_x = (elbow[0] + wrist_pt[0]) // 2
        mid_y = (elbow[1] + wrist_pt[1]) // 2
        
        offset_y = mid_y - 35
        
        for i, name in enumerate(FINGER_NAMES):
            state = self._finger_states[i]
            x_pos = mid_x - 30 + (i * 15)
            color = FINGER_COLORS[name] if state else (50, 50, 50)
            
            cv2.circle(canvas, (x_pos, offset_y), 4, color, -1, cv2.LINE_AA)
            if state:
                cv2.circle(canvas, (x_pos, offset_y), 7, color, 1, cv2.LINE_AA)

    def _draw_gripper(self, canvas, base, wrist_ang, pulse_val, badge_color):
        max_spread = math.radians(45)
        spread = max_spread * self._gripper_open

        for side, color, glow_color in [(-1, COLOR_GRIPPER_L, GLOW_COLOR_GRIPL), (1, COLOR_GRIPPER_R, GLOW_COLOR_GRIPR)]:
            finger_ang = wrist_ang + side * spread
            tip = _polar(base, -finger_ang, self.GRIP_LEN)
            
            # Thick core with glow overlay
            glow = canvas.copy()
            cv2.line(glow, base, tip, glow_color, 16, cv2.LINE_AA)
            cv2.addWeighted(glow, 0.4, canvas, 0.6, 0, canvas)
            cv2.line(canvas, base, tip, color, 8, cv2.LINE_AA)

            tip_r = 10 if self._gripper_open < 0.2 else 8
            tip_col = tuple(min(255, int(c * (0.7 + 0.3 * pulse_val))) for c in glow_color)
            cv2.circle(canvas, tip, tip_r, tip_col, -1, cv2.LINE_AA)
            cv2.circle(canvas, tip, tip_r + 2, badge_color, 1, cv2.LINE_AA)

        status = "LOCKED" if self._gripper_open < 0.25 else \
                 "RELEASED" if self._gripper_open > 0.75 else "PARTIAL"
        lx, ly = base[0] - 28, base[1] + 25
        cv2.putText(canvas, status, (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, badge_color, 1, cv2.LINE_AA)

    def _draw_angle_arc(self, canvas, elbow, ang1, ang2):
        arc_r = 40
        start_deg = int(math.degrees(-ang1)) % 360
        end_deg   = int(math.degrees(-ang2)) % 360
        if start_deg == end_deg:
            return
        cv2.ellipse(canvas, elbow, (arc_r, arc_r), 0,
                    min(start_deg, end_deg), max(start_deg, end_deg),
                    (100, 150, 200), 2, cv2.LINE_AA)
        bend_deg = abs(int(math.degrees(ang2 - ang1))) % 360
        cv2.putText(canvas, f"{bend_deg}\u00b0 Bend", (elbow[0]+45, elbow[1]-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 200, 250), 1, cv2.LINE_AA)

    def _label_midpoint(self, canvas, p1, p2, text):
        mx = (p1[0] + p2[0]) // 2 + 15
        my = (p1[1] + p2[1]) // 2 - 15
        cv2.putText(canvas, text, (mx, my),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 120, 140), 1, cv2.LINE_AA)
