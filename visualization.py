"""
visualization.py — AR Ghost Hand & 5-Finger Prosthetic Renderer
==============================================================
Combines the mechanics of a robotic arm simulation with a glowing,
semi-transparent "ghost hand" overlay. Now upgraded with a 
fully articulated 5-finger robotic gripper and true 2-Segment Inverse Kinematics (IK).

Features:
- True 2-Link Inverse Kinematics for Shoulder/Elbow logic targeting the real hand.
- Full Robotic Hand: 5 independent robotic fingers mapping real bend.
- Ghost Hand: Translucent skeleton drawn on the dark background.
- Finger Influence: Wrist → Fingertip neon lines for active fingers.
- Robotic Base: Metallic structure, physically grounded.
- Auto Reset: Arm smoothly glides back to default posture on signal.
- Adaptive Physics: Zero lag for fast movements, stable for slow ones.
"""

import cv2
import numpy as np
import math

# ══════════════════════════════════════════════════════════════════════
# CONSTANTS & TOPOLOGY
# ══════════════════════════════════════════════════════════════════════

BG_COLOR          = (18, 18, 28)

# Metallic Core colors
COLOR_UPPER_ARM   = (170, 180, 190)
COLOR_FOREARM     = (140, 150, 160)
COLOR_WRIST_LINK  = (110, 120, 130)
COLOR_JOINT       = (230, 240, 250)

# Neon Glows
GLOW_COLOR_UPPER  = (20, 200, 255)
GLOW_COLOR_FORE   = (100, 255, 150)
GLOW_COLOR_WRIST  = (255, 180,  60)
COLOR_JOINT_HL    = (100, 255, 255)
COLOR_GUIDE_LINE  = (35,   45,  55)

GLOW_ALPHA        = 0.50

# Finger Palette
FINGER_COLORS = {
    'thumb':  (80, 200, 255),   # Yellow/Amber
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
    'Resetting...':  (50, 150, 200),
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

    GLOW_ON    = False  # Disabled heavy blur glow passes by default for huge FPS gains
    DEBUG_MODE = True   # Enable debug visualization for true IK vectors

    def __init__(self):
        # Default Pose configuration
        self._DEFAULT_SHOULDER = -math.pi / 2
        self._DEFAULT_ELBOW = 0.1
        self._DEFAULT_WRIST = 0.0
        
        self._shoulder_ang  = self._DEFAULT_SHOULDER
        self._elbow_bend    = self._DEFAULT_ELBOW
        self._wrist_bend    = self._DEFAULT_WRIST
        self._pulse         = 0.0
        self._gesture       = "No Hand"

        # Independent curls for the 5 robotic fingers (0=open, 1.2=curled)
        self._finger_curls  = [0.3] * 5

        # Ghost hand data
        self._hand_pts = None
        self._finger_states = [0, 0, 0, 0, 0]
        
    def update(self, landmarks, finger_states, gesture, w, h, x_fraction=0.5):
        self._pulse = (self._pulse + 0.12) % (2 * math.pi)

        if landmarks is None:
            self._hand_pts = None
            self._finger_states = [0, 0, 0, 0, 0]
            
            if gesture == "Resetting...":
                self._gesture = "Resetting..."
                # Smoothly interpolate back to default resting pose
                s = 0.15 # Fixed dampening for reset phase
                self._shoulder_ang = _lerp_angle(self._shoulder_ang, self._DEFAULT_SHOULDER, s)
                self._elbow_bend   = _lerp_angle(self._elbow_bend,   self._DEFAULT_ELBOW,    s)
                self._wrist_bend   = _lerp_angle(self._wrist_bend,   self._DEFAULT_WRIST,    s)
                for i in range(5):
                    self._finger_curls[i] = _lerp(self._finger_curls[i], 0.3, s)
            else:
                self._gesture = "No Hand"
            return

        self._gesture = gesture
        self._finger_states = finger_states

        self._hand_pts = np.array([
            [int(lm.x * w), int(lm.y * h)] for lm in landmarks
        ])

        # ─── INVERSE KINEMATICS (IK) POSITIONAL MAPPING ────────────────
        wrist     = landmarks[0]
        index_mcp = landmarks[5]

        # Authentic rendering coordinate space for the arm's base
        base_x = int(w * x_fraction)
        base_y = int(h * 0.88)

        # Target End-Effector: The real-world MediaPipe tracked wrist.
        target_x = int(wrist.x * w)
        target_y = int(wrist.y * h)

        # Translate target relative to arm base
        # To match the mathematical Cartesian plane where +Y is Up, we invert Y natively.
        dx = target_x - base_x
        dy = base_y - target_y  # Mount > Target means physically Up, mathematically Positive Y.

        # 1. REACH & WORKSPACE CONSTRAINTS
        L1 = self.SEG_UPPER
        L2 = self.SEG_FORE
        D  = math.hypot(dx, dy)
        max_reach = (L1 + L2) - 0.1  # Micro epsilon to bypass acos(1.0001) floating trap
        
        if D > max_reach:
            dx = dx * (max_reach / D)
            dy = dy * (max_reach / D)
            D = max_reach

        # 2. ELBOW BEND (IK Theta 2)
        D_sq = (dx**2) + (dy**2)
        v = (D_sq - (L1**2) - (L2**2)) / (2 * L1 * L2)
        v = max(-1.0, min(1.0, v))
        
        # Consistent solution: Always use positive elbow bend natively
        # This solves the "elbow flipping over" problem cleanly.
        raw_elbow_bend = math.acos(v)
        
        # Apply gentle kinematic lock (No broken collapsing) [0 -> 160 deg max]
        max_elb_rad = math.radians(160)
        raw_elbow_bend = min(raw_elbow_bend, max_elb_rad)

        # 3. SHOULDER ANGLE (IK Theta 1)
        ang_target = math.atan2(dy, dx)
        ang_offset = math.atan2(L2 * math.sin(raw_elbow_bend), L1 + L2 * math.cos(raw_elbow_bend))
        
        raw_shoulder = ang_target - ang_offset
        # Normalize continuous wrap range [-pi, pi]
        raw_shoulder = (raw_shoulder + math.pi) % (2 * math.pi) - math.pi

        # 4. WRIST LINK MAPPING (Maintain index direction absolute pointer)
        w_dx = (index_mcp.x * w) - target_x
        w_dy = base_y - (target_y + (index_mcp.y * h - target_y)) 
        # Simpler w_dy: screen ty - screen index_y exactly means mathematical Cartesian +y
        w_dy = target_y - (index_mcp.y * h)
        
        # Absolute pointing orientation of the user's hand
        abs_wrist_ang = math.atan2(w_dy, w_dx)
        # Convert to local relative angle of the robotic forearm
        target_wrist_bend = abs_wrist_ang - (raw_shoulder + raw_elbow_bend)
        target_wrist_bend = (target_wrist_bend + math.pi) % (2 * math.pi) - math.pi
        raw_wrist_bend = max(-0.6, min(0.6, target_wrist_bend))

        # 5. ADAPTIVE PHYSICS SMOOTHING (High impact reactivity, low tremor stability)
        diff_sh = abs((raw_shoulder - self._shoulder_ang + math.pi) % (2*math.pi) - math.pi)
        diff_el = abs(raw_elbow_bend - self._elbow_bend)
        diff_wr = abs(raw_wrist_bend - self._wrist_bend)

        def _get_alpha(diff):
            return min(0.95, max(0.20, diff * 1.5))

        self._shoulder_ang = _lerp_angle(self._shoulder_ang, raw_shoulder, _get_alpha(diff_sh))
        self._elbow_bend   = _lerp_angle(self._elbow_bend,   raw_elbow_bend, _get_alpha(diff_el))
        self._wrist_bend   = _lerp_angle(self._wrist_bend,   raw_wrist_bend, _get_alpha(diff_wr))

        # 6. FINGER CURLS (5 fingers)
        raw_finger_curls = [0.0] * 5
        for i, tip_id in enumerate(FINGERTIP_IDS):
            mcp_id = tip_id - 3
            wrist_id = 0
            
            mcp = landmarks[mcp_id]
            tip = landmarks[tip_id]
            w_lm = landmarks[wrist_id]
            
            v1_x, v1_y, v1_z = (mcp.x - w_lm.x)*w, (mcp.y - w_lm.y)*h, (mcp.z - w_lm.z)*w
            v2_x, v2_y, v2_z = (tip.x - mcp.x)*w, (tip.y - mcp.y)*h, (tip.z - mcp.z)*w
            
            dot = v1_x*v2_x + v1_y*v2_y + v1_z*v2_z
            m1 = math.sqrt(v1_x**2 + v1_y**2 + v1_z**2)
            m2 = math.sqrt(v2_x**2 + v2_y**2 + v2_z**2)
            
            if m1 * m2 > 1e-6:
                cos_a = max(-1.0, min(1.0, dot / (m1*m2)))
                ang = math.acos(cos_a)
                raw_finger_curls[i] = min(1.2, ang / 2.0)  # ~0.0 to 1.2
            
            diff_f = abs(raw_finger_curls[i] - self._finger_curls[i])
            f_alpha = min(0.85, max(0.2, diff_f * 2.0))
            self._finger_curls[i] = _lerp(self._finger_curls[i], raw_finger_curls[i], f_alpha)


    def render(self, canvas, x_fraction=0.5, hand_label=""):
        h, w = canvas.shape[:2]
        mount = (int(w * x_fraction), int(h * 0.88))

        self._draw_grid(canvas, mount)
        self._draw_shadow(canvas, mount)
        self._draw_ghost_hand(canvas)
        self._draw_arm(canvas, mount)
        self._draw_mount_base(canvas, mount)

        if hand_label:
            badge = GESTURE_BADGE_COLORS.get(self._gesture, (180, 180, 180))
            lx = mount[0] - 28
            ly = mount[1] - 40
            cv2.putText(canvas, hand_label, (lx, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, badge, 2, cv2.LINE_AA)
            
        if self.DEBUG_MODE and self._hand_pts is not None:
            self._draw_ik_debug_overlay(canvas, mount)

    # ------------------------------------------------------------------
    # Hybrid Drawing Helpers
    # ------------------------------------------------------------------

    def _draw_shadow(self, canvas, mount):
        mx, my = mount
        cv2.ellipse(canvas, (mx, my+30), (110, 20), 0, 0, 360, (10, 10, 15), -1, cv2.LINE_AA)

    def _draw_ghost_hand(self, canvas):
        if self._hand_pts is None:
            return
        pts = self._hand_pts
        if self.GLOW_ON:
            overlay = canvas.copy()
        else:
            overlay = canvas

        palm_pts = np.array([pts[i] for i in PALM_POLYGON_IDS], dtype=np.int32)
        cv2.fillConvexPoly(overlay, palm_pts, (25, 35, 45), cv2.LINE_AA)

        for fi, (name, conns) in enumerate(FINGER_CONNECTIONS.items()):
            color = FINGER_COLORS[name]
            if not self._finger_states[fi]:
                color = tuple(max(0, int(c * 0.25)) for c in color)
            for (a, b) in conns:
                cv2.line(overlay, tuple(pts[a]), tuple(pts[b]), color, 3, cv2.LINE_AA)

        for (a, b) in PALM_CONNECTIONS:
            cv2.line(overlay, tuple(pts[a]), tuple(pts[b]), (80, 80, 80), 2, cv2.LINE_AA)

        if self.GLOW_ON:
            cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0, canvas)

        wrist = tuple(pts[0])
        for fi, tip_id in enumerate(FINGERTIP_IDS):
            is_up = self._finger_states[fi]
            color = FINGER_COLORS[FINGER_NAMES[fi]]
            tip = tuple(pts[tip_id])
            if is_up:
                cv2.line(canvas, wrist, tip, color, 1, cv2.LINE_AA)
                cv2.circle(canvas, tip, 5, color, -1, cv2.LINE_AA)
            else:
                dim = tuple(max(0, int(c * 0.3)) for c in color)
                cv2.line(canvas, wrist, tip, dim, 1, cv2.LINE_AA)

    def _draw_grid(self, canvas, mount):
        h, w = canvas.shape[:2]
        mx, my = mount
        for ang_deg in range(0, 180, 20):
            ang = math.radians(ang_deg)
            end = _polar(mount, -ang, max(w, h))
            cv2.line(canvas, mount, end, COLOR_GUIDE_LINE, 1, cv2.LINE_AA)
        cv2.line(canvas, (0, my), (w, my), COLOR_GUIDE_LINE, 1, cv2.LINE_AA)

    def _draw_mount_base(self, canvas, mount):
        mx, my = mount
        pts = np.array([
            [mx - 70, my + 6], [mx + 70, my + 6],
            [mx + 50, my + 32], [mx - 50, my + 32],
        ], np.int32)
        cv2.fillPoly(canvas, [pts], (40, 45, 50), cv2.LINE_AA)
        cv2.polylines(canvas, [pts], True, (90, 100, 110), 2, cv2.LINE_AA)
        
        for bx in [mx - 45, mx - 15, mx + 15, mx + 45]:
            cv2.circle(canvas, (bx, my + 19), 3, (20, 20, 25), -1, cv2.LINE_AA)
            cv2.circle(canvas, (bx, my + 19), 4, (100, 110, 120), 1, cv2.LINE_AA)
            
        cv2.circle(canvas, mount, 24, (30, 35, 40), -1, cv2.LINE_AA)
        cv2.circle(canvas, mount, 24, (180, 190, 200),  3, cv2.LINE_AA)
        cv2.circle(canvas, mount,  8, GLOW_COLOR_UPPER, -1, cv2.LINE_AA)

    def _draw_arm(self, canvas, mount):
        shoulder_ang = self._shoulder_ang
        elbow_ang = shoulder_ang + self._elbow_bend
        wrist_ang = elbow_ang + self._wrist_bend
        
        elbow = _polar(mount, -shoulder_ang, self.SEG_UPPER)
        wrist_pt  = _polar(elbow, -elbow_ang, self.SEG_FORE)
        palm_base = _polar(wrist_pt, -wrist_ang, self.SEG_WRIST)

        badge = GESTURE_BADGE_COLORS.get(self._gesture, (180, 180, 180))

        # Glowing Neon Cores
        if self.GLOW_ON:
            glow = canvas.copy()
            cv2.line(glow, mount, elbow, GLOW_COLOR_UPPER, 34, cv2.LINE_AA)
            cv2.line(glow, elbow, wrist_pt, GLOW_COLOR_FORE, 26, cv2.LINE_AA)
            cv2.line(glow, wrist_pt, palm_base, GLOW_COLOR_WRIST, 18, cv2.LINE_AA)
            cv2.addWeighted(glow, GLOW_ALPHA, canvas, 1 - GLOW_ALPHA, 0, canvas)

        # Main Metal Structure
        cv2.line(canvas, mount, elbow,       COLOR_UPPER_ARM,  16, cv2.LINE_AA)
        cv2.line(canvas, elbow, wrist_pt,    COLOR_FOREARM,    12, cv2.LINE_AA)
        cv2.line(canvas, wrist_pt, palm_base,COLOR_WRIST_LINK, 9,  cv2.LINE_AA)

        self._draw_angle_arc(canvas, elbow, shoulder_ang, elbow_ang)

        # Primary Joints
        _draw_joint(canvas, elbow,    18, 9, (60,70,80),  COLOR_UPPER_ARM, COLOR_JOINT)
        _draw_joint(canvas, wrist_pt, 14, 7, (50,65,75),  COLOR_FOREARM,   COLOR_JOINT)
        _draw_joint(canvas, palm_base, 12, 6, (45,60,70), COLOR_WRIST_LINK,COLOR_JOINT)

        self._label_midpoint(canvas, mount, elbow, "UPPER ARM")
        self._label_midpoint(canvas, elbow, wrist_pt,  "FOREARM")
        self._label_midpoint(canvas, wrist_pt, palm_base, "WRIST")

        self._draw_arm_finger_influences(canvas, elbow, wrist_pt)

        # ── FULL 5-FINGER ROBOTIC HAND ──
        self._draw_robotic_hand(canvas, palm_base, wrist_ang, badge)

    def _draw_robotic_hand(self, canvas, base, wrist_ang, badge_color):
        spread_angles  = [0.6, 0.25, 0.0, -0.25, -0.6]
        finger_lengths = [
            [28, 20, 16],  # Thumb
            [35, 25, 20],  # Index
            [40, 30, 24],  # Middle
            [36, 26, 20],  # Ring
            [26, 18, 14],  # Pinky
        ]
        
        for idx in range(5):
            base_ang = wrist_ang + spread_angles[idx]
            curl_amount = self._finger_curls[idx]
            joint_bend = curl_amount * 1.15
            
            color = FINGER_COLORS[FINGER_NAMES[idx]]
            pulse = math.sin(self._pulse * 1.5 + idx)
            glow_c = tuple(min(255, int(c * 0.7)) for c in color)
            
            cur_pt = base
            cur_ang = base_ang
            
            for seg_i, seg_len in enumerate(finger_lengths[idx]):
                cur_ang -= joint_bend
                nxt_pt = _polar(cur_pt, -cur_ang, seg_len)
                
                thick = 10 if seg_i == 0 else 8 if seg_i == 1 else 6
                if self.GLOW_ON:
                    glow = canvas.copy()
                    cv2.line(glow, cur_pt, nxt_pt, glow_c, thick + 6, cv2.LINE_AA)
                    cv2.addWeighted(glow, 0.35, canvas, 0.65, 0, canvas)
                else:
                    cv2.line(canvas, cur_pt, nxt_pt, glow_c, thick + 4, cv2.LINE_AA)
                
                # Metallic Core
                cv2.line(canvas, cur_pt, nxt_pt, (160,170,180), thick, cv2.LINE_AA)
                cv2.circle(canvas, cur_pt, thick//2 + 1, COLOR_JOINT, -1, cv2.LINE_AA)
                cur_pt = nxt_pt
                
            # Fingertip core
            tip_r = 7
            tip_h = tuple(min(255, int(c * (0.8 + 0.2 * pulse))) for c in glow_c)
            cv2.circle(canvas, cur_pt, tip_r, tip_h, -1, cv2.LINE_AA)
            cv2.circle(canvas, cur_pt, tip_r+2, badge_color, 1, cv2.LINE_AA)

        lx, ly = base[0] - 28, base[1] + 35
        cv2.putText(canvas, self._gesture, (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, badge_color, 1, cv2.LINE_AA)

    def _draw_arm_finger_influences(self, canvas, elbow, wrist_pt):
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

    def _draw_angle_arc(self, canvas, elbow, ang1, ang2):
        arc_r = 40
        start_deg = int(math.degrees(-ang1)) % 360
        end_deg   = int(math.degrees(-ang2)) % 360
        if start_deg == end_deg: return
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
                    
    def _draw_ik_debug_overlay(self, canvas, mount):
        mx, my = mount
        tx = self._hand_pts[0][0]
        ty = self._hand_pts[0][1]
        
        cv2.line(canvas, mount, (tx, ty), (50, 255, 50), 1, cv2.LINE_AA)
        cv2.circle(canvas, (tx, ty), 4, (50, 255, 50), -1)
        
        ik_text = f"IK -> SHOULDER: {math.degrees(self._shoulder_ang):.1f} | ELBOW: {math.degrees(self._elbow_bend):.1f}"
        cv2.putText(canvas, ik_text, (mx - 150, my - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 100), 1, cv2.LINE_AA)
