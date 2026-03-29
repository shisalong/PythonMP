import sys
import time
import math
import threading
from collections import deque

from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, pyqtSlot, QRect
from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor, QFont

import cv2
import numpy as np
import mediapipe as mp


# ============================================================
#  Constants — sourced from Impairment_Logic_Spec.md
# ============================================================
ISOLATION_INDEX: float = 0.4  # WTA crosstalk gate  (spec: MIN_ISOLATION_INDEX = 0.4)
COOLDOWN_PERIOD: float = 2.0  # seconds             (spec: COOLDOWN_PERIOD = 2 000 ms)
VEL_PEAK_DROP_RATIO: float = 0.50  # fire once velocity falls to ≤50 % of its peak
DWELL_CONFIRM_MS: int = 200  # hold the Peak position for 200ms before firing

# Noise floor to suppress RGB-only "micro jitter" / MediaPipe hallucination effects.
# Spec: 3.1mm ~ roughly 10-15 normalized pixels (camera/hand-size dependent).
NOISE_FLOOR_PX: float = 12.5
DEFAULT_PEAK_POS_TOL_PX: float = NOISE_FLOOR_PX * 0.4

JUMP_TOL_PX: float = 10.0
FLICK_CONFIRM_FRAMES: int = 5

# ============================================================
#  Thumb Shift (Layer) — Thumb(4) + Index(8) pinch
# ============================================================
PINCH_THRESHOLD_MM: float = 20.0
# Wrist->middle MCP length is roughly ~90mm in adult hands; used to convert 15mm
# into a scale-free ratio in image-space.
HAND_SIZE_MM_REFERENCE: float = 90.0
PINCH_THRESHOLD_RATIO: float = PINCH_THRESHOLD_MM / HAND_SIZE_MM_REFERENCE

# MediaPipe landmark IDs for each finger tip
FINGER_TIP_IDS: dict[str, int] = {
    "index": 8,
    "middle": 12,
    "ring": 16,  # collapsed into ring_pinky unit before WTA
    "pinky": 20,  # collapsed into ring_pinky unit before WTA
}

# Human-readable command bound to each finger unit (layered)
COMMAND_MAP_NORMAL: dict[str, str] = {
    "index": "Light",
    "middle": "Scene",
    "ring_pinky": "Alarm",
}
COMMAND_MAP_ADVANCED: dict[str, str] = {
    "middle": "AC",
    "ring_pinky": "TV",
}

# Overlay colours (BGR for OpenCV) per unit
UNIT_COLORS_BGR: dict[str, tuple] = {
    "index": (244, 133, 66),  # blue
    "middle": (53, 67, 234),  # red
    "ring_pinky": (0, 167, 255),  # amber
}


# ============================================================
#  One Euro Filter
# ============================================================
class OneEuroFilter:
    """Low-latency adaptive low-pass filter (Géry et al., 2012)."""

    def __init__(
        self,
        freq: float = 60.0,
        min_cutoff: float = 0.5,
        beta: float = 0.05,
        d_cutoff: float = 1.0,
    ):
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = None

    def _alpha(self, cutoff: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te = 1.0 / self.freq
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x: float) -> float:
        if self.x_prev is None:
            self.x_prev = x
            self.dx_prev = 0.0
            return x
        dx = (x - self.x_prev) * self.freq
        a_d = self._alpha(self.d_cutoff)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff)
        x_hat = a * x + (1.0 - a) * self.x_prev
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat


# ============================================================
#  FingerUnit — per-unit state, filtering, and velocity FSM
# ============================================================
class FingerUnit:
    """
    Encapsulates One Euro–filtered position + velocity for one finger unit.

    Velocity-peak detection with dwell confirmation + 5-frame stability gating
    ─────────────────────────────────────────────────
    IDLE   → RISING  when velocity ≥ vel_threshold
    RISING → DWELL   when velocity drops to ≤ VEL_PEAK_DROP_RATIO × local peak
                     (peak position latched during RISING)
    DWELL  → IDLE    fires is_velocity_peak() = True exactly once per flick
                     when position stays within the calibrated peak tolerance for
                     DWELL_CONFIRM_MS and 5 frames are stable without >10px jumps
    """

    _IDLE = 0
    _RISING = 1
    _DWELL = 2

    def __init__(self, name: str, freq: float = 30.0):
        self.name = name
        self.freq = freq
        # Separate filters: tighter cutoff for position, looser for velocity
        # Tremor tuning: stabilize resting jitter while still allowing fast flicks.
        self._pos_filt = OneEuroFilter(freq=freq, min_cutoff=0.1, beta=0.01)
        self._vel_filt = OneEuroFilter(freq=freq, min_cutoff=2.0, beta=0.0)
        self._prev_pos: float | None = None
        self.velocity: float = 0.0
        self.position: float = 0.0
        # Velocity FSM
        self._fsm_state: int = self._IDLE
        self._local_peak_vel: float = 0.0
        self._peak_pos: float = 0.0
        # Set from calibration: allowable spread around the latched peak.
        self.peak_pos_tol_px: float = DEFAULT_PEAK_POS_TOL_PX
        self._dwell_start_t: float | None = None
        self._prev_dwell_pos: float = 0.0
        self._dwell_pos_buf: deque[float] = deque(maxlen=FLICK_CONFIRM_FRAMES)
        # Tracks max velocity seen during calibration flick
        self.calib_peak_vel: float = 0.0

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        """Hard-reset between calibration phases."""
        self._prev_pos = None
        self.velocity = 0.0
        self.position = 0.0
        self._fsm_state = self._IDLE
        self._local_peak_vel = 0.0
        self._peak_pos = 0.0
        self._dwell_start_t = None
        self._prev_dwell_pos = 0.0
        self._dwell_pos_buf.clear()
        self.calib_peak_vel = 0.0

    def set_peak_pos_tol_px(self, tol_px: float) -> None:
        """Sets allowable peak stability tolerance in pixels (image-space)."""
        self.peak_pos_tol_px = max(0.0, float(tol_px))

    @property
    def is_dwelling(self) -> bool:
        """True while the unit is in the DWELL confirmation window."""
        return self._fsm_state == self._DWELL

    @property
    def fsm_state_name(self) -> str:
        """Human-readable FSM state for debug output."""
        return {self._IDLE: "IDLE", self._RISING: "RISING", self._DWELL: "DWELL"}.get(
            self._fsm_state, "UNKNOWN"
        )

    # ------------------------------------------------------------------ #
    def update(
        self, raw_dist: float, resting_anchor: float | None = None
    ) -> tuple[float, float]:
        """
        Feed one frame sample.
        Returns (filtered_dist, filtered_velocity).
        """
        # Hard-noise-floor: if displacement is within the MediaPipe "hallucination"
        # micro-jitter band, clamp to resting anchor to prevent squeezing/depth jitter.
        if resting_anchor is not None and raw_dist <= NOISE_FLOOR_PX:
            raw_dist = resting_anchor
        fd = self._pos_filt(raw_dist)
        self.position = fd
        if self._prev_pos is None:
            self._prev_pos = fd
            return fd, 0.0
        raw_vel = (fd - self._prev_pos) * self.freq
        fv = self._vel_filt(raw_vel)
        self._prev_pos = fd
        self.velocity = fv
        return fd, fv

    # ------------------------------------------------------------------ #
    def is_velocity_peak(self, vel_threshold: float) -> bool:
        """
        Must be called once per frame, after update().
        Returns True exactly once per kinetic flick — after position dwell
        confirmation at the latched peak position.
        """
        v = self.velocity
        if self._fsm_state == self._IDLE:
            if v >= vel_threshold:
                self._fsm_state = self._RISING
                self._local_peak_vel = v
                self._peak_pos = self.position
                self._dwell_start_t = None
        elif self._fsm_state == self._RISING:
            # Track rising velocity peak and peak position.
            if v > self._local_peak_vel:
                self._local_peak_vel = v
                self._peak_pos = self.position
            # Velocity has fallen back to ≤ 50 % of peak → start dwell.
            elif v <= self._local_peak_vel * VEL_PEAK_DROP_RATIO:
                self._fsm_state = self._DWELL
                self._dwell_start_t = time.perf_counter()
                # Seed dwell tracking so the first-frame jump-check uses a
                # valid reference (previously these were only set inside the
                # now-removed `if _dwell_start_t is None` guard, which was
                # unreachable because _dwell_start_t is always set here first).
                self._dwell_pos_buf.clear()
                self._prev_dwell_pos = self.position
                print(
                    f"[FSM] {self.name.upper()}: RISING → DWELLING "
                    f"(peak_vel={self._local_peak_vel:.1f}, peak_pos={self._peak_pos:.1f})"
                )
        elif self._fsm_state == self._DWELL:
            # Landmark jumping: abort if frame-to-frame displacement is too large.
            if abs(self.position - self._prev_dwell_pos) > JUMP_TOL_PX:
                print(
                    f"[FSM] {self.name.upper()}: DWELL ABORTED — jump "
                    f"{abs(self.position - self._prev_dwell_pos):.1f}px > {JUMP_TOL_PX}px"
                )
                self._fsm_state = self._IDLE
                self._local_peak_vel = 0.0
                self._peak_pos = 0.0
                self._dwell_start_t = None
                self._dwell_pos_buf.clear()
                return False

            self._prev_dwell_pos = self.position
            self._dwell_pos_buf.append(self.position)

            # If position keeps climbing, update the latched peak and restart dwell.
            if self.position > self._peak_pos + self.peak_pos_tol_px:
                self._peak_pos = self.position
                self._dwell_start_t = time.perf_counter()
                self._dwell_pos_buf.clear()
                self._prev_dwell_pos = self.position
                print(
                    f"[FSM] {self.name.upper()}: DWELL peak updated → {self._peak_pos:.1f}px"
                )
                return False

            # Hold position near the peak for long enough and confirm stability
            # over 5 frames to reduce tremor-related ghost triggers.
            if abs(self.position - self._peak_pos) <= self.peak_pos_tol_px:
                elapsed_ms = (time.perf_counter() - self._dwell_start_t) * 1000.0
                print(
                    f"[FSM] {self.name.upper()}: DWELLING "
                    f"{elapsed_ms:.0f}/{DWELL_CONFIRM_MS}ms  "
                    f"frames={len(self._dwell_pos_buf)}/{FLICK_CONFIRM_FRAMES}"
                )
                if (
                    elapsed_ms >= DWELL_CONFIRM_MS
                    and len(self._dwell_pos_buf) >= FLICK_CONFIRM_FRAMES
                ):
                    spread = max(self._dwell_pos_buf) - min(self._dwell_pos_buf)
                    if spread <= JUMP_TOL_PX:
                        print(
                            f"[FSM] {self.name.upper()}: DWELL → FIRED "
                            f"(spread={spread:.1f}px)"
                        )
                        self._fsm_state = self._IDLE
                        self._local_peak_vel = 0.0
                        self._peak_pos = 0.0
                        self._dwell_start_t = None
                        self._dwell_pos_buf.clear()
                        return True
                    else:
                        print(
                            f"[FSM] {self.name.upper()}: DWELL spread too large "
                            f"({spread:.1f}px) — not firing"
                        )
            else:
                print(
                    f"[FSM] {self.name.upper()}: DWELL pos={self.position:.1f} "
                    f"too far from peak={self._peak_pos:.1f} "
                    f"(tol={self.peak_pos_tol_px:.1f}px)"
                )
        return False


# ============================================================
#  Worker Thread
# ============================================================
class VideoThread(QThread):
    change_pixmap = pyqtSignal(np.ndarray)
    # Emits: {"index": 0.0-1.0, "middle": 0.0-1.0, "ring_pinky": 0.0-1.0}
    progress_update = pyqtSignal(object)
    # Now carries the command string so the UI can label the action
    trigger_activation = pyqtSignal(str)
    # Emits whether the Thumb-Index pinch (Shift layer) is active
    layer_active_update = pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        self.running = True
        self.cooldown = False
        # Calibration state
        self.calibrating = False
        self.calib_step = 0
        self.calib_results: list[float] = []
        self.calib_vel_results: list[float] = []
        # Calibrated anchors
        self.resting_anchor: float | None = None
        self.max_flick: float | None = None
        self.vel_threshold: float | None = None  # derived during calib step 2
        self._layer_is_advanced: bool = False
        # Pinch layer hysteresis: require consecutive frames to avoid flicker.
        self._pinch_true_streak: int = 0
        self._pinch_false_streak: int = 0
        # Locked winner: set as soon as any unit enters DWELL so that WTA
        # cannot steal the winner slot during the 200 ms confirmation window.
        self._locked_winner: str | None = None
        self._pos_dwell_start: dict[str, float | None] = {
            "index": None,
            "middle": None,
            "ring_pinky": None,
        }

        self._mp_hands = mp.solutions.hands
        self.hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.5,
        )
        self.cap = cv2.VideoCapture(0)

        # One FingerUnit per WTA unit (ring + pinky already merged)
        self.finger_units: dict[str, FingerUnit] = {
            "index": FingerUnit("index"),
            "middle": FingerUnit("middle"),
            "ring_pinky": FingerUnit("ring_pinky"),
        }

        # Hidden units so ring_pinky velocity can be the max() of ring/pinky.
        # (WTA still uses the merged ring_pinky displacement unit.)
        self._ring_unit = FingerUnit("ring")
        self._pinky_unit = FingerUnit("pinky")

    # ------------------------------------------------------------------ #
    def run(self) -> None:
        while self.running:
            frame_t0 = time.perf_counter()
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)
            overlay = frame_rgb.copy()
            progress: dict[str, float] = {
                "index": 0.0,
                "middle": 0.0,
                "ring_pinky": 0.0,
            }
            index_percent = 0.0

            if results.multi_hand_landmarks:
                ldmk = results.multi_hand_landmarks[0]
                h, w, _ = frame.shape

                wrist = ldmk.landmark[0]
                wx, wy = wrist.x * w, wrist.y * h

                # ── Thumb Shift Layer (Thumb(4) + Index(8) pinch) ─────────
                thumb = ldmk.landmark[4]
                index_tip = ldmk.landmark[8]
                tx_thumb, ty_thumb = thumb.x * w, thumb.y * h
                tx_index, ty_index = index_tip.x * w, index_tip.y * h
                pinch_dist_px = float(
                    np.linalg.norm([tx_thumb - tx_index, ty_thumb - ty_index])
                )

                # Normalize to hand size using wrist (0) -> middle MCP (9).
                middle_mcp = ldmk.landmark[9]
                tx_mcp, ty_mcp = middle_mcp.x * w, middle_mcp.y * h
                hand_size_px = float(np.linalg.norm([tx_mcp - wx, ty_mcp - wy]))
                pinch_ratio = pinch_dist_px / max(1e-6, hand_size_px)

                is_pinched_raw = pinch_ratio < PINCH_THRESHOLD_RATIO
                if is_pinched_raw:
                    self._pinch_true_streak += 1
                    self._pinch_false_streak = 0
                else:
                    self._pinch_false_streak += 1
                    self._pinch_true_streak = 0

                # Pinch layer hysteresis: global state independent of
                # which finger unit is moving (5-frame confirmation).
                if (not self._layer_is_advanced) and (
                    self._pinch_true_streak >= FLICK_CONFIRM_FRAMES
                ):
                    self._layer_is_advanced = True
                    if "index" in self.finger_units:
                        self.finger_units["index"].reset()
                    self.layer_active_update.emit(True)
                elif self._layer_is_advanced and (
                    self._pinch_false_streak >= FLICK_CONFIRM_FRAMES
                ):
                    self._layer_is_advanced = False
                    if "index" in self.finger_units:
                        self.finger_units["index"].reset()
                    self.layer_active_update.emit(False)

                layer_map = (
                    COMMAND_MAP_ADVANCED
                    if self._layer_is_advanced
                    else COMMAND_MAP_NORMAL
                )

                # ── Step 1: raw displacements from wrist ──────────────────
                raw_dists: dict[str, float] = {}
                tip_coords: dict[str, tuple[float, float]] = {}
                for name, tip_id in FINGER_TIP_IDS.items():
                    tip = ldmk.landmark[tip_id]
                    tx, ty = tip.x * w, tip.y * h
                    tip_coords[name] = (tx, ty)
                    raw_dists[name] = float(np.linalg.norm([tx - wx, ty - wy]))

                # ── Step 2: merge Ring + Pinky into one unit ───────────────
                # Spec: "Ulnar Unit … act as a single coupled unit due to
                #        flexion synergy patterns."
                raw_dists["ring_pinky"] = max(raw_dists["ring"], raw_dists["pinky"])

                # ── Step 3: update FingerUnits (filter + velocity) ────────
                unit_state: dict[str, tuple[float, float]] = {}
                for name, fu in self.finger_units.items():
                    if name == "ring_pinky":
                        # Displacement filtered from merged displacement.
                        fd_rp, _ = fu.update(
                            raw_dists["ring_pinky"], resting_anchor=self.resting_anchor
                        )
                        # Velocity hardening: emergency flick if either ring or pinky flicks.
                        _, vel_ring = self._ring_unit.update(
                            raw_dists["ring"], resting_anchor=self.resting_anchor
                        )
                        _, vel_pinky = self._pinky_unit.update(
                            raw_dists["pinky"], resting_anchor=self.resting_anchor
                        )
                        vel_rp = max(vel_ring, vel_pinky)
                        fu.velocity = vel_rp  # used by is_velocity_peak()
                        unit_state[name] = (fd_rp, vel_rp)
                    else:
                        fd, vel = fu.update(
                            raw_dists[name], resting_anchor=self.resting_anchor
                        )
                        unit_state[name] = (fd, vel)

                # ── Step 4: Winner-Takes-All with 0.4 Isolation Index ─────
                # Spec: "If Index moves X, ignore any Middle finger
                #        movement < 0.5X"  (conservative → use 0.4 gate)
                #
                # In Advanced/Pinch layer we disable the Index unit entirely
                # to prevent mechanical conflict.
                names_for_wta = (
                    ("middle", "ring_pinky")
                    if self._layer_is_advanced
                    else ("index", "middle", "ring_pinky")
                )
                max_disp = max(unit_state[n][0] for n in names_for_wta)
                active_units: dict[str, tuple[float, float]] = {
                    name: unit_state[name]
                    for name in names_for_wta
                    if unit_state[name][0] >= ISOLATION_INDEX * max_disp
                }
                # The dominant unit (highest filtered displacement)
                if active_units:
                    winner = max(active_units, key=lambda n: active_units[n][0])
                else:
                    winner = max(names_for_wta, key=lambda n: unit_state[n][0])

                # ── Step 5: record samples during calibration ─────────────
                if self.calibrating:
                    w_dist, w_vel = active_units[winner]
                    if self.calib_step == 1:
                        self.calib_results.append(w_dist)
                    elif self.calib_step == 2:
                        self.calib_results.append(w_dist)
                        # Also collect peak velocities for threshold derivation
                        self.calib_vel_results.append(abs(w_vel))
                        self.finger_units[winner].calib_peak_vel = max(
                            self.finger_units[winner].calib_peak_vel, abs(w_vel)
                        )

                # ── Step 6: progress bar (index displacement) ────────────
                if self.resting_anchor is not None and self.max_flick is not None:
                    total_range = max(1e-6, self.max_flick - self.resting_anchor)
                    for key in ("index", "middle", "ring_pinky"):
                        dist = unit_state[key][0]
                        raw_delta = max(0.0, dist - self.resting_anchor)
                        progress[key] = min(1.0, raw_delta / total_range)
                    index_percent = progress["index"]

                # ── Step 7: dwell-confirmed peak trigger ────────────────
                #
                # Sub-step A: advance every FSM (must happen every frame so
                # dwell timers tick even during cooldown — prevents queued
                # triggers from backing up).  Index is fully skipped when the
                # pinch layer is active ("Occupied Index" rule).
                fired_units: list[str] = []
                if self.vel_threshold is not None:
                    for name, fu in self.finger_units.items():
                        # ── Fix 4: Occupied Index — explicit pinch guard ──
                        if self._layer_is_advanced and name == "index":
                            # Pinch is active: index is mechanically occupied.
                            # Do NOT call is_velocity_peak so its FSM cannot
                            # accidentally advance or fire.
                            print(
                                f"[SKIP] index: pinch layer ON — FSM frozen "
                                f"(state={fu.fsm_state_name})"
                            )
                            continue

                        peak_fired = fu.is_velocity_peak(self.vel_threshold)
                        if peak_fired:
                            fired_units.append(name)

                # Sub-step B: maintain the locked-winner.
                # As soon as any eligible unit enters DWELL we lock it so that
                # WTA displacement ranking cannot steal the slot during the
                # 200 ms confirmation window.
                eligible_for_lock = [
                    name
                    for name, fu in self.finger_units.items()
                    if fu.is_dwelling
                    and not (self._layer_is_advanced and name == "index")
                ]
                if self._locked_winner is None and eligible_for_lock:
                    # Pick the unit with the highest filtered displacement.
                    self._locked_winner = max(
                        eligible_for_lock, key=lambda n: unit_state[n][0]
                    )
                    print(
                        f"[LOCK]  {self._locked_winner.upper()}: entered DWELL — "
                        f"locked as winner (WTA frozen for this unit)"
                    )
                elif self._locked_winner is not None:
                    locked_fu = self.finger_units[self._locked_winner]
                    if not locked_fu.is_dwelling:
                        # Unit left DWELL without firing (abort or fired this frame)
                        print(
                            f"[LOCK]  {self._locked_winner.upper()}: lock released "
                            f"(state={locked_fu.fsm_state_name})"
                        )
                        self._locked_winner = None

                if (not self.cooldown) and fired_units:
                    # ── Fix 2: prefer the locked winner if it fired ──────
                    if self._locked_winner and self._locked_winner in fired_units:
                        fire_unit = self._locked_winner
                    else:
                        # Fallback: prefer WTA-isolated units, then highest disp.
                        isolated = [u for u in fired_units if u in active_units]
                        candidates = isolated if isolated else fired_units
                        fire_unit = max(candidates, key=lambda n: unit_state[n][0])

                    # Release lock now that we've committed to this unit.
                    self._locked_winner = None

                    cmd = layer_map.get(fire_unit, "Unknown")
                    self.cooldown = True
                    # ── Fix 3: emit debug ────────────────────────────────
                    print(
                        f"[EMIT]  trigger_activation → '{cmd}' "
                        f"(unit={fire_unit}, "
                        f"layer={'advanced' if self._layer_is_advanced else 'normal'}, "
                        f"latency={( time.perf_counter() - frame_t0)*1000:.1f}ms)"
                    )
                    self.trigger_activation.emit(cmd)
                    threading.Thread(target=self._cooldown_fn, daemon=True).start()

                # Rare fallback path: if vel_threshold isn't available yet,
                # require the unit to hold its displacement past 90% for 200ms.
                elif (
                    not self.cooldown
                    and self.vel_threshold is None
                    and self.resting_anchor is not None
                    and self.max_flick is not None
                ):
                    keys = (
                        ("middle", "ring_pinky")
                        if self._layer_is_advanced
                        else (
                            "index",
                            "middle",
                            "ring_pinky",
                        )
                    )
                    for key in keys:
                        if key not in active_units:
                            self._pos_dwell_start[key] = None
                            continue
                        if progress[key] >= 0.9:
                            if self._pos_dwell_start[key] is None:
                                self._pos_dwell_start[key] = time.perf_counter()
                            elif (
                                time.perf_counter() - self._pos_dwell_start[key]
                            ) * 1000.0 >= DWELL_CONFIRM_MS:
                                cmd = layer_map.get(key, "Unknown")
                                self.cooldown = True
                                self.trigger_activation.emit(cmd)
                                threading.Thread(
                                    target=self._cooldown_fn, daemon=True
                                ).start()
                                self._pos_dwell_start[key] = None
                                break
                        else:
                            self._pos_dwell_start[key] = None

                # ── Step 8: draw overlays ─────────────────────────────────
                cv2.circle(overlay, (int(wx), int(wy)), 9, (20, 215, 0), 2)

                # Thumb debug visualization (Landmark 4)
                cv2.line(
                    overlay,
                    (int(wx), int(wy)),
                    (int(tx_thumb), int(ty_thumb)),
                    (255, 255, 255),
                    2,
                )
                cv2.circle(
                    overlay, (int(tx_thumb), int(ty_thumb)), 6, (255, 255, 255), -1
                )

                active_names = set(active_units.keys())
                for name in ("index", "middle"):
                    tx, ty = tip_coords[name]
                    bgr = UNIT_COLORS_BGR[name]
                    thick = 3 if name in active_names else 1
                    cv2.circle(overlay, (int(tx), int(ty)), 8, bgr, thick)
                    cv2.line(
                        overlay,
                        (int(wx), int(wy)),
                        (int(tx), int(ty)),
                        bgr,
                        2 if name in active_names else 1,
                    )

                for name in ("ring", "pinky"):
                    tx, ty = tip_coords[name]
                    bgr = UNIT_COLORS_BGR["ring_pinky"]
                    thick = 3 if "ring_pinky" in active_names else 1
                    cv2.circle(overlay, (int(tx), int(ty)), 8, bgr, thick)
                    cv2.line(
                        overlay,
                        (int(wx), int(wy)),
                        (int(tx), int(ty)),
                        bgr,
                        2 if "ring_pinky" in active_names else 1,
                    )

                # Bright ring on winner tip(s)
                if winner == "ring_pinky":
                    for name in ("ring", "pinky"):
                        tx, ty = tip_coords[name]
                        cv2.circle(
                            overlay,
                            (int(tx), int(ty)),
                            14,
                            UNIT_COLORS_BGR["ring_pinky"],
                            3,
                        )
                else:
                    tx, ty = tip_coords[winner]
                    cv2.circle(
                        overlay, (int(tx), int(ty)), 14, UNIT_COLORS_BGR[winner], 3
                    )

                # HUD labels
                suppressed = [n for n in unit_state if n not in active_names]
                suppressed_str = ", ".join(suppressed) if suppressed else "none"
                cv2.putText(
                    overlay,
                    f"Winner : {winner}",
                    (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    overlay,
                    f"Gated  : {suppressed_str}",
                    (10, 54),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (180, 180, 180),
                    1,
                )
                if self.vel_threshold is not None:
                    w_vel_abs = abs(active_units[winner][1])
                    cv2.putText(
                        overlay,
                        f"Vel: {w_vel_abs:.1f} / thr {self.vel_threshold:.1f}",
                        (10, 78),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.50,
                        (140, 220, 255),
                        1,
                    )
            else:
                # No hand detected → ensure the layer indicator turns OFF.
                if self._layer_is_advanced:
                    self._layer_is_advanced = False
                    self.layer_active_update.emit(False)

            self.progress_update.emit(progress)
            self.change_pixmap.emit(overlay)

        self.cap.release()

    # ------------------------------------------------------------------ #
    def _cooldown_fn(self) -> None:
        """Blocks the cooldown flag for COOLDOWN_PERIOD seconds."""
        time.sleep(COOLDOWN_PERIOD)
        self.cooldown = False

    def stop(self) -> None:
        self.running = False
        self.wait()


# ============================================================
#  Main Window
# ============================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Finger Flick Accessibility Control")
        self.setFixedSize(880, 640)

        # Video display
        self.video_label = QLabel()
        self.video_label.setFixedSize(640, 480)
        self.video_label.setStyleSheet("background-color: #222;")

        # Three side-by-side progress bars
        self.progress_bars = MultiUnitProgressWidget()
        self.progress_bars.setFixedSize(240, 480)

        # Layer indicator
        self.layer_indicator = QLabel("LAYER ACTIVE: OFF")
        self.layer_indicator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layer_indicator.setFixedHeight(44)
        self.layer_indicator.setStyleSheet(
            "background-color: #222; color: #aaa; font-size: 14px; font-weight: bold;"
        )

        # Calibrate button
        self.calib_button = QPushButton("Calibrate")
        self.calib_button.clicked.connect(self.start_calibration)

        # Last-action label
        self.action_label = QLabel("—")
        self.action_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.action_label.setStyleSheet(
            "color: #aef; font-size: 15px; font-weight: bold;"
        )

        # Layout
        left_col = QVBoxLayout()
        left_col.addWidget(self.video_label)
        left_col.addWidget(self.action_label)
        left_col.addWidget(self.calib_button)

        root_h = QHBoxLayout()
        root_h.addLayout(left_col)
        right_col = QVBoxLayout()
        right_col.addWidget(self.layer_indicator)
        right_col.addWidget(self.progress_bars)
        right_col.addStretch(1)
        root_h.addLayout(right_col)

        holder = QWidget()
        holder.setLayout(root_h)
        self.setCentralWidget(holder)

        # Video thread
        self.thread = VideoThread()
        self.thread.change_pixmap.connect(self.update_image)
        self.thread.progress_update.connect(self.progress_bars.set_value)
        self.thread.trigger_activation.connect(self.activate_action)
        self.thread.layer_active_update.connect(self.set_layer_active)
        self.thread.start()

        self._calibrating = False

    # ------------------------------------------------------------------ #
    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img: np.ndarray) -> None:
        h, w, ch = cv_img.shape
        qt_img = QImage(cv_img.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_img))

    # ------------------------------------------------------------------ #
    def start_calibration(self) -> None:
        if self._calibrating:
            return
        self._calibrating = True

        def _worker() -> None:
            t = self.thread

            # ── Phase 1: resting position ──────────────────────────────
            self.calib_button.setText("Hold hand still…")
            time.sleep(0.4)
            # Reset all units before recording
            for fu in t.finger_units.values():
                fu.reset()
            t._ring_unit.reset()
            t._pinky_unit.reset()
            t.calib_results = []
            t.calib_vel_results = []
            t.calibrating = True
            t.calib_step = 1
            time.sleep(3.0)

            resting_val = float(np.mean(t.calib_results)) if t.calib_results else 0.0
            t.resting_anchor = resting_val

            # ── Phase 2: max flick (kinetic) ───────────────────────────
            t.calib_results = []
            t.calib_vel_results = []
            t.calib_step = 2
            self.calib_button.setText("Flick hard & fast!")
            time.sleep(2.5)

            max_disp_val = (
                float(np.max(t.calib_results))
                if t.calib_results
                else resting_val + 60.0
            )
            # Derive velocity threshold as 30 % of the peak velocity seen
            # during the flick phase — personalised to the user's motor range
            peak_vel = (
                float(np.max(t.calib_vel_results)) if t.calib_vel_results else 0.0
            )
            if peak_vel > 1.0:
                t.vel_threshold = 0.30 * peak_vel
            else:
                # Fallback: estimate from displacement range and a 150 ms flick
                span = max(max_disp_val - resting_val, 35.0)
                t.vel_threshold = span / 0.15 * 0.30

            t.max_flick = max(max_disp_val, resting_val + 35.0)
            # Calibrated peak stability tolerance:
            # 15% of the user's calibrated max-flick displacement range.
            flick_range = max(1e-6, float(t.max_flick - t.resting_anchor))
            peak_pos_tol = 0.15 * flick_range
            for fu in t.finger_units.values():
                fu.set_peak_pos_tol_px(peak_pos_tol)
            t._ring_unit.set_peak_pos_tol_px(peak_pos_tol)
            t._pinky_unit.set_peak_pos_tol_px(peak_pos_tol)
            t.calibrating = False
            t.calib_step = 0

            self.calib_button.setText("Calibrate")
            self._calibrating = False

        threading.Thread(target=_worker, daemon=True).start()

    # ------------------------------------------------------------------ #
    @pyqtSlot(str)
    def activate_action(self, cmd: str) -> None:
        """
        Receives the command string from the WTA winner unit.
        Extend here to publish over MQTT, trigger OS shortcuts, etc.
        """
        print(f"ACTION  → {cmd}")
        self.action_label.setText(f"⚡ {cmd}")
        # e.g.: mqtt_client.publish("home/control", cmd)

    # ------------------------------------------------------------------ #
    @pyqtSlot(bool)
    def set_layer_active(self, active: bool) -> None:
        if active:
            self.layer_indicator.setText("LAYER ACTIVE: ON (Advanced)")
            self.layer_indicator.setStyleSheet(
                "background-color: #145a2a; color: #b9ffcf; font-size: 14px; font-weight: bold;"
            )
            self.progress_bars.set_index_disabled(True)
        else:
            self.layer_indicator.setText("LAYER ACTIVE: OFF")
            self.layer_indicator.setStyleSheet(
                "background-color: #222; color: #aaa; font-size: 14px; font-weight: bold;"
            )
            self.progress_bars.set_index_disabled(False)

    # ------------------------------------------------------------------ #
    def closeEvent(self, event) -> None:
        self.thread.stop()
        event.accept()


# ============================================================
#  Multi-unit Progress Bars (Index/Middle/Ring-Pinky)
# ============================================================
class UnitProgressBarWidget(QWidget):
    def __init__(self, label: str, fill_color: QColor):
        super().__init__()
        self.label = label
        self.fill_color = fill_color
        self.disabled: bool = False
        self.disabled_fill_color = QColor(90, 90, 95)
        self.value: float = 0.0

        # Keep a pleasant aspect ratio inside MultiUnitProgressWidget.
        self.setMinimumWidth(70)

    def set_disabled(self, disabled: bool) -> None:
        self.disabled = bool(disabled)
        if self.disabled:
            self.value = 0.0
        self.update()

    def set_value(self, val: float) -> None:
        if self.disabled:
            self.value = 0.0
            return
        self.value = max(0.0, min(1.0, float(val)))
        self.update()

    def paintEvent(self, event) -> None:
        qp = QPainter(self)
        w, h = self.width(), self.height()
        fill_h = int(self.value * h)

        # Background
        qp.setBrush(QColor(40, 40, 45))
        qp.setPen(Qt.PenStyle.NoPen)
        qp.drawRect(0, 0, w, h)

        # Fill (unit-specific color)
        qp.setBrush(self.disabled_fill_color if self.disabled else self.fill_color)
        qp.drawRect(4, h - fill_h, w - 8, fill_h)

        # Border
        qp.setPen(QColor(255, 255, 255))
        qp.setBrush(Qt.BrushStyle.NoBrush)
        qp.drawRect(4, 2, w - 8, h - 4)

        # Unit label (top)
        qp.setPen(QColor(230, 230, 230))
        qp.setFont(QFont("Arial", 9))
        label = "GRAY/DISABLED" if self.disabled else self.label
        qp.drawText(QRect(0, 2, w, 18), Qt.AlignmentFlag.AlignCenter, label)

        # Percentage label (center)
        qp.setPen(QColor(233, 233, 233))
        qp.drawText(
            QRect(0, h // 2 - 10, w, 20),
            Qt.AlignmentFlag.AlignCenter,
            f"{int(self.value * 100)}%",
        )


class MultiUnitProgressWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.index_bar = UnitProgressBarWidget("Index", QColor(66, 153, 255))
        self.middle_bar = UnitProgressBarWidget("Middle", QColor(255, 82, 82))
        self.ring_bar = UnitProgressBarWidget("Ring/Pinky", QColor(255, 191, 0))

        root = QHBoxLayout()
        root.setContentsMargins(6, 0, 6, 0)
        root.setSpacing(8)
        root.addWidget(self.index_bar)
        root.addWidget(self.middle_bar)
        root.addWidget(self.ring_bar)
        self.setLayout(root)

    def set_value(self, val: dict[str, float] | float) -> None:
        if isinstance(val, dict):
            self.index_bar.set_value(val.get("index", 0.0))
            self.middle_bar.set_value(val.get("middle", 0.0))
            self.ring_bar.set_value(val.get("ring_pinky", 0.0))
        else:
            # Backward compatibility: treat as index.
            self.index_bar.set_value(float(val))
        self.update()

    def set_index_disabled(self, disabled: bool) -> None:
        self.index_bar.set_disabled(disabled)


# ============================================================
def main() -> None:
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
