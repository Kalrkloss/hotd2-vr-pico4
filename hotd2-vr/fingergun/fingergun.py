"""
Finger guns for Flycast's emulated light gun, tracked with a webcam (MediaPipe).

  aim     move / point your finger gun at the screen
  fire    drop your thumb onto your index finger, like the hammer of a finger gun
  reload  point the gun down, or take your hand out of view below: an off-screen shot

Three ways to aim (m cycles): mix (default: both of the next two, averaged), hand position
(where your hand is) and gun direction (how your hand is turned, fitted over the whole
hand). The calibration measures all of them at once. Keys in the preview window:

  k        guided calibration: centre, the four screen edges, then three test shots
  c        quick re-centre: aim at the centre of the game screen and press c
  [ ]      more / less movement needed for the same crosshair travel (calmer / snappier)
  t g      trigger more / less sensitive
  m        aiming: mix -> hand position -> gun direction
  q, Esc   quit

Sends "LG <player> <x> <y> <buttons>" datagrams (x, y in 1/10000ths of the screen) to
Flycast's UDP light gun input (vr.UdpPort, 127.0.0.1 only). Only one tracker runs at a
time. Remembers calibration in settings.json; writes status.json (twice a second) and
session.csv (every frame): numbers only, no images.
"""
import argparse
import collections
import csv
import json
import math
import socket
import sys
import threading
import time
import traceback
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

HERE = Path(__file__).resolve().parent
SETTINGS = HERE / "settings.json"
# Holding this local port open is the "already running" lock.
INSTANCE_PORT = 27016

# MediaPipe hand landmark indices
WRIST, THUMB_TIP, INDEX_MCP, INDEX_PIP, INDEX_TIP = 0, 4, 5, 6, 8
# Points that stay rigid in a finger-gun grip: wrist, knuckles, the extended index finger.
# The thumb is left out on purpose: it is the hammer, and must not steer the aim.
RIGID = (0, 5, 6, 7, 8, 9, 13, 17)

BTN_TRIGGER, BTN_RELOAD = 1, 2

# Per measurement: default distance from centre to each screen edge (left, right, up,
# down), the smallest allowed, how still "still" is during calibration, and how far an
# edge must be from the centre to count.
MODES = {
    "hand":  {"range": np.array([0.20, 0.20, 0.15, 0.15]), "min": 0.03, "still": 0.012, "edge": 0.015},
    "angle": {"range": np.radians([25.0, 25.0, 18.0, 18.0]), "min": math.radians(3.0),
              "still": math.radians(2.0), "edge": math.radians(3.0)},
}
AIM_MODES = ("mix", "hand", "angle")
AIM_NAMES = {"mix": "mix", "hand": "handpositie", "angle": "handrichting"}


class OneEuro:
    """One Euro filter: smooth when slow, responsive when fast (Casiez et al. 2012)."""

    def __init__(self, min_cutoff, beta, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x = self.dx = None
        self.t = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        x = np.asarray(x, dtype=float)
        if self.x is None:
            self.x, self.dx, self.t = x, np.zeros_like(x), t
            return x
        dt = max(t - self.t, 1e-3)
        dx = (x - self.x) / dt
        self.dx = self.dx + self._alpha(self.d_cutoff, dt) * (dx - self.dx)
        cutoff = self.min_cutoff + self.beta * np.linalg.norm(self.dx)
        self.x = self.x + self._alpha(cutoff, dt) * (x - self.x)
        self.t = t
        return self.x


def unit(v):
    return v / (np.linalg.norm(v) + 1e-9)


def yaw_pitch(v):
    """Direction angles in camera axes (x right, y down, z away from the camera)."""
    return math.atan2(v[0], -v[2]), math.atan2(v[1], math.hypot(v[0], v[2]))


def kabsch(ref, cur):
    """Rotation that best maps the centred point set ref onto cur (least squares)."""
    u, _, vt = np.linalg.svd(ref.T @ cur)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    return vt.T @ np.diag([1.0, 1.0, d]) @ u.T


class LatestFrame:
    """Grabs camera frames on a thread so the tracker always works on the newest one."""

    def __init__(self, index, width, height, fps):
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        # Note: forcing MJPG (for 60 fps) gave black frames at 1 fps on this machine's webcam.
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        if not self.cap.isOpened():
            raise SystemExit(f"Cannot open camera {index}")
        self.frame, self.stamp = None, 0.0
        self.lock = threading.Lock()
        self.running = True
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while self.running:
            ok, frame = self.cap.read()
            if ok:
                with self.lock:
                    self.frame, self.stamp = frame, time.perf_counter()

    def get(self):
        with self.lock:
            return self.frame, self.stamp

    def close(self):
        self.running = False
        self.cap.release()


class Trigger:
    """
    Thumb trigger that adapts to the hand pose. The thumb's "open" distance seen by the
    tracker changes with how the hand is held (aiming up, it reads shorter), so a fixed
    threshold either misses shots or never lets go. Instead: follow the open level, fire
    when the thumb drops well below it, let go once it comes back up from its lowest point.
    """

    # A "press" that stays down this long is the pose changing, not a shot.
    SETTLE = 0.6

    def __init__(self, ratio, release_cm):
        self.ratio = ratio              # fire below open level x ratio
        self.release_cm = release_cm    # let go this far above the lowest point of the press
        self.open_cm = None
        self.low_cm = 0.0
        self.down_for = 0.0
        self.pressed = False

    def threshold(self):
        return (self.open_cm or 0.0) * self.ratio

    def update(self, cm, dt):
        if self.open_cm is None:
            self.open_cm = cm
        if not self.pressed:
            # follow the open level: quickly up when the thumb opens further, slowly down
            tau = 0.25 if cm > self.open_cm else 0.8
            self.open_cm += (1 - math.exp(-dt / tau)) * (cm - self.open_cm)
            self.open_cm = max(self.open_cm, 1.5)
            if cm < self.threshold():
                self.pressed, self.low_cm, self.down_for = True, cm, 0.0
        else:
            self.low_cm = min(self.low_cm, cm)
            self.down_for += dt
            if cm > self.low_cm + self.release_cm or cm > self.open_cm * 0.85:
                self.pressed = False
            elif self.down_for > self.SETTLE:
                # held this way: that's how the open thumb reads in this pose now
                self.pressed, self.open_cm = False, max(cm / self.ratio * 0.9, 1.5)
        return self.pressed


class Gun:
    """One player's finger gun: aiming, trigger and reload state."""

    def __init__(self, player, args):
        self.player = player
        self.args = args
        self.mode = args.aim
        self.centers = {"hand": np.array([0.5, 0.5]), "angle": np.zeros(2)}
        self.ranges = {m: MODES[m]["range"] * args.range for m in MODES}
        # per measurement and edge (left, right, up, down): did it follow you to that edge
        # during calibration? Mix only averages the ones that did, per axis.
        self.reliable = {m: np.ones(4, dtype=bool) for m in MODES}
        self.trigger = Trigger(args.trigger, args.release_cm)
        self.filter = OneEuro(args.min_cutoff, args.beta)
        self.history = collections.deque(maxlen=45)   # (time, screen xy) for the shot position
        # last three raw readings per measurement, for the spike filter
        self.recent = {m: collections.deque(maxlen=3) for m in MODES}
        self.aim = {"hand": np.array([0.5, 0.5]), "angle": np.zeros(2)}   # spike-filtered
        self.targets = {m: np.array([0.5, 0.5]) for m in MODES}
        self.signals = {}       # every aiming method's raw reading, for session.csv
        self.ref_pts = None     # hand pose (centred RIGID points) recorded at calibration
        self.ref_dir = None     # gun direction in that pose
        self.cur_pts = None
        self.cur_dir = None
        self.held_until = 0.0
        self.held_pos = None
        self.reload_armed = True
        self.reload_until = 0.0
        self.thumb_cm = 0.0
        self.pitch_deg = 0.0
        self.hand_y = 0.5
        self.target = np.array([0.5, 0.5])
        self.screen = np.array([0.5, 0.5])
        self.offscreen = False
        self.calibrated = False
        self.seen = 0.0
        self.lost_handled = True
        self.last_t = None

    @property
    def pressed(self):
        return self.trigger.pressed

    @property
    def base(self):
        """The measurement shown and used for calibration checks: hand, unless aiming by angle."""
        return "angle" if self.mode == "angle" else "hand"

    @property
    def smooth_aim(self):
        return self.aim[self.base]

    def clear_recent(self):
        for q in self.recent.values():
            q.clear()

    def set_center(self):
        """What you aim at now becomes the centre of the screen, for every measurement."""
        self.centers["hand"] = self.aim["hand"].copy()
        if self.cur_pts is not None:
            # gun direction is measured as the hand's turn away from this pose
            self.ref_pts, self.ref_dir = self.cur_pts.copy(), self.cur_dir.copy()
            self.centers["angle"] = np.zeros(2)
            self.aim["angle"] = np.zeros(2)
        self.clear_recent()

    def scale_range(self, factor):
        for m in (MODES if self.mode == "mix" else (self.mode,)):
            self.ranges[m] = np.maximum(self.ranges[m] * factor, MODES[m]["min"])

    def to_screen(self, m, aim):
        """Aim -> screen, with its own range towards each edge (left, right, up, down)."""
        d = aim - self.centers[m]
        left, right, up, down = self.ranges[m]
        return 0.5 + 0.5 * d / np.array([left if d[0] < 0 else right, up if d[1] < 0 else down])

    def mix(self):
        """Average of the measurements that track you along each axis."""
        out = np.zeros(2)
        for axis, edges in ((0, [0, 1]), (1, [2, 3])):
            use = [m for m in MODES if self.reliable[m][edges].all()] or ["hand"]
            out[axis] = np.mean([self.targets[m][axis] for m in use])
        return out

    def update(self, lm, wl, now):
        args = self.args
        p = lambda i: np.array([lm[i].x, lm[i].y])
        w = lambda i: np.array([wl[i].x, wl[i].y, wl[i].z])
        dt = 1 / 30 if self.last_t is None else min(now - self.last_t, 0.5)
        self.last_t = now

        # Gun direction in 3D (metres). The barrel blends wrist->fingertip (long, steady)
        # with knuckle->fingertip (the finger).
        barrel = unit(unit(w(INDEX_TIP) - w(WRIST)) + unit(w(INDEX_TIP) - w(INDEX_MCP)))
        yaw, pitch = yaw_pitch(barrel)
        pts = np.array([w(i) for i in RIGID])
        self.cur_pts, self.cur_dir = pts - pts.mean(axis=0), barrel
        # The preview and your sense of left/right are mirrored relative to the camera.
        self.signals["barrel"] = np.array([-yaw, pitch])
        if self.ref_pts is not None:
            # Turn of the whole hand since calibration, fitted over all rigid points.
            v = kabsch(self.ref_pts, self.cur_pts) @ self.ref_dir
            (ry, rp), (y0, p0) = yaw_pitch(v), yaw_pitch(self.ref_dir)
            self.signals["angle"] = np.array([-(ry - y0), rp - p0])
        else:
            self.signals["angle"] = self.signals["barrel"]
        hand = p(INDEX_MCP) + 0.5 * (p(INDEX_TIP) - p(INDEX_MCP))
        self.signals["hand"] = np.array([1.0 - hand[0], hand[1]])
        self.pitch_deg, self.hand_y = math.degrees(pitch), p(INDEX_MCP)[1]

        # Median of the last three readings: drops one-frame glitches for one frame of delay.
        for m in MODES:
            self.recent[m].append(self.signals[m])
            self.aim[m] = np.median(np.array(self.recent[m]), axis=0)
            self.targets[m] = self.to_screen(m, self.aim[m])
        self.target = self.mix() if self.mode == "mix" else self.targets[self.mode]

        # Trigger: thumb tip against the index finger, measured in 3D.
        self.thumb_cm = 100 * min(np.linalg.norm(w(THUMB_TIP) - w(INDEX_PIP)),
                                  np.linalg.norm(w(THUMB_TIP) - w(INDEX_MCP)))
        was = self.trigger.pressed
        self.trigger.update(self.thumb_cm, dt)

        # Reload: gun pointed down. Outside the aiming range the crosshair stays on the
        # screen edge, so you can still shoot there.
        self.offscreen = self.pitch_deg > args.reload_deg

        smoothed = self.filter(np.clip(self.target, 0.0, 1.0), now)
        self.history.append((now, smoothed.copy()))
        if self.trigger.pressed and not was:
            # Shoot where you were aiming just before the thumb moved: the middle of the
            # positions from shot_from to shot_to seconds ago (immune to the thumb's tug
            # on the hand and to one-off jitter), held long enough for the game to read it.
            window = [pos for t, pos in self.history if args.shot_to <= now - t <= args.shot_from]
            older = [pos for t, pos in self.history if now - t >= args.shot_to]
            self.held_pos = (np.median(np.array(window), axis=0) if window
                             else older[-1] if older else smoothed.copy())
            self.held_until = now + args.hold
        self.screen = self.held_pos if (self.held_pos is not None and now < self.held_until) else smoothed
        self.seen = now
        self.lost_handled = False

    def check_lost(self, now):
        """Hand gone: release the trigger, and count a hand that left low as a reload."""
        if now - self.seen <= 0.25:
            return
        self.trigger.pressed = False
        if not self.lost_handled:
            self.lost_handled = True
            if self.pitch_deg > 20 or self.hand_y > 0.8:
                self.reload_until = now + 0.12

    def buttons(self, now):
        b = 0
        if self.offscreen:
            # one off-screen shot per "gun down", re-armed once the gun comes back up
            if self.reload_armed:
                self.reload_armed = False
                self.reload_until = now + 0.12
        else:
            self.reload_armed = True
        if now < self.reload_until:
            b |= BTN_RELOAD
        elif self.trigger.pressed and not self.offscreen:
            b |= BTN_TRIGGER
        return b


class Wizard:
    """
    Guided calibration. Each aiming step captures itself once you hold still for a moment
    (space captures straight away); the last step tunes the trigger on three test shots.
    Every aiming measurement is calibrated at once, so switching with m needs no redo.
    """

    STEPS = [
        ("intro", "Pak je vingerpistool erbij"),
        ("center", "Richt op het MIDDEN van het gamescherm"),
        ("up", "Richt op de BOVENRAND van het scherm"),
        ("down", "Richt op de ONDERRAND van het scherm"),
        ("left", "Richt op de LINKERRAND van het scherm"),
        ("right", "Richt op de RECHTERRAND van het scherm"),
        ("shoot", "Richt ergens op het scherm en schiet 3 keer"),
    ]
    HOLD = 0.7          # seconds of holding still to capture an aim
    SHOTS = 3

    def __init__(self, gun):
        self.gun = gun
        self.step = 0
        self.samples = collections.deque()      # (time, aim) for the stillness check
        self.progress = 0.0
        self.points = {}
        self.warning = ""
        self.dips = []
        self.dip_low = None
        self.open_level = None
        self.done_at = None
        self.intro_pressed = True   # a thumb already down doesn't count as the start shot
        gun.clear_recent()

    @property
    def key(self):
        return self.STEPS[self.step][0] if self.step < len(self.STEPS) else "done"

    @property
    def finished(self):
        return self.done_at is not None and time.perf_counter() - self.done_at > 2.5

    def update(self, now, visible, force=False):
        if self.key == "done":
            return
        if not visible and not (self.key == "intro" and force):
            self.samples.clear()
            self.progress, self.warning = 0.0, "Ik zie je hand niet"
            return
        self.warning = ""
        if self.key == "intro":
            # begin when you're ready: space, or one shot with the finger gun
            if force or (self.gun.pressed and not self.intro_pressed):
                self.step += 1
                self.samples.clear()
            self.intro_pressed = self.gun.pressed
            return
        if self.key == "shoot":
            self._shoot()
            return
        base = self.gun.base
        aim = self.gun.aim[base].copy()
        self.samples.append((now, aim))
        while self.samples and now - self.samples[0][0] > self.HOLD + 0.1:
            self.samples.popleft()
        pts = np.array([a for _, a in self.samples])
        span = now - self.samples[0][0]
        still = np.max(np.linalg.norm(pts - pts.mean(axis=0), axis=1)) < MODES[base]["still"]
        self.progress = min(span / self.HOLD, 1.0) if still else 0.0
        if not still:
            # restart the hold from the newest reading
            self.samples.clear()
            self.samples.append((now, aim))
        if (self.progress >= 1.0 or force) and self._accept(aim):
            self.points[self.key] = {m: self.gun.aim[m].copy() for m in MODES}
            self.samples.clear()
            self.progress = 0.0
            self.step += 1

    def _accept(self, aim):
        if self.key == "center":
            self.gun.set_center()
            return True
        # In mix either measurement following you to the edge is enough.
        g = self.gun
        ok = False
        for m in (MODES if g.mode == "mix" else (g.base,)):
            d, need = g.aim[m] - g.centers[m], MODES[m]["edge"]
            ok |= {"up": -d[1] > need, "down": d[1] > need, "left": -d[0] > need, "right": d[0] > need}[self.key]
        if not ok:
            self.warning = "Nog iets verder richten"
        return ok

    def _shoot(self):
        cm = self.gun.thumb_cm
        if self.open_level is None:
            self.open_level = cm
        if self.dip_low is None:
            self.open_level += 0.1 * (max(cm, self.open_level * 0.75) - self.open_level)
            if cm < self.open_level * 0.7:
                self.dip_low = cm
        else:
            self.dip_low = min(self.dip_low, cm)
            if cm > self.open_level * 0.85:
                self.dips.append(self.dip_low / self.open_level)
                self.dip_low = None
        self.progress = len(self.dips) / self.SHOTS
        if len(self.dips) >= self.SHOTS:
            self._finish()

    def _finish(self):
        g = self.gun
        for m in MODES:
            c, r = g.centers[m], MODES[m]["range"].copy()
            got = {k: v[m] for k, v in self.points.items()}
            # How far this measurement moved towards each edge. A bit inside the edges you
            # pointed at, so the very edge is reachable. If it barely moved that way (you
            # reached that edge with the other kind of motion), keep the default for that
            # edge: a tiny range would make the crosshair twitch on every bit of noise.
            for i, (edge, axis, sign) in enumerate((("left", 0, -1), ("right", 0, 1), ("up", 1, -1), ("down", 1, 1))):
                if edge in got:
                    span = sign * (got[edge][axis] - c[axis])
                    g.reliable[m][i] = span > MODES[m]["edge"]
                    if g.reliable[m][i]:
                        r[i] = 0.95 * span
            g.ranges[m] = r
        if self.dips:
            closed = float(np.median(self.dips))
            # fire a little under halfway between your pressed and open thumb
            g.trigger.ratio = float(np.clip(closed + 0.45 * (1 - closed), 0.45, 0.85))
        g.calibrated = True
        self.step = len(self.STEPS)
        self.done_at = time.perf_counter()
        save_settings(g)


def save_settings(gun):
    data = {"version": 3, "mode": gun.mode, "trigger_ratio": gun.trigger.ratio,
            "centers": {k: v.tolist() for k, v in gun.centers.items()},
            "ranges": {k: v.tolist() for k, v in gun.ranges.items()},
            "reliable": {k: v.tolist() for k, v in gun.reliable.items()}}
    if gun.ref_pts is not None:
        data["reference"] = {"points": gun.ref_pts.tolist(), "direction": gun.ref_dir.tolist()}
    try:
        SETTINGS.write_text(json.dumps(data, indent=1))
    except OSError:
        pass


def load_settings(gun):
    """Returns False when there is nothing usable yet (first run or older format)."""
    try:
        data = json.loads(SETTINGS.read_text())
    except (OSError, ValueError):
        return False
    if data.get("version", 0) < 3:
        # older files calibrated only one measurement; keep just the trigger
        gun.trigger.ratio = data.get("trigger_ratio", gun.trigger.ratio)
        return False
    gun.mode = data.get("mode", gun.mode)
    gun.trigger.ratio = data.get("trigger_ratio", gun.trigger.ratio)
    for k, v in data.get("centers", {}).items():
        gun.centers[k] = np.array(v)
    for k, v in data.get("ranges", {}).items():
        gun.ranges[k] = np.array(v)
    for k, v in data.get("reliable", {}).items():
        gun.reliable[k] = np.array(v, dtype=bool)
    ref = data.get("reference")
    if ref:
        gun.ref_pts, gun.ref_dir = np.array(ref["points"]), np.array(ref["direction"])
    gun.calibrated = True
    return True


def put(img, text, org, scale=0.6, colour=(255, 255, 255), thick=1):
    """Text on a dimmed box, readable on any camera image. (An outline drawn as thicker
    text doesn't line up in OpenCV 5: thicker text also gets wider.)"""
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = org
    h, w = img.shape[:2]
    x0, y0, x1, y1 = max(x - 5, 0), max(y - th - 6, 0), min(x + tw + 5, w), min(y + base + 4, h)
    img[y0:y1, x0:x1] = (img[y0:y1, x0:x1] * 0.3).astype(np.uint8)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thick, cv2.LINE_AA)


def draw(frame, guns, hands, wizard, fps):
    h, w = frame.shape[:2]
    out = cv2.flip(frame, 1)
    for lm in hands:
        for i in (WRIST, THUMB_TIP, INDEX_MCP, INDEX_PIP, INDEX_TIP):
            cv2.circle(out, (int((1 - lm[i].x) * w), int(lm[i].y * h)), 3, (255, 200, 0), -1)

    # mini game screen, bottom right
    sw, sh = 160, 120
    ox, oy = w - sw - 10, h - sh - 10

    if wizard is not None:
        out = (out * 0.45).astype(np.uint8)
        cv2.rectangle(out, (ox, oy), (ox + sw, oy + sh), (255, 255, 255), 1)
        key = wizard.key
        if key == "done":
            put(out, "Klaar! Veel plezier.", (20, 60), 1.0, (0, 255, 120), 2)
        else:
            put(out, f"Kalibratie - stap {wizard.step + 1} van {len(wizard.STEPS)}", (20, 34), 0.6, (200, 200, 200), 1)
            put(out, wizard.STEPS[wizard.step][1], (20, 70), 0.7, (0, 255, 255), 2)
            hint = {"intro": "Druk op SPATIE of schiet een keer om te beginnen",
                    "shoot": "Duim omlaag en weer omhoog"}.get(key, "Houd even stil - hij legt zichzelf vast (spatie = nu)")
            put(out, hint, (20, 100), 0.55)
            if wizard.warning:
                put(out, wizard.warning, (20, 130), 0.6, (0, 140, 255), 2)
            # progress bar
            bx, by, bw = 20, 150, 300
            cv2.rectangle(out, (bx, by), (bx + bw, by + 14), (255, 255, 255), 1)
            cv2.rectangle(out, (bx, by), (bx + int(bw * wizard.progress), by + 14), (0, 255, 120), -1)
            if key == "shoot":
                put(out, f"{len(wizard.dips)} / {wizard.SHOTS} schoten", (bx + bw + 10, by + 13), 0.55)
            # where to aim, on the mini screen
            spot = {"intro": (0.5, 0.5), "center": (0.5, 0.5), "up": (0.5, 0.0), "down": (0.5, 1.0),
                    "left": (0.0, 0.5), "right": (1.0, 0.5), "shoot": (0.5, 0.5)}[key]
            cv2.circle(out, (ox + int(spot[0] * sw), oy + int(spot[1] * sh)), 9, (0, 255, 255), 2)
            put(out, "Esc = stoppen", (20, h - 16), 0.5, (200, 200, 200), 1)
        return out

    cv2.rectangle(out, (ox, oy), (ox + sw, oy + sh), (255, 255, 255), 1)
    for gun in guns:
        colour = (0, 0, 255) if gun.pressed else (0, 255, 0)
        cv2.circle(out, (ox + int(gun.screen[0] * sw), oy + int(gun.screen[1] * sh)), 6, colour, -1)
        state = "HERLADEN" if gun.offscreen else ("PANG" if gun.pressed else "richten")
        put(out, f"P{gun.player + 1} {state}   duim {gun.thumb_cm:3.1f} cm (schiet < {gun.trigger.threshold():3.1f})"
                 f"   [{AIM_NAMES[gun.mode]}]", (10, 24 + 26 * gun.player), 0.5)
    put(out, f"{fps:.0f} fps", (w - 70, 24), 0.5, (200, 200, 200))
    if not guns[0].calibrated:
        put(out, "Druk K voor de kalibratie", (10, h - 16), 0.65, (0, 255, 255), 2)
    else:
        put(out, "K = kalibreren   C = midden   [ ] = gevoeligheid   M = richtmethode", (10, h - 16), 0.45, (220, 220, 220))
    return out


def claim_single_instance():
    """Two trackers would both steer the crosshair. Hold a local port as the lock."""
    lock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        lock.bind(("127.0.0.1", INSTANCE_PORT))
        lock.listen(1)
    except OSError:
        print("\nDe vingerpistool-tracker draait al. Gebruik dat venster (of sluit het eerst).\n")
        sys.exit(1)
    return lock


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--port", type=int, default=27015, help="Flycast vr.UdpPort")
    ap.add_argument("--players", type=int, default=1, choices=(1, 2))
    ap.add_argument("--aim", choices=AIM_MODES, default="mix")
    ap.add_argument("--range", type=float, default=1.0, help="aiming range scale (bigger = more movement)")
    ap.add_argument("--trigger", type=float, default=0.62, help="fire below this fraction of the open thumb distance")
    ap.add_argument("--release-cm", type=float, default=1.0, help="thumb travel back up that ends a shot")
    ap.add_argument("--shot-from", type=float, default=0.30, help="shot position: median of the aim from this many s ago...")
    ap.add_argument("--shot-to", type=float, default=0.10, help="...to this many s ago")
    ap.add_argument("--hold", type=float, default=0.20, help="seconds the shot position is held for the game")
    ap.add_argument("--min-cutoff", type=float, default=0.6, help="smoothing when still (Hz, lower = calmer)")
    ap.add_argument("--beta", type=float, default=1.5, help="how fast smoothing lets go when you move")
    ap.add_argument("--reload-deg", type=float, default=35.0, help="gun this far below level = reload")
    ap.add_argument("--no-preview", action="store_true")
    args = ap.parse_args()

    lock = claim_single_instance()
    landmarker = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(HERE / "hand_landmarker.task")),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=args.players,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5))
    cam = LatestFrame(args.camera, 640, 480, 60)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dest = ("127.0.0.1", args.port)
    guns = [Gun(i, args) for i in range(args.players)]
    # first run, or settings from an older version: start with the guided calibration
    wizard = None if load_settings(guns[0]) else Wizard(guns[0])
    last_stamp, t0, last_ms = 0.0, time.perf_counter(), -1
    fps_t, fps_n, fps = time.perf_counter(), 0, 0.0
    status_t = 0.0

    session = open(HERE / "session.csv", "w", newline="")
    log = csv.writer(session)
    log.writerow(["t", "fps", "hands", "mode", "target_x", "target_y", "screen_x", "screen_y",
                  "thumb_cm", "open_cm", "pressed", "reload", "buttons", "wizard", "pitch_deg",
                  "hand_x", "hand_y", "angle_x", "angle_y", "barrel_x", "barrel_y",
                  "t_hand_x", "t_hand_y", "t_angle_x", "t_angle_y"])

    print(__doc__)
    try:
        while True:
            frame, stamp = cam.get()
            if frame is None or stamp == last_stamp:
                time.sleep(0.002)
                continue
            last_stamp = stamp
            now = time.perf_counter()
            # MediaPipe needs strictly increasing timestamps; two frames can land in one millisecond.
            last_ms = max(int((stamp - t0) * 1000), last_ms + 1)
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), last_ms)
                # Two players: the left hand on screen is P1, the right one P2 (mirrored view).
                found = sorted(zip(result.hand_landmarks, result.hand_world_landmarks),
                               key=lambda pair: 1 - pair[0][INDEX_MCP].x)
                for gun, (lm, wl) in zip(guns, found):
                    gun.update(lm, wl, now)
                if wizard is not None:
                    wizard.update(now, visible=len(found) > 0)
            except Exception:
                # One bad frame shouldn't end the game: note it and carry on.
                with open(HERE / "crash.log", "a") as f:
                    f.write(f"--- {time.strftime('%H:%M:%S')}\n{traceback.format_exc()}")
                found = []

            for gun in guns:
                gun.check_lost(now)
                if wizard is not None:
                    # no shots and a centred crosshair while calibrating
                    x, y, buttons = 0.5, 0.5, 0
                else:
                    x, y = gun.screen
                    buttons = gun.buttons(now)
                sock.sendto(f"LG {gun.player} {round(x * 10000)} {round(y * 10000)} {buttons}".encode(), dest)
                sig = lambda k: gun.signals.get(k, (math.nan, math.nan))
                log.writerow([f"{now - t0:.3f}", f"{fps:.1f}", len(found), gun.mode,
                              f"{gun.target[0]:.3f}", f"{gun.target[1]:.3f}", f"{x:.3f}", f"{y:.3f}",
                              f"{gun.thumb_cm:.2f}", f"{gun.trigger.open_cm or 0:.2f}", int(gun.pressed),
                              int(gun.offscreen), buttons, wizard.key if wizard else "", f"{gun.pitch_deg:.1f}"]
                             + [f"{v:.4f}" for k in ("hand", "angle", "barrel") for v in sig(k)]
                             + [f"{v:.3f}" for m in MODES for v in gun.targets[m]])

            if now - status_t >= 0.5:
                status_t = now
                session.flush()
                status = {"time": round(now - t0, 1), "fps": round(fps, 1), "hands": len(found),
                          "wizard": wizard.key if wizard else None, "guns": [
                    {"mode": g.mode, "seen_ago": round(now - g.seen, 2),
                     "target": [round(float(v), 3) for v in g.target], "screen": [round(float(v), 3) for v in g.screen],
                     "ranges": {m: [round(float(v), 4) for v in g.ranges[m]] for m in MODES},
                     "thumb_cm": round(float(g.thumb_cm), 2), "open_cm": round(float(g.trigger.open_cm or 0), 2),
                     "fire_below_cm": round(float(g.trigger.threshold()), 2), "pitch_deg": round(g.pitch_deg, 1),
                     "pressed": bool(g.pressed), "reload": bool(g.offscreen)} for g in guns]}
                try:
                    (HERE / "status.json").write_text(json.dumps(status))
                except OSError:
                    pass    # being read right now; next one in half a second

            fps_n += 1
            if now - fps_t >= 1.0:
                fps, fps_n, fps_t = fps_n / (now - fps_t), 0, now
            if not args.no_preview:
                cv2.imshow("finger guns", draw(frame, guns, [lm for lm, _ in found], wizard, fps))
            if wizard is not None and wizard.finished:
                wizard = None

            key = cv2.waitKey(1) & 0xFF
            g = guns[0]
            if wizard is not None:
                if key == 27:
                    wizard = None
                elif key == ord(" "):
                    wizard.update(now, visible=len(found) > 0, force=True)
                continue
            if key in (ord("q"), 27):
                break
            if key == ord("k"):
                wizard = Wizard(g)
            elif key == ord("c"):
                g.set_center()
                g.calibrated = True
            elif key == ord("["):
                g.scale_range(1.1)
            elif key == ord("]"):
                g.scale_range(1 / 1.1)
            elif key == ord("t"):
                g.trigger.ratio = min(g.trigger.ratio + 0.04, 0.9)
            elif key == ord("g"):
                g.trigger.ratio = max(g.trigger.ratio - 0.04, 0.3)
            elif key == ord("m"):
                g.mode = AIM_MODES[(AIM_MODES.index(g.mode) + 1) % len(AIM_MODES)]
                g.clear_recent()
            if key in (ord("c"), ord("["), ord("]"), ord("t"), ord("g"), ord("m")):
                save_settings(g)
    finally:
        session.close()
        cam.close()
        cv2.destroyAllWindows()
        lock.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        with open(HERE / "crash.log", "a") as f:
            f.write(f"--- {time.strftime('%H:%M:%S')} (fatal)\n{traceback.format_exc()}")
        raise
