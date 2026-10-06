"""
Finger guns for Flycast's emulated light gun, tracked with a webcam (MediaPipe).

  aim     move / point your finger gun at the screen
  fire    drop your thumb onto your index finger, like the hammer of a finger gun
  reload  open your hand for a moment (spread the other fingers): an off-screen shot.
          --reload down: the old way, point the gun down or leave the view below.

Ways to aim (m cycles): personal (default after a calibration: where your hand is and how
it is turned, combined the way you aimed at nine calibration points, like a ray from your
finger to the screen), mix (both measurements averaged), hand position only and gun
direction only. The calibration measures all of them at once. Keys in the preview window:

  k        guided calibration: centre, edges, corners, then three test shots
  c        quick re-centre: aim at the centre of the game screen and press c
  [ ]      more / less movement needed for the same crosshair travel (calmer / snappier)
  t g      trigger more / less sensitive
  m        aiming: personal -> mix -> hand position -> gun direction
  x        marker on/off in session.csv (e.g. around a stretch where you don't shoot)
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
# The fingers that are curled in a finger gun (MCP, PIP, TIP): straightened = open hand.
CURLED = ((9, 10, 12), (13, 14, 16), (17, 18, 20))
# Landmarks logged raw to session.csv: wrist, index MCP/PIP/tip, middle and pinky MCP, thumb tip.
LOG_POINTS = (0, 5, 6, 8, 9, 13, 17, 4, 1, 2, 3, 7)

BTN_TRIGGER, BTN_RELOAD = 1, 2

# Per measurement: default distance from centre to each screen edge (left, right, up,
# down), the smallest allowed, how still "still" is during calibration, and how far an
# edge must be from the centre to count.
MODES = {
    "hand":  {"range": np.array([0.20, 0.20, 0.15, 0.15]), "min": 0.03, "still": 0.012, "edge": 0.015},
    "angle": {"range": np.radians([25.0, 25.0, 18.0, 18.0]), "min": math.radians(3.0),
              "still": math.radians(2.0), "edge": math.radians(3.0)},
}
AIM_MODES = ("fit", "mix", "hand", "angle")
AIM_NAMES = {"fit": "persoonlijk", "mix": "mix", "hand": "handpositie", "angle": "handrichting"}
# Calibration targets on the game screen for the personal fit. While calibrating, the
# crosshair is sent to the current target, so it shows in the game where to aim. A bit in
# from the edges: a marker on the very edge is half off-screen.
FIT_POINTS = {"center": (0.5, 0.5), "up": (0.5, 0.1), "down": (0.5, 0.9), "left": (0.1, 0.5),
              "right": (0.9, 0.5), "upleft": (0.1, 0.1), "upright": (0.9, 0.1),
              "downleft": (0.1, 0.9), "downright": (0.9, 0.9)}
GAME_PX = np.array([640, 480])


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
    tracker changes with how the hand is held, so a fixed threshold either misses shots or
    never lets go. Instead: follow the open level, fire when the thumb drops well below it,
    let go once it comes back up from its lowest point.

    Finger tilted up steeply (median pitch of the last 3 frames below --steep-deg): the
    camera can't see the thumb behind the index finger any more and MediaPipe guesses it
    lying against the finger, which reads as pressed. In that pose the trigger keeps its
    own open level, and a drop only arms a shot: it fires when the thumb comes back up
    within --return-time (a real tap does; a hidden thumb mostly stays "down"). Frames
    where the tilt jumps more than --jump-deg are misreads and start no press. Replayed on
    two real sessions: suspect presses aiming up 10 -> 4, held (stuck) presses at the top
    1.3 -> 0.3 s, every deliberate press kept; level-finger shots unchanged (+~4 ms), steep
    ones ~50 ms later (the shot still goes where you aimed when the thumb went down).
    """

    # A "press" that stays down this long is the pose changing, not a shot.
    SETTLE = 0.6
    STEEP_HYST = 4.0        # degrees back up before the steep pose ends
    STEEP_SEED = 0.65       # first steep open level, relative to the level one

    def __init__(self, ratio, release_cm, steep_deg=-25.0, return_s=0.4, jump_deg=20.0):
        self.ratio = ratio              # fire below open level x ratio
        self.release_cm = release_cm    # let go this far above the lowest point of the press
        self.steep_deg, self.return_s, self.jump_deg = steep_deg, return_s, jump_deg
        self.levels = {"level": None, "steep": None}
        self.pose = "level"
        self.pitches = collections.deque(maxlen=3)
        self.prev_pitch = None
        self.pressed = False
        self.tap = False                # a fire-on-return press: up again next frame
        self.pending = None             # steep: (dip start time, lowest cm) of an armed drop
        self.low_cm = 0.0
        self.down_for = 0.0
        self.dip_t = None               # when this press's thumb drop started
        self.event = ""                 # for session.csv: press/armed/return/settle/jump

    @property
    def open_cm(self):
        return self.levels[self.pose]

    def threshold(self):
        return (self.open_cm or 0.0) * self.ratio

    def reset(self):
        """Hand lost, opened or settling after a reload: no press, and no tilt history
        from before (it made the first shot after a reload late)."""
        self.pressed = self.tap = False
        self.pending = None
        self.pitches.clear()
        self.prev_pitch = None

    def update(self, cm, dt, pitch=0.0, now=0.0):
        self.event = ""
        jump = self.prev_pitch is not None and abs(pitch - self.prev_pitch) > self.jump_deg
        self.prev_pitch = pitch
        if jump and not self.pressed:
            # a misread frame: no pose change, no learning, no new press
            self.event = "jump"
            return False
        self.pitches.append(pitch)
        med = float(np.median(self.pitches))
        if self.pose == "level" and med < self.steep_deg:
            self.pose = "steep"
        elif self.pose == "steep" and med > self.steep_deg + self.STEEP_HYST:
            self.pose = "level"
        L = self.levels
        if L["level"] is None:
            L["level"] = cm
        if L["steep"] is None:
            L["steep"] = L["level"] * self.STEEP_SEED

        if self.tap:
            self.pressed = self.tap = False
            return False
        if self.pending is not None:
            t0, low = self.pending[0], min(self.pending[1], cm)
            self.pending = (t0, low)
            if cm > low + self.release_cm or cm > L["steep"] * 0.85:
                # the thumb came back: that was a tap, fire it now
                self.pending, self.pressed, self.tap, self.dip_t = None, True, True, t0
                self.event = "return"
            elif self.pose == "level":
                self.pending = None             # tilted back down while "pressed": not a tap
            elif now - t0 > self.return_s:
                # stayed down: that's how the hidden thumb reads in this pose now
                self.pending = None
                L["steep"] = max(cm / self.ratio * 0.9, 1.5)
                self.event = "settle"
            return self.pressed
        if not self.pressed:
            # follow the open level: quickly up when the thumb opens further, slowly down
            k = self.pose
            tau = 0.25 if cm > L[k] else 0.8
            L[k] = max(L[k] + (1 - math.exp(-dt / tau)) * (cm - L[k]), 1.5)
            if cm < L[k] * self.ratio:
                if k == "steep":
                    self.pending, self.event = (now, cm), "armed"
                else:
                    self.pressed, self.low_cm, self.down_for, self.dip_t = True, cm, 0.0, now
                    self.event = "press"
            return self.pressed
        self.low_cm = min(self.low_cm, cm)
        self.down_for += dt
        if cm > self.low_cm + self.release_cm or cm > self.open_cm * 0.85:
            self.pressed = False
        elif self.down_for > self.SETTLE:
            # held this way: that's how the open thumb reads in this pose now
            self.pressed = False
            L[self.pose] = max(cm / self.ratio * 0.9, 1.5)
            self.event = "settle"
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
        self.trigger = Trigger(args.trigger, args.release_cm, args.steep_deg, args.return_time, args.jump_deg)
        self.filter = OneEuro(args.min_cutoff, args.beta)
        self.history = collections.deque(maxlen=45)   # (time, screen xy) for the shot position
        # last few raw readings per measurement, for the spike filter
        self.recent = {m: collections.deque(maxlen=args.spike_frames) for m in MODES}
        self.aim = {"hand": np.array([0.5, 0.5]), "angle": np.zeros(2)}   # spike-filtered
        self.targets = {m: np.array([0.5, 0.5]) for m in MODES}
        # personal fit: screen = 0.5 + gain * (fit @ features - 0.5) + offset
        self.fit = None             # 2 x len(features())
        self.fit_gain = 1.0
        self.fit_offset = np.zeros(2)
        self.fit_error_px = None    # how well the fit matched the calibration points
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
        self.fingers = 0            # curled fingers straightened right now
        self.reach = [0.0] * 3      # per curled finger: tip-to-knuckle / first segment
        self.raw = np.full(len(LOG_POINTS) * 3, np.nan)     # for session.csv
        self.straight = [False] * 3
        self.bend = [0.0] * 3       # per curled finger: cos of the bend at its middle joint
        # what was sent to the game, for the preview: counts and recent events
        self.last_buttons = 0
        self.shots = self.reloads = 0
        self.shot_at = self.reload_at = -10.0
        self.events = collections.deque(maxlen=5)   # (time, text)
        self.open_evidence = 0.0    # seconds of open-hand reading, see update()
        self.velocity = np.zeros(2)
        self.hand_open = False
        self.out = None             # what the sender sent last (--output chase)
        self.out_t = 0.0
        self.ease_until = 0.0       # after a shot hold: ease back slower, no snap
        self.recent_targets = collections.deque(maxlen=30)     # (time, target) for C
        self.trigger_quiet_until = 0.0
        self.lock = threading.Lock()    # update() runs on the main loop, output() on the sender
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

    def features(self):
        """What the personal fit works from: where the hand is and how it is turned."""
        return np.r_[self.aim["hand"], self.aim["angle"], 1.0]

    def fit_target(self):
        raw = self.fit @ self.features()
        return 0.5 + self.fit_gain * (raw - 0.5) + self.fit_offset

    def recenter(self, now):
        """C: what you aim at now becomes the centre. The personal fit only shifts (its
        hand-turn reference must stay the one it was calibrated with), by the median aim of
        the last 0.3 s rather than one frame."""
        if self.mode == "fit" and self.fit is not None:
            recent = [tg for t, tg in self.recent_targets if now - t <= 0.3] or [self.fit_target()]
            self.fit_offset += 0.5 - np.median(np.array(recent), axis=0)
            self.recent_targets.clear()
            self.clear_recent()
        else:
            self.set_center()
        self.events.append((now, "midden gezet"))

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
        if self.mode == "fit":
            self.fit_gain = float(np.clip(self.fit_gain / factor, 0.3, 3.0))
            return
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
        self.seen = now
        # image x, y and MediaPipe's relative depth of a few landmarks, for tuning later
        self.raw = np.array([[lm[i].x, lm[i].y, lm[i].z] for i in LOG_POINTS]).reshape(-1)
        self.lost_handled = False

        # Open hand: two of the three curled fingers straight (the pinky often reads half
        # bent). A finger counts as straight when it bends little at the middle joint AND
        # its tip is far from its knuckle: curled fingers hidden behind the hand are
        # sometimes guessed straight-ish, but rarely also that long.
        self.reach = [np.linalg.norm(w(tip) - w(mcp)) / (np.linalg.norm(w(pip) - w(mcp)) + 1e-6)
                      for mcp, pip, tip in CURLED]
        self.bend = [float(np.dot(unit(w(pip) - w(mcp)), unit(w(tip) - w(pip)))) for mcp, pip, tip in CURLED]
        self.straight = [b > args.open_bend and r > args.open_reach for b, r in zip(self.bend, self.reach)]
        self.fingers = sum(self.straight)
        # Open-hand evidence: frames that read open add their time, frames that don't take
        # it away again, so one missed frame (at 15 fps a third of the wait) no longer
        # restarts the count. Capped at --open-time so a closed hand reads closed again
        # within that time. On a real session: every reload of the old rule, the attempts it
        # missed, 0.1-0.6 s sooner, and no more doubtful ones.
        step = min(dt, 0.2)
        if self.fingers >= 2:
            self.open_evidence = min(self.open_evidence + step, args.open_time)
        else:
            self.open_evidence = max(self.open_evidence - step, 0.0)
        was_open = self.hand_open
        if not self.hand_open and self.open_evidence >= args.open_time - 1e-6:
            self.hand_open = True
        elif self.hand_open and self.open_evidence <= 0.0:
            self.hand_open = False
            # the hand settles back into the gun grip: no shot from that, and the
            # readings of the open hand shouldn't steer the aim
            self.trigger_quiet_until = now + 0.3
            self.clear_recent()
        if args.reload == "open" and (self.hand_open or was_open):
            # Opening the hand also opens the thumb: don't let the trigger learn that as
            # its open level, and keep the crosshair where it was.
            self.trigger.reset()
            self.offscreen = self.hand_open
            return

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

        # Median of the last --spike-frames readings: drops glitches up to half that long
        # (five catches the two-frame flips seen when the finger points at the camera).
        for m in MODES:
            self.recent[m].append(self.signals[m])
            self.aim[m] = np.median(np.array(self.recent[m]), axis=0)
            self.targets[m] = self.to_screen(m, self.aim[m])
        if self.mode == "fit" and self.fit is not None:
            self.target = self.fit_target()
            self.recent_targets.append((now, self.target.copy()))
        elif self.mode in ("mix", "fit"):
            self.target = self.mix()
        else:
            self.target = self.targets[self.mode]

        # Trigger: thumb tip against the index finger, measured in 3D.
        self.thumb_cm = 100 * min(np.linalg.norm(w(THUMB_TIP) - w(INDEX_PIP)),
                                  np.linalg.norm(w(THUMB_TIP) - w(INDEX_MCP)))
        was = self.trigger.pressed
        if now < self.trigger_quiet_until:
            self.trigger.reset()
        else:
            self.trigger.update(self.thumb_cm, dt, self.pitch_deg, now)

        # Reload (--reload down): gun pointed down. Outside the aiming range the crosshair
        # stays on the screen edge, so you can still shoot there.
        self.offscreen = args.reload == "down" and self.pitch_deg > args.reload_deg

        smoothed = self.filter(np.clip(self.target, 0.0, 1.0), now)
        # how fast the crosshair itself moves now, for gliding between camera frames
        if self.history and now - self.history[-1][0] < 0.15:
            self.velocity = (smoothed - self.history[-1][1]) / max(now - self.history[-1][0], 1e-3)
        else:
            self.velocity = np.zeros(2)
        self.history.append((now, smoothed.copy()))
        if self.trigger.pressed and not was:
            # Shoot where you were aiming just before the thumb moved: the middle of the
            # positions from shot_from to shot_to seconds before the drop started (immune to
            # the thumb's tug on the hand and to one-off jitter), held long enough for the
            # game to read it. (A steep shot fires when the thumb comes back: still aim from
            # when it went down.)
            ref = self.trigger.dip_t if self.trigger.dip_t is not None else now
            window = [pos for t, pos in self.history if args.shot_to <= ref - t <= args.shot_from]
            older = [pos for t, pos in self.history if ref - t >= args.shot_to]
            self.held_pos = (np.median(np.array(window), axis=0) if window
                             else older[-1] if older else smoothed.copy())
            self.held_until = now + args.hold
            self.out = None             # the shot goes exactly where it was aimed
            self.ease_until = self.held_until + 0.15
        self.screen = self.held_pos if (self.held_pos is not None and now < self.held_until) else smoothed

    def output(self, now):
        """
        Crosshair to send now. The camera gives ~30 new positions a second, the game reads
        60, so between camera frames (--output):
          chase   ease towards the newest position (time constant --chase-tau): no
                  stepping and no overshoot, for ~30 ms. Measured on a real session the
                  smoothest: spikes over 20 px at the centre 3.0% -> 1.9%.
          extrap  carry on along the current movement (at most --glide s): snappier,
                  but it carries tracking noise further.
          step    send each camera position as it is.
        """
        if self.held_pos is not None and now < self.held_until:
            # the shot goes exactly where it was aimed; afterwards ease back from there
            self.out, self.out_t = self.screen.copy(), now
            return self.screen
        if self.hand_open or self.last_t is None or now - self.seen > 0.25:
            self.out = None
            return self.screen
        mode = self.args.output
        if mode == "extrap":
            ahead = min(now - self.last_t, self.args.glide)
            return np.clip(self.screen + self.velocity * max(ahead, 0.0), 0.0, 1.0)
        if mode == "step" or self.out is None:
            self.out, self.out_t = self.screen.copy(), now
            return self.out
        # after a shot the crosshair returns from the shot position gently, not in a jump
        tau = max(self.args.chase_tau, 0.08) if now < self.ease_until else self.args.chase_tau
        dt = min(max(now - self.out_t, 0.0), 0.1)
        self.out = self.out + (1.0 - math.exp(-dt / max(tau, 1e-3))) * (self.screen - self.out)
        self.out_t = now
        return self.out

    def check_lost(self, now):
        """Hand gone: release the trigger (--reload down: a hand that left low reloads)."""
        if now - self.seen <= 0.25:
            return
        self.trigger.reset()
        self.hand_open = False
        self.open_evidence = 0.0
        if not self.lost_handled:
            self.lost_handled = True
            self.events.append((now, "hand kwijt"))
            if self.args.reload == "down" and (self.pitch_deg > 20 or self.hand_y > 0.8):
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
        new = b & ~self.last_buttons
        if new & BTN_TRIGGER:
            self.shots += 1
            self.shot_at = now
            self.events.append((now, "PANG"))
        if new & BTN_RELOAD:
            self.reloads += 1
            self.reload_at = now
            self.events.append((now, "HERLADEN"))
        self.last_buttons = b
        return b


class Wizard:
    """
    Guided calibration. Each aiming step captures itself once you hold still for a moment
    (space captures straight away); the last step tunes the trigger on three test shots.
    Every aiming measurement is calibrated at once, so switching with m needs no redo.
    All readings of each hold go into the personal fit (a least-squares fit over nine points).
    """

    STEPS = [
        ("intro", "Pak je vingerpistool erbij"),
        ("center", "Richt op het vizier in de game: MIDDEN"),
        ("up", "Richt op het vizier in de game: BOVEN"),
        ("down", "Richt op het vizier in de game: ONDER"),
        ("left", "Richt op het vizier in de game: LINKS"),
        ("right", "Richt op het vizier in de game: RECHTS"),
        ("upleft", "Richt op het vizier in de game: LINKSBOVEN"),
        ("upright", "Richt op het vizier in de game: RECHTSBOVEN"),
        ("downleft", "Richt op het vizier in de game: LINKSONDER"),
        ("downright", "Richt op het vizier in de game: RECHTSONDER"),
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
        self.fit_data = []      # (features of every reading in the hold, screen target)
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
        self.samples.append((now, aim, self.gun.features()))
        while self.samples and now - self.samples[0][0] > self.HOLD + 0.1:
            self.samples.popleft()
        pts = np.array([s[1] for s in self.samples])
        span = now - self.samples[0][0]
        still = np.max(np.linalg.norm(pts - pts.mean(axis=0), axis=1)) < MODES[base]["still"]
        self.progress = min(span / self.HOLD, 1.0) if still else 0.0
        if not still:
            # restart the hold from the newest reading
            self.samples.clear()
            self.samples.append((now, aim, self.gun.features()))
        if (self.progress >= 1.0 or force) and self._accept(aim):
            self.points[self.key] = {m: self.gun.aim[m].copy() for m in MODES}
            feats = np.array([s[2] for s in self.samples])
            if self.key == "center":
                # the hand-turn reference was just set to this pose: these readings were
                # measured against the old one, so re-centre their turn on zero
                feats[:, 2:4] -= feats[:, 2:4].mean(axis=0)
            self.fit_data.append((feats, np.array(FIT_POINTS[self.key])))
            self.samples.clear()
            self.progress = 0.0
            self.step += 1

    def _accept(self, aim):
        if self.key == "center":
            self.gun.set_center()
            return True
        if self.key not in ("up", "down", "left", "right"):
            return True     # corners: only used by the fit, which weighs them itself
        # In mix either measurement following you to the edge is enough; the personal fit
        # on hand position needs the hand itself to have moved that way.
        g = self.gun
        ok = False
        if g.mode in ("mix", "fit"):
            measures = ("hand",) if g.args.fit_with == "hand" else MODES
        else:
            measures = (g.base,)
        for m in measures:
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
        self._fit()
        if self.dips:
            closed = float(np.median(self.dips))
            # fire a little under halfway between your pressed and open thumb
            g.trigger.ratio = float(np.clip(closed + 0.45 * (1 - closed), 0.45, 0.85))
        g.calibrated = True
        self.step = len(self.STEPS)
        self.done_at = time.perf_counter()
        save_settings(g)

    def _fit(self):
        g = self.gun
        if len(self.fit_data) < 5:
            return
        g.fit, g.fit_error_px = fit_aim(self.fit_data, g.args.fit_with)
        g.fit_gain, g.fit_offset = 1.0, np.zeros(2)
        g.mode = "fit"


# Which features() columns drive screen x and y in the personal fit (4 = the constant).
# "hand": x from the hand's left/right position, y from its height. Measured on real
# sessions this beats adding the hand turn: the turn is read from the finger's 3D
# direction, which falls apart when the finger points at the camera (the middle of the
# screen) - noisy, flipping, squashed - and it reads differently in play than while
# calibrating. Cross-validated over the nine points: 108 px vs 170 px.
FIT_COLUMNS = {"hand": ([0, 4], [1, 4]), "hand+angle": ([0, 1, 2, 3, 4], [0, 1, 2, 3, 4])}


def fit_aim(fit_data, use="hand", ridge=1e-3, centre_weight=8.0):
    """
    Weighted least squares from features() to the screen over every reading of every
    calibration point (each point weighs the same, however long its hold, except the
    centre: that is where you aim most, so it counts centre_weight times; on a real
    calibration that took its miss from 27 to 14 px at no cost elsewhere). Features are
    scaled alike and lightly ridged, so one that barely moved can't get a huge weight (that
    would amplify its noise). Returns the 2 x 5 weights and the mean error in game px.
    """
    X = np.vstack([f for f, _ in fit_data])
    Y = np.vstack([np.repeat(t[None], len(f), axis=0) for f, t in fit_data])
    point_w = [(centre_weight if np.allclose(t, 0.5) else 1.0) / len(f) for f, t in fit_data]
    sw = np.sqrt(np.concatenate([np.full(len(f), w) for (f, _), w in zip(fit_data, point_w)]))[:, None]
    W = np.zeros((2, X.shape[1]))
    for axis, cols in enumerate(FIT_COLUMNS[use]):
        Xa = X[:, cols]
        scale = np.where(np.array(cols) == 4, 1.0, Xa.std(axis=0) + 1e-6)
        A = Xa / scale * sw
        reg = np.diag([0.0 if c == 4 else ridge for c in cols])
        W[axis, cols] = np.linalg.solve(A.T @ A + reg, A.T @ (Y[:, axis:axis + 1] * sw))[:, 0] / scale
    errors = [np.linalg.norm((np.median(f @ W.T, axis=0) - t) * GAME_PX) for f, t in fit_data]
    return W, float(np.mean(errors))


def save_settings(gun):
    data = {"version": 3, "mode": gun.mode, "trigger_ratio": gun.trigger.ratio,
            "centers": {k: v.tolist() for k, v in gun.centers.items()},
            "ranges": {k: v.tolist() for k, v in gun.ranges.items()},
            "reliable": {k: v.tolist() for k, v in gun.reliable.items()}}
    if gun.ref_pts is not None:
        data["reference"] = {"points": gun.ref_pts.tolist(), "direction": gun.ref_dir.tolist()}
    if gun.fit is not None:
        data["fit"] = {"weights": gun.fit.tolist(), "gain": gun.fit_gain,
                       "offset": gun.fit_offset.tolist(), "error_px": gun.fit_error_px}
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
    fit = data.get("fit")
    if fit:
        gun.fit = np.array(fit["weights"])
        gun.fit_gain, gun.fit_offset = fit["gain"], np.array(fit["offset"])
        gun.fit_error_px = fit.get("error_px")
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


# Colours (BGR) for the preview: one per input, used on the hand and in the panel.
C_AIM, C_FIRE, C_RELOAD = (90, 230, 90), (60, 60, 255), (255, 170, 40)
C_IDLE, C_TEXT, C_DIM = (110, 110, 110), (235, 235, 235), (150, 150, 150)
PANEL_W = 300
FLASH = 0.35        # seconds a shot / reload lights up


def draw_hand(out, lm, gun):
    """The tracked hand, coloured by what each part does: index aims, thumb fires, the
    other three fingers reload when straightened."""
    h, w = out.shape[:2]
    pt = lambda i: (int((1 - lm[i].x) * w), int(lm[i].y * h))
    thumb = C_FIRE if gun.pressed else (80, 200, 255)
    for chain in ((0, 5, 9, 13, 17, 0),):
        for a, b in zip(chain, chain[1:]):
            cv2.line(out, pt(a), pt(b), C_IDLE, 2, cv2.LINE_AA)
    bones = [((0, 1, 2, 3, 4), thumb), ((5, 6, 7, 8), C_AIM)]
    bones += [((m, m + 1, m + 2, m + 3), C_RELOAD if s else C_IDLE)
              for m, s in zip((9, 13, 17), gun.straight)]
    for chain, colour in bones:
        for a, b in zip(chain, chain[1:]):
            cv2.line(out, pt(a), pt(b), colour, 3, cv2.LINE_AA)
    cv2.circle(out, pt(INDEX_TIP), 7, C_AIM, 2, cv2.LINE_AA)
    cv2.circle(out, pt(THUMB_TIP), 7, thumb, -1 if gun.pressed else 2, cv2.LINE_AA)


def bar(img, x, y, w, h, frac, colour, mark=None):
    cv2.rectangle(img, (x, y), (x + w, y + h), (70, 70, 70), -1)
    cv2.rectangle(img, (x, y), (x + int(w * min(max(frac, 0.0), 1.0)), y + h), colour, -1)
    if mark is not None:
        mx = x + int(w * min(max(mark, 0.0), 1.0))
        cv2.line(img, (mx, y - 4), (mx, y + h + 4), C_TEXT, 2)


def card(img, x, y, w, h, title, how, colour, active, flash):
    """A box per input: lit border while active, filled for a moment after it fires."""
    if flash > 0:
        overlay = img.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), colour, -1)
        cv2.addWeighted(overlay, 0.35 * flash, img, 1 - 0.35 * flash, 0, img)
    cv2.rectangle(img, (x, y), (x + w, y + h), colour if active or flash > 0 else (80, 80, 80), 2)
    cv2.putText(img, title, (x + 10, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2, cv2.LINE_AA)
    (tw, _), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.putText(img, how, (x + 18 + tw, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_DIM, 1, cv2.LINE_AA)


def text(img, s, org, scale=0.5, colour=C_TEXT, thick=1):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thick, cv2.LINE_AA)


def draw_panel(panel, gun, y0, height, now, compact):
    x, w = 10, PANEL_W - 20
    seen = now - gun.seen < 0.25
    text(panel, f"SPELER {gun.player + 1}", (x, y0 + 22), 0.65, C_TEXT, 2)
    if not seen:
        text(panel, "HAND NIET IN BEELD", (x + 110, y0 + 22), 0.5, (0, 140, 255), 2)
    else:
        text(panel, f"richten: {AIM_NAMES[gun.mode]}", (x + 120, y0 + 22), 0.45, C_DIM)
    y = y0 + 34

    # Aim: the game screen with the crosshair; a red edge where you aim past it.
    ch = 96 if compact else 120
    card(panel, x, y, w, ch, "RICHTEN", "", C_AIM, seen, 0)
    sh = 66 if compact else 90
    sw = sh * 4 // 3
    sx, sy = x + w - sw - 10, y + (ch - sh) // 2
    cv2.rectangle(panel, (sx, sy), (sx + sw, sy + sh), (30, 30, 30), -1)
    cv2.rectangle(panel, (sx, sy), (sx + sw, sy + sh), C_DIM, 1)
    t = gun.target
    for out_, p0, p1 in ((t[0] < 0, (sx, sy), (sx, sy + sh)), (t[0] > 1, (sx + sw, sy), (sx + sw, sy + sh)),
                         (t[1] < 0, (sx, sy), (sx + sw, sy)), (t[1] > 1, (sx, sy + sh), (sx + sw, sy + sh))):
        if out_ and seen:
            cv2.line(panel, p0, p1, (0, 0, 255), 3)
    trail = [pos for ts, pos in gun.history if now - ts < 0.4]
    for a, b in zip(trail, trail[1:]):
        cv2.line(panel, (sx + int(a[0] * sw), sy + int(a[1] * sh)), (sx + int(b[0] * sw), sy + int(b[1] * sh)),
                 (60, 120, 60), 1, cv2.LINE_AA)
    cx, cy = sx + int(gun.screen[0] * sw), sy + int(gun.screen[1] * sh)
    colour = C_FIRE if now - gun.shot_at < FLASH else C_AIM
    cv2.circle(panel, (cx, cy), 6, colour, 2, cv2.LINE_AA)
    cv2.line(panel, (cx - 10, cy), (cx + 10, cy), colour, 1)
    cv2.line(panel, (cx, cy - 10), (cx, cy + 10), colour, 1)
    outside = seen and (t.min() < 0 or t.max() > 1)
    text(panel, "buiten beeld" if outside else ("op scherm" if seen else "-"), (x + 10, y + 50),
         0.45, (0, 0, 255) if outside else C_DIM)
    text(panel, "met je wijsvinger", (x + 10, y + 70), 0.4, C_DIM)
    text(panel, "C = midden", (x + 10, y + ch - 12), 0.4, C_DIM)
    y += ch + 8

    # Fire: thumb distance against the firing point.
    ch = 78 if compact else 84
    fflash = max(0.0, 1 - (now - gun.shot_at) / FLASH)
    card(panel, x, y, w, ch, "SCHIETEN", "duim omlaag", C_FIRE, gun.pressed, fflash)
    text(panel, f"{gun.shots}x", (x + w - 50, y + 22), 0.55, C_TEXT, 2)
    full = max(gun.trigger.open_cm or 1.0, gun.thumb_cm, 1.0) * 1.15
    bar(panel, x + 10, y + 36, w - 20, 12, gun.thumb_cm / full, C_FIRE if gun.pressed else (80, 200, 255),
        mark=gun.trigger.threshold() / full)
    steep = gun.trigger.pose == "steep"
    idle = (f"steil: schiet als duim terugkomt ({gun.thumb_cm:.1f} cm)" if steep
            else f"duim {gun.thumb_cm:.1f} cm  -  schiet onder de streep")
    text(panel, "PANG!" if gun.pressed else idle,
         (x + 10, y + ch - 12), 0.55 if gun.pressed else 0.4, C_FIRE if gun.pressed else C_DIM,
         2 if gun.pressed else 1)
    y += ch + 8

    # Reload: which of the three fingers read straight, and how long the hand has been open.
    ch = 78 if compact else 84
    rflash = max(0.0, 1 - (now - gun.reload_at) / FLASH)
    how = "hand open" if gun.args.reload == "open" else "omlaag richten"
    card(panel, x, y, w, ch, "HERLADEN", how, C_RELOAD, gun.offscreen, rflash)
    text(panel, f"{gun.reloads}x", (x + w - 50, y + 22), 0.55, C_TEXT, 2)
    if gun.args.reload == "open":
        for i, (name, s) in enumerate(zip(("middel", "ring", "pink"), gun.straight)):
            bx = x + 10 + i * 62
            cv2.rectangle(panel, (bx, y + 34), (bx + 54, y + 50), C_RELOAD if s and seen else (70, 70, 70), -1)
            text(panel, name, (bx + 4, y + 47), 0.4, (20, 20, 20) if s and seen else C_DIM)
        held = gun.open_evidence / gun.args.open_time
        bar(panel, x + 200, y + 38, w - 210, 8, 1.0 if gun.hand_open else held, C_RELOAD)
        hint = "HERLADEN!" if gun.hand_open else "strek 2 van de 3 vingers"
    else:
        bar(panel, x + 10, y + 38, w - 20, 8, gun.pitch_deg / gun.args.reload_deg, C_RELOAD, mark=1.0)
        hint = "HERLADEN!" if gun.offscreen else f"kanteling {gun.pitch_deg:.0f} / {gun.args.reload_deg:.0f} graden"
    text(panel, hint, (x + 10, y + ch - 12), 0.55 if gun.offscreen else 0.4,
         C_RELOAD if gun.offscreen else C_DIM, 2 if gun.offscreen else 1)
    y += ch + 8

    if not compact:
        # what the game got, newest first
        text(panel, "laatst:", (x, y + 14), 0.45, C_DIM)
        for i, (ts, ev) in enumerate(reversed(gun.events)):
            colour = {"PANG": C_FIRE, "HERLADEN": C_RELOAD}.get(ev, (0, 140, 255))
            text(panel, f"{ev}  {now - ts:4.1f}s", (x + 60 + (i % 2) * 115, y + 14 + (i // 2) * 18), 0.45, colour)


def draw(frame, guns, hands, wizard, fps):
    h, w = frame.shape[:2]
    out = cv2.flip(frame, 1)
    for lm, gun in zip(hands, guns):
        draw_hand(out, lm, gun)

    # mini game screen, bottom right
    sw, sh = 160, 120
    ox, oy = w - sw - 10, h - sh - 10

    if wizard is not None:
        out = (out * 0.45).astype(np.uint8)
        cv2.rectangle(out, (ox, oy), (ox + sw, oy + sh), (255, 255, 255), 1)
        key = wizard.key
        if key == "done":
            put(out, "Klaar! Veel plezier.", (20, 60), 1.0, (0, 255, 120), 2)
            if guns[0].fit_error_px is not None:
                err = guns[0].fit_error_px
                put(out, f"Gemiddelde afwijking op de kalibratiepunten: {err:.0f} px", (20, 100), 0.55,
                    (0, 255, 120) if err < 90 else (0, 140, 255))
                if err >= 90:
                    put(out, "Veel: druk K en richt elk punt rustig en precies", (20, 128), 0.55, (0, 140, 255))
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
            spot = FIT_POINTS.get(key, (0.5, 0.5))
            cv2.circle(out, (ox + int(spot[0] * sw), oy + int(spot[1] * sh)), 9, (0, 255, 255), 2)
            put(out, "Esc = stoppen", (20, h - 16), 0.5, (200, 200, 200), 1)
        return out

    now = time.perf_counter()
    # the camera image flashes on what the game just got
    for gun in guns:
        for at, colour in ((gun.shot_at, C_FIRE), (gun.reload_at, C_RELOAD)):
            if now - at < FLASH:
                cv2.rectangle(out, (0, 0), (w - 1, h - 1), colour, 8)
    camera_fps = (0, 140, 255) if fps < 20 else C_DIM
    put(out, f"camera {fps:.0f} fps", (w - 135, 24), 0.5, camera_fps)
    if fps < 20:
        put(out, "traag: meer licht of minder andere programma's", (10, h - 44), 0.45, camera_fps)
    if not guns[0].calibrated:
        put(out, "Druk K voor de kalibratie", (10, h - 16), 0.65, (0, 255, 255), 2)
    else:
        put(out, "K = kalibreren (9 punten)   C = midden   [ ] = gevoeligheid   M = richtmethode   Q = stoppen",
            (10, h - 16), 0.42, (220, 220, 220))

    compact = len(guns) > 1
    section = 312 if compact else max(h, 430)
    height = max(h, section * len(guns))
    panel = np.full((height, PANEL_W, 3), 24, np.uint8)
    for i, gun in enumerate(guns):
        with gun.lock:     # the sender thread adds events and moves the crosshair
            draw_panel(panel, gun, i * section, section, now, compact)
    canvas = np.zeros((height, w + PANEL_W, 3), np.uint8)
    canvas[:h, :w] = out
    canvas[:, w:] = panel
    return canvas


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
    ap.add_argument("--aim", choices=AIM_MODES, default="fit", help="fit needs a calibration (k), else mix")
    ap.add_argument("--range", type=float, default=1.0, help="aiming range scale (bigger = more movement)")
    ap.add_argument("--trigger", type=float, default=0.62, help="fire below this fraction of the open thumb distance")
    ap.add_argument("--release-cm", type=float, default=1.0, help="thumb travel back up that ends a shot")
    ap.add_argument("--steep-deg", type=float, default=-25.0,
                    help="finger tilted up more than this: shots fire when the thumb comes back (-90 = off)")
    ap.add_argument("--return-time", type=float, default=0.4, help="--steep-deg: how long a tap may take")
    ap.add_argument("--jump-deg", type=float, default=20.0,
                    help="a frame whose tilt jumps more than this starts no shot (999 = off)")
    # The thumb starts dropping about a frame before the press registers and the aim is
    # already smoothed (~80 ms behind): looking further back shot where you aimed 0.2-0.4 s ago.
    ap.add_argument("--shot-from", type=float, default=0.15, help="shot position: median of the aim from this many s ago...")
    ap.add_argument("--shot-to", type=float, default=0.05, help="...to this many s ago")
    ap.add_argument("--hold", type=float, default=0.10, help="seconds the shot position is held for the game")
    # Filter defaults from replaying a real session at 120 Hz: fewer centre spikes than
    # 0.6/1.5 for ~8 ms. (A 5-frame median with 0.4/0.7 is calmer but ~55 ms slower.)
    ap.add_argument("--spike-frames", type=int, default=3, help="median over this many frames against glitches")
    ap.add_argument("--min-cutoff", type=float, default=0.8, help="smoothing when still (Hz, lower = calmer)")
    ap.add_argument("--beta", type=float, default=1.0, help="how fast smoothing lets go when you move")
    ap.add_argument("--fit-with", choices=tuple(FIT_COLUMNS), default="hand",
                    help="what the personal calibration aims with")
    ap.add_argument("--reload", choices=("open", "down"), default="open",
                    help="reload by opening your hand, or by pointing the gun down")
    ap.add_argument("--reload-deg", type=float, default=35.0, help="--reload down: this far below level")
    ap.add_argument("--rate", type=float, default=120.0, help="crosshair updates sent per second")
    ap.add_argument("--open-time", type=float, default=0.12,
                    help="seconds of open hand (a missed frame takes some off, not all) to reload")
    ap.add_argument("--open-reach", type=float, default=2.0,
                    help="a finger is straight when its tip is this many first segments from its knuckle")
    # Measured on a session: an opened hand's middle finger often reads only 0.55-0.7, which
    # at 0.7 delayed reloads by 0.4-2 s; curled fingers in the gun grip stay under 0.6 (99%).
    ap.add_argument("--open-bend", type=float, default=0.6,
                    help="...and its middle joint bends less than this (cosine; 1 = dead straight)")
    ap.add_argument("--output", choices=("chase", "extrap", "step"), default="chase",
                    help="crosshair between camera frames: ease towards it, run ahead, or step")
    ap.add_argument("--chase-tau", type=float, default=0.03, help="--output chase: time constant (s)")
    ap.add_argument("--glide", type=float, default=0.05, help="--output extrap: max seconds ahead")
    ap.add_argument("--no-preview", action="store_true")
    args = ap.parse_args()

    lock = claim_single_instance()
    if sys.platform == "win32":
        # Ahead of background work (renders, builds) so the camera keeps its frame rate.
        import ctypes
        ABOVE_NORMAL_PRIORITY_CLASS = 0x8000
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), ABOVE_NORMAL_PRIORITY_CLASS)
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
    sent = [(0.5, 0.5, 0)] * len(guns)     # last thing sent per gun, for session.csv
    stop = threading.Event()

    def sender():
        """Sends every gun at a steady rate, between camera frames too."""
        period = 1.0 / args.rate
        nxt = time.perf_counter()
        while not stop.is_set():
            now = time.perf_counter()
            for i, gun in enumerate(guns):
                with gun.lock:
                    gun.check_lost(now)
                    if wizard is not None:
                        # no shots while calibrating; the crosshair marks where to aim
                        x, y = FIT_POINTS.get(wizard.key, (0.5, 0.5)) if wizard.gun is gun else (0.5, 0.5)
                        buttons = 0
                        gun.out = None
                    else:
                        (x, y), buttons = gun.output(now), gun.buttons(now)
                sent[i] = (x, y, buttons)
                sock.sendto(f"LG {gun.player} {round(x * 10000)} {round(y * 10000)} {buttons}".encode(), dest)
            nxt += period
            if time.perf_counter() - nxt > 0.1:
                nxt = time.perf_counter()     # fell behind (sleep, debugger): don't burst
            stop.wait(max(nxt - time.perf_counter(), 0.0))

    threading.Thread(target=sender, daemon=True).start()
    last_stamp, t0, last_ms = 0.0, time.perf_counter(), -1
    fps_t, fps_n, fps = time.perf_counter(), 0, 0.0
    status_t = 0.0
    mark = False        # x: you mark a stretch (e.g. "not shooting now") in session.csv

    session = open(HERE / "session.csv", "w", newline="")
    log = csv.writer(session)
    log.writerow(["t", "fps", "hands", "mode", "target_x", "target_y", "screen_x", "screen_y",
                  "thumb_cm", "open_cm", "pressed", "reload", "buttons", "wizard", "pitch_deg",
                  "hand_x", "hand_y", "angle_x", "angle_y", "barrel_x", "barrel_y",
                  "t_hand_x", "t_hand_y", "t_angle_x", "t_angle_y", "fingers", "reach_middle", "reach_ring", "reach_pinky", "smooth_x", "smooth_y"]
                 + [f"lm{i}_{a}" for i in LOG_POINTS for a in "xyz"] + ["bend_middle", "bend_ring", "bend_pinky", "trig_pose", "open_level", "open_steep",
                    "trig_event", "armed", "mark"])

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
                    with gun.lock:
                        gun.update(lm, wl, now)
                if wizard is not None:
                    wizard.update(now, visible=len(found) > 0)
            except Exception:
                # One bad frame shouldn't end the game: note it and carry on.
                with open(HERE / "crash.log", "a") as f:
                    f.write(f"--- {time.strftime('%H:%M:%S')}\n{traceback.format_exc()}")
                found = []

            for gun, (x, y, buttons) in zip(guns, sent):
                sig = lambda k: gun.signals.get(k, (math.nan, math.nan))
                log.writerow([f"{now - t0:.3f}", f"{fps:.1f}", len(found), gun.mode,
                              f"{gun.target[0]:.3f}", f"{gun.target[1]:.3f}", f"{x:.3f}", f"{y:.3f}",
                              f"{gun.thumb_cm:.2f}", f"{gun.trigger.open_cm or 0:.2f}", int(gun.pressed),
                              int(gun.offscreen), buttons, wizard.key if wizard else "", f"{gun.pitch_deg:.1f}"]
                             + [f"{v:.4f}" for k in ("hand", "angle", "barrel") for v in sig(k)]
                             + [f"{v:.3f}" for m in MODES for v in gun.targets[m]] + [gun.fingers] + [f"{v:.2f}" for v in gun.reach]
                             + [f"{v:.4f}" for v in gun.screen] + [f"{v:.5f}" for v in gun.raw]
                             + [f"{v:.3f}" for v in gun.bend]
                             + [int(gun.trigger.pose == "steep"), f"{gun.trigger.levels['level'] or 0:.2f}",
                                f"{gun.trigger.levels['steep'] or 0:.2f}", gun.trigger.event,
                                int(gun.trigger.pending is not None), int(mark)])

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
                     "pressed": bool(g.pressed), "reload": bool(g.offscreen),
                     "trigger_pose": g.trigger.pose, "open_level_cm": round(float(g.trigger.levels["level"] or 0), 2),
                     "open_steep_cm": round(float(g.trigger.levels["steep"] or 0), 2)} for g in guns]}
                try:
                    (HERE / "status.json").write_text(json.dumps(status))
                except OSError:
                    pass    # being read right now; next one in half a second

            fps_n += 1
            if now - fps_t >= 1.0:
                fps, fps_n, fps_t = fps_n / (now - fps_t), 0, now
            if not args.no_preview:
                shown = draw(frame, guns, [lm for lm, _ in found], wizard, fps)
                if mark:
                    put(shown, "MARKER AAN (x)", (10, 56), 0.6, (255, 0, 255), 2)
                cv2.imshow("finger guns", shown)
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
                with g.lock:
                    g.recenter(now)
                g.calibrated = True
            elif key == ord("["):
                g.scale_range(1.1)
            elif key == ord("]"):
                g.scale_range(1 / 1.1)
            elif key == ord("t"):
                g.trigger.ratio = min(g.trigger.ratio + 0.04, 0.9)
            elif key == ord("g"):
                g.trigger.ratio = max(g.trigger.ratio - 0.04, 0.3)
            elif key == ord("x"):
                mark = not mark
            elif key == ord("m"):
                modes = AIM_MODES if g.fit is not None else AIM_MODES[1:]
                g.mode = modes[(modes.index(g.mode) + 1) % len(modes)] if g.mode in modes else modes[0]
                g.clear_recent()
            if key in (ord("c"), ord("["), ord("]"), ord("t"), ord("g"), ord("m")):
                save_settings(g)
    finally:
        stop.set()
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
