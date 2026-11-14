"""Joint calibration of the table plane, top-camera re-aims and floor-camera poses.

Unknowns (all lengths in units of the top camera's height h above the table):
- n: table normal in the reference top camera (2 DOF);
- for every other top-camera run r: a rotation R_r (the camera was re-aimed on its mount, assumed
  not translated): a pixel p of run r has the reference-camera ray R_r K^-1 p;
- for every floor-camera run f: pose (R_f, t_f) mapping reference-top-camera coords to floor coords.

Observations:
- bricks seen alone in both cameras in an episode's first frame. The top camera sees the brick face
  nearest to it, the floor camera (standing across the table) the opposite face, so the floor
  bbox bottom-centre is predicted as top bottom-centre + 2 * half_depth along the top camera's
  ground direction, corrected for the actual floor viewing direction;
- sheet edges: in every sheet run, far edge ⟂ left edge and far edge ∥ seam (weighted).
Residuals: floor reprojection error (px, soft-L1) and weighted line-angle errors.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares


def _ndir(theta, phi):
    return np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])


def _rot(w):
    return cv2.Rodrigues(np.asarray(w, float).reshape(3, 1))[0]


@dataclass
class Correspondence:
    episode: int
    top_run: str
    floor_run: str
    top_px: np.ndarray  # bbox bottom-centre in the top image
    floor_px: np.ndarray  # bbox bottom-centre in the floor image


@dataclass
class SheetLines:
    top_run: str
    far: np.ndarray
    left: np.ndarray
    seam: np.ndarray


class JointCalib:
    def __init__(self, Kt, Kf, ref_run: str, top_runs: list[str], floor_runs: list[str],
                 corr: list[Correspondence], lines: list[SheetLines], half_depth: float,
                 w_line_px_per_deg: float = 3.0):
        self.Kt, self.Kf = Kt, Kf
        self.ref = ref_run
        self.top_runs = [r for r in top_runs if r != ref_run]
        self.floor_runs = floor_runs
        self.corr, self.lines = corr, lines
        self.D = half_depth
        self.w = w_line_px_per_deg

    # parameter vector layout: [theta, phi] + 3 per non-ref top run + 6 per floor run
    def unpack(self, x):
        n = _ndir(x[0], x[1])
        i = 2
        R = {self.ref: np.eye(3)}
        for r in self.top_runs:
            R[r] = _rot(x[i:i + 3]); i += 3
        F = {}
        for f in self.floor_runs:
            F[f] = (_rot(x[i:i + 3]), x[i + 3:i + 6]); i += 6
        return n, R, F

    def ray(self, R, p):
        r = R @ np.linalg.solve(self.Kt, [p[0], p[1], 1.0])
        return r

    def plane_point(self, n, R, p):
        r = self.ray(R, p)
        return r / (n @ r)

    def floor_prediction(self, n, R, F, c: Correspondence):
        X = self.plane_point(n, R[c.top_run], c.top_px)
        g = X - n  # camera foot is n * 1 (h = 1); ground direction from the top camera
        g /= np.linalg.norm(g)
        P = X + self.D * g  # footprint centre
        Rf, tf = F[c.floor_run]
        Cf = -Rf.T @ tf  # floor camera centre in reference coords
        e = P - Cf
        e -= (e @ n) * n
        e /= np.linalg.norm(e)
        Q = P - self.D * e  # face nearest to the floor camera
        q = self.Kf @ (Rf @ Q + tf)
        return q[:2] / q[2]

    def line_angles(self, n, R, s: SheetLines):
        def d(l):
            m = R[s.top_run] @ (self.Kt.T @ l)
            v = np.cross(n, m)
            return v / np.linalg.norm(v)
        a = np.degrees(np.arccos(np.clip(abs(d(s.far) @ d(s.left)), 0, 1)))
        b = np.degrees(np.arccos(np.clip(abs(d(s.far) @ d(s.seam)), 0, 1)))
        return a, b

    def residuals(self, x):
        n, R, F = self.unpack(x)
        out = []
        for c in self.corr:
            out.extend(self.floor_prediction(n, R, F, c) - c.floor_px)
        for s in self.lines:
            a, b = self.line_angles(n, R, s)
            out += [self.w * (a - 90.0), self.w * b]
        return np.array(out)

    def init(self, n0):
        x = [np.arccos(n0[2]), np.arctan2(n0[1], n0[0])]
        x += [0.0, 0.0, 0.0] * len(self.top_runs)
        _, R, _ = self.unpack(np.r_[x, np.zeros(6 * len(self.floor_runs))])

        def obs(f):
            obj, img = [], []
            for c in self.corr:
                if c.floor_run != f:
                    continue
                X = self.plane_point(n0, R[c.top_run], c.top_px)
                g = (X - n0) / np.linalg.norm(X - n0)
                obj.append(X + 2 * self.D * g); img.append(c.floor_px)
            return np.array(obj), np.array(img, float)

        # the floor camera was only nudged between runs: solve the best-observed run first and use
        # its pose to start the others (avoids the planar-PnP mirror solution for small runs)
        order = sorted(self.floor_runs, key=lambda f: -len(obs(f)[0]))
        poses = {}
        obj, img = obs(order[0])
        ok, rv, tv, _ = cv2.solvePnPRansac(obj, img, self.Kf, None, reprojectionError=10.0, iterationsCount=1000)
        poses[order[0]] = (rv, tv)
        for f in order[1:]:
            obj, img = obs(f)
            rv0, tv0 = poses[order[0]]
            ok, rv, tv = cv2.solvePnP(obj, img, self.Kf, None, rv0.copy(), tv0.copy(), useExtrinsicGuess=True,
                                      flags=cv2.SOLVEPNP_ITERATIVE)
            poses[f] = (rv, tv)
        for f in self.floor_runs:
            x += list(poses[f][0].ravel()) + list(poses[f][1].ravel())
        return np.array(x, float)

    def fit(self, n0):
        x0 = self.init(n0)
        sol = least_squares(self.residuals, x0, loss="soft_l1", f_scale=8.0, max_nfev=2000)
        return sol

    def corr_errors(self, x):
        n, R, F = self.unpack(x)
        return np.array([np.linalg.norm(self.floor_prediction(n, R, F, c) - c.floor_px) for c in self.corr])
