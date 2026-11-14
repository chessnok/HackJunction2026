"""Vectorized FK for exploration, built from the parsed URDF."""
import json
import numpy as np
from urdf_fk import parse

CAL = json.load(open("/home/intel/dev/configs/calibration/follower.json"))["calibration"]
NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
J = parse()
CHAIN = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
LINKS = ["shoulder_link", "upper_arm_link", "lower_arm_link", "wrist_link", "gripper_link"]
RMIN = np.array([CAL[n]["range_min"] for n in NAMES], float)
RMAX = np.array([CAL[n]["range_max"] for n in NAMES], float)
MID = (RMIN + RMAX) / 2


def norm_to_ticks(n):
    n = np.asarray(n, float)
    t = np.empty_like(n)
    span = RMAX - RMIN
    t[..., :5] = RMIN[:5] + (n[..., :5] + 100) / 200 * span[:5]
    t[..., 5] = RMIN[5] + n[..., 5] / 100 * span[5]
    return t


def rz(q):
    c, s = np.cos(q), np.sin(q)
    T = np.zeros(q.shape + (4, 4))
    T[..., 0, 0] = c; T[..., 0, 1] = -s; T[..., 1, 0] = s; T[..., 1, 1] = c
    T[..., 2, 2] = 1; T[..., 3, 3] = 1
    return T


def fk_all(q):
    """q: (N,6) URDF radians. Returns dict link -> (N,4,4)."""
    q = np.atleast_2d(q)
    N = len(q)
    T = np.broadcast_to(np.eye(4), (N, 4, 4)).copy()
    out = {"base_link": T.copy()}
    for i, (jn, ln) in enumerate(zip(CHAIN, LINKS)):
        T = T @ J[jn]["T"] @ rz(q[:, i])
        out[ln] = T
    out["gripper_frame_link"] = out["gripper_link"] @ J["gripper_frame_joint"]["T"]
    out["moving_jaw_so101_v1_link"] = out["gripper_link"] @ J["gripper"]["T"] @ rz(q[:, 5])
    return out


def q_from_ticks(t, centers, signs=(1, 1, 1, 1, 1, 1), grip_zero=None):
    t = np.atleast_2d(t)
    q = np.empty_like(t)
    for i in range(5):
        q[:, i] = signs[i] * (t[:, i] - centers[i]) * 2 * np.pi / 4096
    gz = RMIN[5] if grip_zero is None else grip_zero
    q[:, 5] = signs[5] * (t[:, 5] - gz) * 2 * np.pi / 4096
    return q


def apply(T, pts):
    """T (N,4,4), pts (M,3) -> (N,M,3)."""
    return np.einsum("nij,mj->nmi", T[:, :3, :3], pts) + T[:, None, :3, 3]
