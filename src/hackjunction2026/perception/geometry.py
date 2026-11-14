"""Small projective-geometry helpers: robust line fits, intersections, homographies."""

from __future__ import annotations

import numpy as np


def line_through(p, q) -> np.ndarray:
    """Homogeneous line through two image points, normalized so a^2 + b^2 = 1."""
    l = np.cross([p[0], p[1], 1.0], [q[0], q[1], 1.0])
    return l / np.hypot(l[0], l[1])


def intersect(l1, l2) -> np.ndarray:
    p = np.cross(l1, l2)
    return p[:2] / p[2]


def fit_line_tls(pts: np.ndarray) -> np.ndarray:
    """Total-least-squares line (a, b, c) with a^2 + b^2 = 1 through Nx2 points."""
    c = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - c)
    n = vt[-1]  # normal
    return np.array([n[0], n[1], -n @ c])


def fit_line_ransac(pts: np.ndarray, thresh: float = 1.5, iters: int = 400, seed: int = 0):
    """RANSAC + TLS refit. Returns (line, inlier mask) or (None, None) for < 2 points."""
    pts = np.asarray(pts, float)
    if len(pts) < 2:
        return None, None
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(iters):
        i, j = rng.choice(len(pts), 2, replace=False)
        if np.allclose(pts[i], pts[j]):
            continue
        l = line_through(pts[i], pts[j])
        inl = np.abs(pts @ l[:2] + l[2]) < thresh
        if best is None or inl.sum() > best.sum():
            best = inl
    line = fit_line_tls(pts[best])
    inl = np.abs(pts @ line[:2] + line[2]) < thresh
    return fit_line_tls(pts[inl]), inl


def apply_h(H: np.ndarray, pts) -> np.ndarray:
    """Apply a 3x3 homography to Nx2 points."""
    pts = np.atleast_2d(np.asarray(pts, float))
    q = np.c_[pts, np.ones(len(pts))] @ H.T
    return q[:, :2] / q[:, 2:3]


def fit_similarity(src: np.ndarray, dst: np.ndarray, scale: bool = True):
    """Least-squares 2D similarity (Umeyama) dst ~ s R src + t. Returns 3x3 matrix."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(b.T @ a / len(src))
    D = np.diag([1.0, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    s = (S * np.diag(D)).sum() / a.var(0).sum() if scale else 1.0
    t = mu_d - s * R @ mu_s
    M = np.eye(3)
    M[:2, :2] = s * R
    M[:2, 2] = t
    return M
