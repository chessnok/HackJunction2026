"""Quick URDF parser + FK for exploration (not the final library)."""
import xml.etree.ElementTree as ET
import numpy as np

URDF = ("/home/intel/dev/third_party/physical-ai-studio/application/backend/.venv/lib/python3.13/"
        "site-packages/urdf/so101-follower/so101_follower.urdf")


def rpy_to_R(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def T_from(xyz, rpy):
    T = np.eye(4)
    T[:3, :3] = rpy_to_R(*rpy)
    T[:3, 3] = xyz
    return T


def parse(path=URDF):
    root = ET.parse(path).getroot()
    joints = {}
    for j in root.findall("joint"):
        o = j.find("origin")
        xyz = [float(v) for v in o.get("xyz").split()]
        rpy = [float(v) for v in o.get("rpy").split()]
        ax = j.find("axis")
        axis = [float(v) for v in ax.get("xyz").split()] if ax is not None else [0, 0, 0]
        lim = j.find("limit")
        joints[j.get("name")] = dict(type=j.get("type"), parent=j.find("parent").get("link"),
                                    child=j.find("child").get("link"), T=T_from(xyz, rpy),
                                    axis=np.array(axis),
                                    lower=float(lim.get("lower")) if lim is not None else None,
                                    upper=float(lim.get("upper")) if lim is not None else None)
    return joints


def rot_axis(axis, q):
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(q) * K + (1 - np.cos(q)) * K @ K
    T = np.eye(4)
    T[:3, :3] = R
    return T


def fk(joints, q: dict):
    """Return dict link -> 4x4 world transform (base_link = identity)."""
    out = {"base_link": np.eye(4)}
    changed = True
    while changed:
        changed = False
        for name, j in joints.items():
            if j["parent"] in out and j["child"] not in out:
                T = out[j["parent"]] @ j["T"]
                if j["type"] == "revolute":
                    T = T @ rot_axis(j["axis"], q.get(name, 0.0))
                out[j["child"]] = T
                changed = True
    return out


if __name__ == "__main__":
    J = parse()
    for n, j in J.items():
        print(n, j["type"], j["parent"], "->", j["child"], j["axis"], j["lower"], j["upper"])
    L = fk(J, {})
    np.set_printoptions(suppress=True, precision=4)
    for k, T in L.items():
        print(k, (T[:3, 3] * 1000).round(1), "z-axis", T[:3, 2].round(3))
