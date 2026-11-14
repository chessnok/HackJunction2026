"""Load URDF visual meshes into their link frames."""
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh

from urdf_fk import URDF, T_from

UDIR = Path(URDF).parent


def link_meshes(path=URDF):
    root = ET.parse(path).getroot()
    out = {}
    for link in root.findall("link"):
        parts = []
        for v in link.findall("visual"):
            o = v.find("origin")
            T = T_from([float(x) for x in o.get("xyz").split()], [float(x) for x in o.get("rpy").split()])
            fn = v.find("geometry/mesh").get("filename")
            m = trimesh.load(UDIR / fn, force="mesh")
            m.apply_transform(T)
            parts.append((fn, m))
        out[link.get("name")] = parts
    return out


if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=1)
    LM = link_meshes()
    for ln, parts in LM.items():
        for fn, m in parts:
            b = m.bounds * 1000
            print(ln, fn, "bounds mm", b[0], b[1], "nverts", len(m.vertices))
    # base bottom
    allb = np.vstack([m.vertices for _, m in LM["base_link"]]) * 1000
    print("base min z", allb[:, 2].min(), "max z", allb[:, 2].max(), "x range", allb[:, 0].min(), allb[:, 0].max(),
          "y range", allb[:, 1].min(), allb[:, 1].max())
    # fixed jaw tip in gripper_link frame
    for fn, m in LM["gripper_link"]:
        v = m.vertices * 1000
        i = np.argmin(v[:, 2])
        print("gripper_link", fn, "min z vertex", v[i])
        low = v[v[:, 2] < v[:, 2].min() + 3]
        print("   tip region (within 3mm of min z): mean", low.mean(0), "n", len(low), "x range", low[:, 0].min(),
              low[:, 0].max(), "y range", low[:, 1].min(), low[:, 1].max())
    for fn, m in LM["moving_jaw_so101_v1_link"]:
        v = m.vertices * 1000
        r = np.linalg.norm(v[:, :2], axis=1)
        i = np.argmax(r)
        print("moving jaw farthest from axis (xy):", v[i], "r", r[i])
        print("   bounds", v.min(0), v.max(0))
