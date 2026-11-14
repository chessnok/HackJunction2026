import glob
import numpy as np
import pandas as pd

D = "/home/intel/.local/share/physicalai/datasets/431a71f7-50c6-4e38-9e38-d85984dde78b"


def load():
    df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(D + "/data/chunk-*/*.parquet"))])
    df = df.sort_values("index").reset_index(drop=True)
    st = np.stack(df["observation.state"].values).astype(np.float64)
    ac = np.stack(df["action"].values).astype(np.float64)
    return df, st, ac


def grasp_events(df, st, ac, min_stall=10, gap=6.0):
    """Frames where the gripper is commanded more closed than it is and has stopped (object in jaws).

    Returns list of (episode, frame_index, row) using the first frame of each stall + 5 frames.
    """
    ev = []
    for ep, d in df.groupby("episode_index"):
        rows = d.index.values
        g = st[rows, 5]
        a = ac[rows, 5]
        stall = (a < g - gap) & (g > 5)
        dg = np.abs(np.gradient(g))
        stall &= dg < 0.5
        # runs
        i = 0
        n = len(rows)
        while i < n:
            if stall[i]:
                j = i
                while j < n and stall[j]:
                    j += 1
                if j - i >= min_stall:
                    k = min(i + 5, j - 1)
                    ev.append((ep, int(d.frame_index.values[k]), int(rows[k]), i, j))
                i = j
            else:
                i += 1
    return ev


if __name__ == "__main__":
    df, st, ac = load()
    ev = grasp_events(df, st, ac)
    np.set_printoptions(suppress=True, precision=1, linewidth=200)
    print(len(ev), "events in", df.episode_index.nunique(), "episodes")
    for e in ev:
        print(e[:2], "stall frames", e[3], e[4], "state", st[e[2]], "act g", ac[e[2], 5].round(1))
    eps_with = {e[0] for e in ev}
    print("episodes without event", sorted(set(df.episode_index.unique()) - eps_with))
