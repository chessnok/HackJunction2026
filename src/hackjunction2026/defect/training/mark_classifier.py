"""Supervised "mark visible in this crop?" classifier: an extra vote next to PatchCore (ADR 0006).

    anomalib-env/bin/python -m hackjunction2026.defect.training.mark_classifier --dataset bricks_z14t

- Features: frozen ImageNet resnet18 (torchvision weights), average + max pooled layer3 and layer4 maps
  (1536 values; max pooling keeps small, local marks). The trunk runs as an OpenVINO IR on --device
  (NPU by default), so the CPU stays free.
- Classifier: standardised logistic regression, balanced classes.
- Labels: crop label from the dataset (test/defect = mark visible, by gemini-4-argon + user review;
  good and defect_clean = no mark). Extra positives for training only: the SYNTHETIC strokes of
  <dataset>-synth. Training rows are augmented: hflip, rot90, rot270, colour jitter, blur.
- 5-fold GroupKFold by snapshot / episode: no brick is in both train and test. Out-of-fold scores for
  every real crop -> outputs/mark_classifier/<run>/oof_scores.csv; metrics at 0 / 5 % false positives.
- The final model (all data) is exported as one IR (trunk + linear head + sigmoid) and timed on --device.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import cv2
import numpy as np

DATASETS = Path("captures/anomalib/dataset")
OUT = Path("outputs/mark_classifier")
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
SIZE = 224


def trunk_module():
    import torch
    from torchvision.models import ResNet18_Weights, resnet18

    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval()

    class Trunk(torch.nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, x):
            n = self.net
            x = n.maxpool(n.relu(n.bn1(n.conv1(x))))
            x3 = n.layer3(n.layer2(n.layer1(x)))
            x4 = n.layer4(x3)
            return torch.cat([x3.mean((2, 3)), x3.amax((2, 3)), x4.mean((2, 3)), x4.amax((2, 3))], 1)

    return Trunk(m)


def prep(bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(cv2.resize(bgr, (SIZE, SIZE), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    return ((rgb.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None]


def augs(img: np.ndarray, rng: np.random.Generator) -> list[np.ndarray]:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= rng.uniform(0.85, 1.15)
    hsv[..., 2] = hsv[..., 2] * rng.uniform(0.85, 1.15) + rng.uniform(-10, 10)
    jit = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    return [cv2.flip(img, 1), cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE),
            jit, cv2.GaussianBlur(img, (0, 0), rng.uniform(0.6, 1.4))]


def rate(xs, t):
    return float(np.mean(np.asarray(xs) > t)) if len(xs) else None


def t_for(neg, max_fpr):
    cand = sorted(set(neg))
    ok = [t for t in cand if rate(neg, t) <= max_fpr]
    return float(ok[0]) if ok else float(max(cand))


def auroc(neg, pos):
    if not len(neg) or not len(pos):
        return None
    n, p = np.asarray(neg), np.asarray(pos)
    return float(((p[:, None] > n[None, :]).sum() + 0.5 * (p[:, None] == n[None, :]).sum()) / (len(n) * len(p)))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="bricks_z14t")
    p.add_argument("--device", default="NPU")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--C", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args(argv)

    import openvino as ov
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    torch.set_num_threads(2)
    rng = np.random.default_rng(a.seed)
    run = OUT / f"{time.strftime('%Y%m%d-%H%M%S')}_resnet18lr_{a.dataset}"
    run.mkdir(parents=True)
    root, sroot = DATASETS / a.dataset, DATASETS / f"{a.dataset}-synth"

    with open(root / "index.csv", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] != "excluded"]
    for r in rows:
        r["y"] = 1 if r["cls"] == "defect" else 0
        r["grp"] = r["scene"] if r["source"].startswith("snapshot:") else f"{r['session']}/{r['brick']}"
        r["real"] = True
    synth = []
    if (sroot / "index.csv").exists():
        with open(sroot / "index.csv", newline="") as f:
            for r in csv.DictReader(f):
                if r["cls"] == "defect":
                    r.update(y=1, real=False, root=str(sroot),
                             grp=r["scene"] if r["source"].startswith("snapshot:") else f"{r['session']}/{r['brick']}")
                    synth.append(r)

    # ---- features on the device (trunk as OpenVINO IR)
    t0 = time.time()
    trunk = trunk_module()
    ov_trunk = ov.convert_model(trunk, example_input=torch.zeros(1, 3, SIZE, SIZE), input=[1, 3, SIZE, SIZE])  # static: NPU
    compiled = ov.Core().compile_model(ov_trunk, a.device, {"PERFORMANCE_HINT": "LATENCY"})
    req = compiled.create_infer_request()

    def feat(img):
        return np.asarray(req.infer({0: prep(img)})[compiled.outputs[0]])[0].copy()

    for r in rows + synth:
        img = cv2.imread(str(Path(r.get("root", root)) / r["path"]))
        r["f"] = feat(img)
        r["faug"] = [feat(x) for x in augs(img, rng)]
    print(f"features for {len(rows)} real + {len(synth)} synthetic crops (x6 with augmentations) "
          f"on {a.device}: {time.time() - t0:.0f} s", flush=True)

    # ---- 5-fold CV grouped by brick
    groups = np.array([r["grp"] for r in rows])
    oof = np.zeros(len(rows))
    for k, (tr, te) in enumerate(GroupKFold(n_splits=a.folds).split(rows, groups=groups)):
        test_groups = set(groups[te])
        X, y = [], []
        for i in tr:
            for f in [rows[i]["f"], *rows[i]["faug"]]:
                X.append(f)
                y.append(rows[i]["y"])
        for r in synth:
            if r["grp"] not in test_groups:
                for f in [r["f"], *r["faug"]]:
                    X.append(f)
                    y.append(1)
        sc = StandardScaler().fit(X)
        clf = LogisticRegression(C=a.C, class_weight="balanced", max_iter=3000).fit(sc.transform(X), y)
        oof[te] = clf.predict_proba(sc.transform([rows[i]["f"] for i in te]))[:, 1]
        print(f"fold {k}: train {len(y)} rows ({sum(y)} positive), test {len(te)} crops", flush=True)

    with open(run / "oof_scores.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "split", "cls", "label", "view", "source", "scene", "session", "brick", "cycle", "score"])
        for r, s in zip(rows, oof, strict=True):
            w.writerow([r["path"], r["split"], r["cls"], r["label"], r["view"], r["source"], r["scene"], r["session"],
                        r["brick"], r["cycle"], round(float(s), 5)])

    # ---- metrics (snapshot crops; brick = OR over a snapshot's crops)
    snap = [i for i, r in enumerate(rows) if r["source"].startswith("snapshot:")]
    neg = [oof[i] for i in snap if rows[i]["label"] == "good"]
    pos = [oof[i] for i in snap if rows[i]["cls"] == "defect"]
    ep_neg = [oof[i] for i, r in enumerate(rows) if not r["source"].startswith("snapshot:")]
    bricks: dict[str, dict] = {}
    for i in snap:
        b = bricks.setdefault(rows[i]["scene"], {"score": 0.0, "label": rows[i]["label"], "test": rows[i]["split"] == "test"})
        b["score"] = max(b["score"], float(oof[i]))
    bg = [b["score"] for b in bricks.values() if b["label"] == "good"]
    bd = [b["score"] for b in bricks.values() if b["label"] == "defect"]
    res = {"run": run.name, "dataset": a.dataset, "n": {"good_snapshot_crops": len(neg), "mark_crops": len(pos),
           "episode_crops": len(ep_neg), "good_bricks": len(bg), "defective_bricks": len(bd), "synthetic_positives": len(synth)},
           "auroc": {"crop": auroc(neg, pos), "brick": auroc(bg, bd)}, "at": {}}
    for fpr in (0.0, 0.05):
        t = t_for(neg, fpr)
        res["at"][f"{int(fpr * 100)}%"] = {"threshold": round(t, 4), "crop_fpr": rate(neg, t), "recall_crop": rate(pos, t),
                                           "recall_brick": rate(bd, t), "fpr_brick": rate(bg, t),
                                           "fpr_episode_crops": rate(ep_neg, t)}
    print(json.dumps(res, indent=1), flush=True)

    # ---- final model on all data, exported as one IR, timed on the device
    X, y = [], []
    for r in rows + synth:
        for f in [r["f"], *r["faug"]]:
            X.append(f)
            y.append(r["y"])
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=a.C, class_weight="balanced", max_iter=3000).fit(sc.transform(X), y)
    wvec = clf.coef_[0] / sc.scale_
    bias = float(clf.intercept_[0] - (clf.coef_[0] * sc.mean_ / sc.scale_).sum())

    class Full(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = trunk
            self.head = torch.nn.Linear(len(wvec), 1)
            with torch.no_grad():
                self.head.weight.copy_(torch.tensor(wvec, dtype=torch.float32)[None])
                self.head.bias.fill_(bias)

        def forward(self, x):  # x: normalized RGB NCHW 224
            return torch.sigmoid(self.head(self.trunk(x)))

    full = Full().eval()
    ir = ov.convert_model(full, example_input=torch.zeros(1, 3, SIZE, SIZE), input=[1, 3, SIZE, SIZE])
    ov.save_model(ir, run / "model.xml", compress_to_fp16=True)
    comp = ov.Core().compile_model(str(run / "model.xml"), a.device, {"PERFORMANCE_HINT": "LATENCY"})
    r2 = comp.create_infer_request()
    x = prep(cv2.imread(str(root / rows[0]["path"])))
    for _ in range(10):
        r2.infer({0: x})
    lat = []
    for _ in range(100):
        t1 = time.perf_counter()
        r2.infer({0: x})
        lat.append((time.perf_counter() - t1) * 1000)
    res["export"] = {"xml": str(run / "model.xml"), "device": a.device, "median_ms": round(statistics.median(lat), 2),
                     "p95_ms": round(float(np.percentile(lat, 95)), 2),
                     "threshold_5pct_fpr": res["at"]["5%"]["threshold"],
                     "input": "BGR crop -> resize 224 -> RGB /255 -> ImageNet mean/std, NCHW"}
    (run / "result.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res["export"]), f"\nwritten {run}", flush=True)


if __name__ == "__main__":
    main()
