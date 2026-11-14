#!/usr/bin/env python3
"""Fine-tune the "mark visible" classifier (resnet18, all layers) on a Modal GPU, 5-fold CV by brick.

Ephemeral app (no deploy, no volume): only brick crops (white background, no people) are sent, as
JPEG bytes. Same labels and grouping as training/mark_classifier.py; synthetic strokes are extra train
positives only in folds that do not test their parent brick. Returns out-of-fold scores for every real
crop and an ONNX of the model trained on all crops.

Run from the repo root with Studio's Python (modal 1.6, opencv); Modal credentials in ~/.modal.toml:
  third_party/physical-ai-studio/application/backend/.venv/bin/python -m modal run scripts/mark_classifier_modal.py \
      --dataset bricks_z14t --epochs 8
Writes outputs/mark_classifier/<time>_finetune_<dataset>/{oof_scores.csv, model.onnx, result.json}.
"""

from __future__ import annotations

import modal

app = modal.App("hj-mark-classifier-ft")
image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch==2.5.1", "torchvision==0.20.1", "pillow", "numpy<2.3", "onnx")


@app.function(image=image, gpu="T4", timeout=1800)
def train(blobs: list[bytes], labels: list[int], folds: list[int], real: list[bool], epochs: int, n_folds: int) -> dict:
    import io
    import random

    import numpy as np
    import torch
    import torchvision.transforms as T
    from PIL import Image
    from torchvision.models import ResNet18_Weights, resnet18

    torch.manual_seed(0)
    random.seed(0)
    dev = "cuda"
    imgs = [Image.open(io.BytesIO(b)).convert("RGB") for b in blobs]
    norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    class Rot90:
        def __call__(self, im):
            return im.rotate(90 * random.randint(0, 3))

    tf_train = T.Compose([Rot90(), T.RandomHorizontalFlip(), T.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                          T.ColorJitter(0.2, 0.2, 0.15, 0.02), T.RandomApply([T.GaussianBlur(5, (0.1, 1.5))], p=0.3),
                          T.ToTensor(), norm])
    tf_test = T.Compose([T.Resize(224), T.ToTensor(), norm])

    def fit(idx):
        m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        m.fc = torch.nn.Linear(512, 1)
        m = m.to(dev)
        y = np.array([labels[i] for i in idx], np.float32)
        w = np.where(y == 1, 0.5 / max(y.mean(), 1e-3), 0.5 / max(1 - y.mean(), 1e-3))
        sampler = torch.utils.data.WeightedRandomSampler(torch.tensor(w, dtype=torch.double), len(idx), replacement=True)
        opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-4)
        steps = epochs * ((len(idx) + 31) // 32)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=steps)
        loss_f = torch.nn.BCEWithLogitsLoss()
        order = list(sampler)
        for _ in range(epochs):
            order = list(sampler)
            m.train()
            for s in range(0, len(order), 32):
                batch = [idx[j] for j in order[s : s + 32]]
                x = torch.stack([tf_train(imgs[i]) for i in batch]).to(dev)
                t = torch.tensor([labels[i] for i in batch], dtype=torch.float32, device=dev)
                opt.zero_grad()
                loss = loss_f(m(x).squeeze(1), t)
                loss.backward()
                opt.step()
                sched.step()
        return m.eval()

    @torch.no_grad()
    def predict(m, idx):
        out = []
        for s in range(0, len(idx), 64):
            x = torch.stack([tf_test(imgs[i]) for i in idx[s : s + 64]]).to(dev)
            out += torch.sigmoid(m(x).squeeze(1)).cpu().tolist()
        return out

    n = len(imgs)
    oof = [None] * n
    for k in range(n_folds):
        tr = [i for i in range(n) if folds[i] != k]
        te = [i for i in range(n) if folds[i] == k and real[i]]
        m = fit(tr)
        for i, p in zip(te, predict(m, te), strict=True):
            oof[i] = p
    final = fit(list(range(n)))

    class WithSigmoid(torch.nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, x):
            return torch.sigmoid(self.net(x))

    buf = io.BytesIO()
    torch.onnx.export(WithSigmoid(final).cpu().eval(), torch.zeros(1, 3, 224, 224), buf, opset_version=17,
                      input_names=["image"], output_names=["prob"])
    return {"oof": oof, "onnx": buf.getvalue()}


@app.local_entrypoint()
def main(dataset: str = "bricks_z14t", epochs: int = 8, folds: int = 5):
    import csv
    import json
    import time
    from pathlib import Path

    import cv2

    root = Path("captures/anomalib/dataset") / dataset
    sroot = Path("captures/anomalib/dataset") / f"{dataset}-synth"
    with open(root / "index.csv", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] != "excluded"]
    for r in rows:
        r["root"] = root
    synth = []
    if (sroot / "index.csv").exists():
        with open(sroot / "index.csv", newline="") as f:
            synth = [{**r, "root": sroot} for r in csv.DictReader(f) if r["cls"] == "defect"]

    def group(r):
        return r["scene"] if r["source"].startswith("snapshot:") else f"{r['session']}/{r['brick']}"

    groups = sorted({group(r) for r in rows})
    fold_of = {g: i % folds for i, g in enumerate(groups)}  # round-robin over sorted groups
    items = rows + synth
    blobs = [cv2.imencode(".jpg", cv2.imread(str(r["root"] / r["path"])), [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()
             for r in items]
    labels = [1 if r["cls"] == "defect" else 0 for r in rows] + [1] * len(synth)
    real = [True] * len(rows) + [False] * len(synth)
    fl = [fold_of[group(r)] for r in items]
    print(f"{len(rows)} real + {len(synth)} synthetic crops, {sum(len(b) for b in blobs) / 1e6:.1f} MB as JPEG, "
          f"{sum(labels)} positives; {folds} folds over {len(groups)} groups", flush=True)
    t0 = time.time()
    res = train.remote(blobs, labels, fl, real, epochs, folds)
    run = Path("outputs/mark_classifier") / f"{time.strftime('%Y%m%d-%H%M%S')}_finetune_{dataset}"
    run.mkdir(parents=True)
    (run / "model.onnx").write_bytes(res["onnx"])
    with open(run / "oof_scores.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "split", "cls", "label", "view", "source", "scene", "session", "brick", "cycle", "score"])
        for r, s in zip(rows, res["oof"][: len(rows)], strict=True):
            w.writerow([r["path"], r["split"], r["cls"], r["label"], r["view"], r["source"], r["scene"], r["session"],
                        r["brick"], r["cycle"], round(float(s), 5)])
    (run / "result.json").write_text(json.dumps({"dataset": dataset, "epochs": epochs, "folds": folds,
                                                 "seconds": round(time.time() - t0, 1), "gpu": "T4"}, indent=2))
    print(f"done in {time.time() - t0:.0f} s -> {run}")
