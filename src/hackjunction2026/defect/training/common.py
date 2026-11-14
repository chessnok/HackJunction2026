"""Shared code for every training scenario: dataset -> train -> export -> evaluate -> run.json.

Each scenario (train_padim.py, train_patchcore.py, ...) only says which model to build; this file
does the rest the same way for all of them, so their results are comparable.

Outputs (git-ignored):
    outputs/anomalib/runs/<run_id>/run.json      settings, dataset, timings, metrics, export paths
    outputs/anomalib/runs/<run_id>/eval.json     written by evaluate.py (called at the end)
    exports/anomalib/<run_id>/<fp16|int8_ptq>/weights/openvino/model.xml   the deployable model
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

DATASETS_DIR = Path("captures/anomalib/dataset")
RUNS_DIR = Path("outputs/anomalib/runs")
EXPORTS_DIR = Path("exports/anomalib")
COMPRESSIONS = ("fp16", "int8_ptq")


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def parse_args(name: str, argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=f"Train the {name} scenario on a dataset built by make_dataset.")
    p.add_argument("--dataset", default="v1", help=f"dataset name under {DATASETS_DIR}")
    p.add_argument("--accelerator", choices=("auto", "xpu", "cpu"), default="auto",
                   help="auto = Intel GPU (XPU) if available, else CPU")
    p.add_argument("--epochs", type=int, help="override the model's own number of epochs")
    p.add_argument("--max-steps", type=int, help="override the model's own number of training steps")
    p.add_argument("--batch-size", type=int, help="override the scenario's training batch size")
    p.add_argument("--image-size", type=int, default=256, help="must match the dataset crop size")
    p.add_argument("--tag", default="", help="suffix for the run id, e.g. 'coreset001'")
    p.add_argument("--no-export", action="store_true", help="skip OpenVINO export (and evaluation)")
    p.add_argument("--eval-device", default="CPU", help="OpenVINO device for evaluate.py (CPU, GPU, NPU)")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", "4")),
                   help="torch CPU threads (default $OMP_NUM_THREADS or 4). Studio drops recordings when the "
                        "CPU is busy: use 2 (and --num-workers 0) while the team records")
    return p.parse_args(argv)


def pick_accelerator(choice: str) -> str:
    import torch

    if choice == "auto":
        return "xpu" if torch.xpu.is_available() else "cpu"
    if choice == "xpu" and not torch.xpu.is_available():
        msg = ("--accelerator xpu but torch.xpu.is_available() is False. On the station laptop check the "
               "GPU runtime (T-008) and ZE_ENABLE_ALT_DRIVERS (scripts/anomalib.sh sets it).")
        raise SystemExit(msg)
    return choice


def make_datamodule(root: Path, batch_size: int, num_workers: int, seed: int = 42, eval_batch_size: int = 16):
    from anomalib.data import Folder

    has_defects = any((root / "test" / "defect").glob("*.png"))
    common: dict[str, Any] = {
        "name": root.name, "root": root, "normal_dir": "train/good", "normal_test_dir": "test/good",
        "train_batch_size": batch_size, "eval_batch_size": eval_batch_size, "num_workers": num_workers, "seed": seed,
    }
    if has_defects:
        # Threshold from half of the test set (val), metrics on the other half.
        dm = Folder(**common, abnormal_dir="test/defect", test_split_mode="from_dir",
                    val_split_mode="from_test", val_split_ratio=0.5)
    else:
        # No real defects yet: Anomalib paints synthetic anomalies for validation and test.
        dm = Folder(**common, test_split_mode="synthetic", val_split_mode="synthetic", val_split_ratio=0.2)
    return dm, has_defects


def run_scenario(name: str, build_model: Callable[[int], Any], *, batch_size: int = 16,
                 notes: str = "", argv: list[str] | None = None) -> Path:
    """Train one scenario. `build_model(image_size)` returns an Anomalib model."""
    args = parse_args(name, argv)
    from anomalib.deploy import CompressionType, ExportType
    from anomalib.engine import Engine, SingleXPUStrategy, XPUAccelerator

    root = DATASETS_DIR / args.dataset
    if not (root / "train" / "good").is_dir():
        msg = f"{root}/train/good not found: build it with python -m hackjunction2026.defect.cropping.make_dataset"
        raise SystemExit(msg)
    import torch

    torch.set_num_threads(args.threads)
    accelerator = pick_accelerator(args.accelerator)
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}_{name}_{args.dataset}" + (f"_{args.tag}" if args.tag else "")
    run_dir, export_dir = RUNS_DIR / run_id, EXPORTS_DIR / run_id
    run_dir.mkdir(parents=True)
    print(f"[{name}] run {run_id} on {accelerator}")

    dm, has_defects = make_datamodule(root, args.batch_size or batch_size, args.num_workers)
    model = build_model(args.image_size)
    trainer_kwargs: dict[str, Any] = {}
    if args.epochs is not None:
        trainer_kwargs["max_epochs"] = args.epochs
    if args.max_steps is not None:
        trainer_kwargs["max_steps"] = args.max_steps
    if accelerator == "xpu":
        trainer_kwargs.update(accelerator=XPUAccelerator(), strategy=SingleXPUStrategy())
    else:
        trainer_kwargs.update(accelerator="cpu")
    engine = Engine(default_root_dir=run_dir / "lightning", **trainer_kwargs)

    run: dict[str, Any] = {
        "run_id": run_id, "scenario": name, "notes": notes, "git": git_sha(),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "accelerator": accelerator,
        "dataset": {"name": args.dataset, "path": str(root), "has_real_defects": has_defects,
                    "summary": json.loads((root / "dataset.json").read_text())
                    if (root / "dataset.json").exists() else None},
        "args": vars(args), "model_class": type(model).__name__,
        "model_hparams": {k: str(v) for k, v in getattr(model, "hparams", {}).items()},
    }
    t0 = time.time()
    engine.fit(model=model, datamodule=dm)
    run["train_seconds"] = round(time.time() - t0, 1)
    try:
        run["test_metrics"] = {k: float(v) for k, v in engine.test(model=model, datamodule=dm)[0].items()}
    except Exception as exc:  # noqa: BLE001  metrics are informative; the export still matters
        run["test_metrics"] = {"error": repr(exc)}

    run["exports"] = {}
    if not args.no_export:
        # Static batch 1: the robot scores one crop at a time, the NPU wants static shapes, and some
        # models (PatchCore) otherwise get a batch of 2 baked in by the dynamo exporter. The INT8
        # calibration then has to feed single images too.
        export_dm, _ = make_datamodule(root, 1, args.num_workers, eval_batch_size=1)
        for comp in COMPRESSIONS:
            t1 = time.time()
            try:
                xml = engine.export(model=model, export_type=ExportType.OPENVINO, export_root=export_dir / comp,
                                    input_size=(args.image_size, args.image_size),
                                    compression_type=CompressionType(comp), datamodule=export_dm,
                                    onnx_kwargs={"dynamic_shapes": None})
                xml = Path(xml)
                size_mb = sum(f.stat().st_size for f in xml.parent.iterdir()) / 1e6
                run["exports"][comp] = {"xml": str(xml), "size_mb": round(size_mb, 2),
                                        "seconds": round(time.time() - t1, 1)}
            except Exception as exc:  # noqa: BLE001  e.g. INT8 calibration not supported for a model
                run["exports"][comp] = {"error": repr(exc), "trace": traceback.format_exc(limit=3)}
            print(f"[{name}] export {comp}: {run['exports'][comp].get('xml', run['exports'][comp].get('error'))}")

    run["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (run_dir / "run.json").write_text(json.dumps(run, indent=2))
    print(f"[{name}] train {run['train_seconds']} s, test metrics {run['test_metrics']}")

    if run["exports"] and any("xml" in e for e in run["exports"].values()):
        from hackjunction2026.defect.training.evaluate import evaluate_run

        evaluate_run(run_dir, device=args.eval_device)
    print(f"[{name}] done: {run_dir}")
    return run_dir
