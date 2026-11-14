"""Compare all trained runs in one table (quality + speed + size). Also written to compare.md.

    uv run --extra anomalib python -m hackjunction2026.defect.training.compare [--dataset v1]

Columns: per-cycle false alarms (good bricks rejected) and misses (defects accepted) at the
threshold, AUROC, recommended threshold, latency, model size, training time. Pick the model with
the best per-cycle quality first, speed second.
"""

from __future__ import annotations

import argparse
import json

from hackjunction2026.defect.training.common import RUNS_DIR


def fmt(v, pct: bool = False) -> str:
    if v is None:
        return "-"
    return f"{100 * v:.0f}%" if pct else (f"{v:.3f}" if isinstance(v, float) else str(v))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", help="only runs on this dataset")
    a = p.parse_args(argv)

    if not RUNS_DIR.is_dir():
        print(f"no runs yet in {RUNS_DIR} (train a scenario first)")
        return
    lines = [("| run | model | dataset | cycles good/defect | false alarms | misses | AUROC (cycle) | "
              "rec. threshold | latency ms (device) | size MB | train s |"),
             "|" + "---|" * 11]
    for run_dir in sorted(RUNS_DIR.glob("*")):
        if not (run_dir / "run.json").exists():
            continue
        run = json.loads((run_dir / "run.json").read_text())
        if a.dataset and run["dataset"]["name"] != a.dataset:
            continue
        evals = json.loads((run_dir / "eval.json").read_text()) if (run_dir / "eval.json").exists() else {}
        comps = evals or {c: None for c, e in run["exports"].items() if "xml" in e} or {"-": None}
        for comp, ev in comps.items():
            size = run["exports"].get(comp, {}).get("size_mb")
            if ev and ev.get("broken"):
                cells = ["BROKEN (constant score)", "-", "-", "-", "-", f"{ev['latency_ms']['median']} ({ev['device']})"]
            elif ev:
                c, rec = ev["cycles"], ev["recommended"]
                cells = [f"{c['n_good']}/{c['n_defect']}", fmt(c["fpr"], True), fmt(c["fnr"], True),
                         fmt(c["auroc"]), fmt(rec["threshold"]) if rec else "-",
                         f"{ev['latency_ms']['median']} ({ev['device']})"]
            else:
                cells = ["not evaluated", "-", "-", "-", "-", "-"]
            lines.append("| " + " | ".join([run["run_id"], f"{run['model_class']} {comp}", run["dataset"]["name"],
                                            *cells, fmt(size), fmt(run.get("train_seconds"))]) + " |")
    table = "\n".join(lines)
    print(table)
    out = RUNS_DIR.parent / "compare.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(table + "\n")
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
