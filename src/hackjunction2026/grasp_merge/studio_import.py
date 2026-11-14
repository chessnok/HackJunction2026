"""Import a dataset folder into the running Studio through its own import API (what the UI does).

``POST /api/projects/{project}/imports/datasets:prepare`` -> ``PUT .../{job}:upload`` (ZIP) ->
worker detects LeRobot v3 and builds a draft (``awaiting_user_review``) ->
``POST .../{job}:finalize`` with the environment -> worker extracts the ZIP into
``~/.local/share/physicalai/datasets/<new id>`` and adds the dataset record. Studio is not stopped
or restarted; existing datasets are not touched (the adapter only extracts the uploaded archive).
"""

from __future__ import annotations

import time
import zipfile
from pathlib import Path

import httpx


def zip_dataset(root: Path, zip_path: Path) -> Path:
    """Stored (uncompressed) ZIP with the dataset at the archive root: video and parquet are compressed."""
    tmp = zip_path.with_suffix(".part")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as z:
        for f in sorted(root.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(root).as_posix())
    tmp.replace(zip_path)
    return zip_path


def _job(c: httpx.Client, job_id: str) -> dict:
    r = c.get(f"/api/jobs/{job_id}")
    r.raise_for_status()
    return r.json()


def _wait(c: httpx.Client, job_id: str, until, timeout_s: float, log) -> dict:
    t0, last = time.time(), None
    while time.time() - t0 < timeout_s:
        j = _job(c, job_id)
        state = (j.get("status"), (j.get("payload") or {}).get("step"))
        if state != last:
            log(f"  job {job_id}: status {state[0]}, step {state[1]}, progress {j.get('progress')}")
            last = state
        if j.get("status") in ("failed", "canceled"):
            raise RuntimeError(f"import job {job_id} {j.get('status')}: {j.get('message')} {j.get('payload')}")
        if until(j):
            return j
        time.sleep(3)
    raise TimeoutError(f"import job {job_id} did not reach the expected state in {timeout_s} s")


def import_dataset(zip_path: Path, name: str, project_id: str, environment_id: str, default_task: str = "",
                   base_url: str = "http://localhost:7860", log=print) -> dict:
    with httpx.Client(base_url=base_url, timeout=httpx.Timeout(60.0, read=1800.0, write=1800.0)) as c:
        names = [d["name"] for p in c.get("/api/projects").json() if p["id"] == project_id for d in p["datasets"]]
        if name in names:
            raise RuntimeError(f"project {project_id} already has a dataset named {name!r}")
        r = c.post(f"/api/projects/{project_id}/imports/datasets:prepare",
                   data={"format_hint": "lerobot_v3", "dataset_name": name})
        r.raise_for_status()
        job_id = r.json()["id"]
        log(f"import job {job_id} prepared; uploading {zip_path.stat().st_size / 1e6:.0f} MB")
        with zip_path.open("rb") as fh:
            r = c.put(f"/api/projects/{project_id}/imports/datasets/{job_id}:upload",
                      files={"archive": (zip_path.name, fh, "application/zip")})
        r.raise_for_status()
        j = _wait(c, job_id, lambda j: (j.get("payload") or {}).get("step") == "awaiting_user_review", 1800, log)
        report = (j.get("payload") or {}).get("validation_report") or {}
        msgs = report.get("messages", [])
        for m in msgs:
            log(f"  validation {m.get('severity')}: {m.get('message')}")
        if any(m.get("severity") == "error" for m in msgs):
            c.post(f"/api/projects/{project_id}/imports/datasets/{job_id}:cancel")
            raise RuntimeError(f"import draft has errors: {msgs}")
        draft = (j.get("payload") or {}).get("dataset_manifest_draft") or {}
        log(f"  draft: {draft.get('statistics')}")
        r = c.post(f"/api/projects/{project_id}/imports/datasets/{job_id}:finalize",
                   json={"environment_id": environment_id, "default_task": default_task})
        r.raise_for_status()
        j = _wait(c, job_id, lambda j: j.get("status") == "completed", 1800, log)
        ds_id = (j.get("payload") or {}).get("result_dataset_id")
        return {"job_id": job_id, "dataset_id": ds_id, "draft_statistics": draft.get("statistics"),
                "validation_messages": msgs}
