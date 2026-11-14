#!/usr/bin/env python3
"""Local tool to set the cut frame of each OLD episode by hand: three cameras in sync, keyboard driven.

Serves a page on http://127.0.0.1:8765 that plays the OLD dataset's own videos (no re-encoding, nothing leaves
the laptop). Each decision is written at once to the overrides file read by scripts/grasp_merge.py
(`episode,cut_frame[,end_frame]` or `episode,skip`).

  PY=third_party/physical-ai-studio/application/backend/.venv/bin/python
  $PY scripts/cut_tool.py --overrides configs/grasp_merge_overrides.csv
  then open http://127.0.0.1:8765 in Firefox

Keys: Space play/pause · ←/→ 1 frame · Shift+←/→ 10 frames · C start here · E end here (last kept frame)
      R clear the end · S skip episode · U all back to automatic · N / P next / previous · [ ] speed
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[1]
OLD = Path.home() / ".local/share/physicalai/datasets/431a71f7-50c6-4e38-9e38-d85984dde78b"
CAMS = ["top", "floor", "arm"]
LOCK = threading.Lock()


def load_episodes(cuts_csv: Path) -> list[dict]:
    meta = []
    for f in sorted(glob.glob(str(OLD / "meta/episodes/*/*.parquet"))):
        meta += pq.read_table(f).to_pylist()
    meta = {m["episode_index"]: m for m in meta}
    cuts = {int(r["episode"]): r for r in csv.DictReader(cuts_csv.open())} if cuts_csv.exists() else {}
    grip: dict[int, list[float]] = {}
    for f in sorted(glob.glob(str(OLD / "data/*/*.parquet"))):
        t = pq.read_table(f, columns=["episode_index", "frame_index", "observation.state"]).to_pydict()
        for ep, fi, st in zip(t["episode_index"], t["frame_index"], t["observation.state"]):
            grip.setdefault(ep, []).append((fi, round(float(st[5]), 1)))
    out = []
    for ep in sorted(meta):
        m, c = meta[ep], cuts.get(ep, {})
        vids = {}
        for cam in CAMS:
            k = f"observation.images.{cam}"
            vids[cam] = {"url": f"/video/{cam}/{m[f'videos/{k}/chunk_index']}/{m[f'videos/{k}/file_index']}",
                         "from": float(m[f"videos/{k}/from_timestamp"])}
        g = [v for _, v in sorted(grip.get(ep, []))]
        out.append({"episode": ep, "task": (m.get("tasks") or [""])[0], "length": int(m["length"]),
                    "auto_cut": int(c["cut_frame"]) if c.get("cut_frame") else None,
                    "close_frame": int(c["close_frame"]) if c.get("close_frame") else None,
                    "confidence": c.get("confidence", ""), "flags": c.get("flags", ""), "videos": vids, "gripper": g})
    return out


def read_overrides(path: Path) -> dict[int, dict]:
    """``episode,cut_frame[,end_frame]``: cut_frame = int, ``auto`` or ``skip``; end_frame = last kept frame."""
    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            s = line.split("#", 1)[0].strip()
            if s and not s.lower().startswith("episode"):
                parts = [x.strip() for x in s.split(",")]
                out[int(parts[0])] = {"start": parts[1] if len(parts) > 1 and parts[1] else "auto",
                                      "end": parts[2] if len(parts) > 2 and parts[2] else None}
    return out


def write_override(path: Path, ep: int, start: str | None, end: str | None) -> dict[int, dict]:
    """Set one episode's line; start None and end None removes it."""
    with LOCK:
        lines = path.read_text().splitlines() if path.exists() else [
            "# Human corrections to the automatic cut points of scripts/grasp_merge.py (OLD dataset 431a71f7).",
            "# episode,cut_frame[,end_frame]: cut_frame = first kept frame, 'auto' or 'skip';",
            "# end_frame = last kept frame (inclusive), optional. Written by scripts/cut_tool.py."]
        keep = [ln for ln in lines if not re.match(rf"^\s*{ep}\s*,", ln.split("#", 1)[0])]
        if start == "skip":
            keep.append(f"{ep},skip")
        elif start not in (None, "auto") or end is not None:
            keep.append(f"{ep},{start or 'auto'}" + (f",{end}" if end is not None else ""))
        tmp = path.with_suffix(".tmp")
        tmp.write_text("\n".join(keep) + "\n")
        os.replace(tmp, path)
        return read_overrides(path)


class Handler(BaseHTTPRequestHandler):
    episodes: list[dict] = []
    overrides_path: Path = REPO / "configs/grasp_merge_overrides.csv"

    def log_message(self, *a):  # quiet
        pass

    def _send(self, code, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        if self.path == "/api/episodes":
            ov = read_overrides(self.overrides_path)
            return self._send(200, json.dumps({"episodes": self.episodes, "overrides": {str(k): v for k, v in ov.items()},
                                               "fps": 30, "overrides_file": str(self.overrides_path)}).encode(), "application/json")
        m = re.match(r"^/video/(top|floor|arm)/(\d+)/(\d+)$", self.path)
        if m:
            cam, chunk, fi = m.group(1), int(m.group(2)), int(m.group(3))
            f = OLD / f"videos/observation.images.{cam}/chunk-{chunk:03d}/file-{fi:03d}.mp4"
            return self._file(f)
        self._send(404, b"not found", "text/plain")

    def _file(self, f: Path):
        size = f.stat().st_size
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        if rng and (mm := re.match(r"bytes=(\d*)-(\d*)", rng)):
            if mm.group(1):
                start = int(mm.group(1))
                end = int(mm.group(2)) if mm.group(2) else size - 1
            else:
                start = size - int(mm.group(2))
            end = min(end, size - 1)
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        with f.open("rb") as fh:
            fh.seek(start)
            left = end - start + 1
            try:
                while left > 0:
                    chunk = fh.read(min(1 << 20, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass

    def do_POST(self):
        if self.path != "/api/cut":
            return self._send(404, b"not found", "text/plain")
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        ep = int(body["episode"])
        cur = read_overrides(self.overrides_path).get(ep, {"start": "auto", "end": None})
        start, end = cur["start"], cur["end"]
        if body.get("clear"):
            start, end = "auto", None
        elif body.get("skip"):
            start, end = "skip", None
        elif body.get("clear_end"):
            end = None
        elif "end_frame" in body:
            end = str(int(body["end_frame"]))
            if start == "skip":
                start = "auto"
        elif "cut_frame" in body:
            start = str(int(body["cut_frame"]))
        ov = write_override(self.overrides_path, ep, start, end)
        self._send(200, json.dumps({"overrides": {str(k): v for k, v in ov.items()}}).encode(), "application/json")


PAGE = r"""<!doctype html><html lang="ru"><head><meta charset="utf-8"><title>Cut Tool</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#101314;--panel:#1a1f20;--ink:#e8ecec;--muted:#93a0a1;--line:#2d3536;--cut:#ff5252;--auto:#ffb300;--keep:#43c47a;--acc:#7fb2ff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.4 system-ui,sans-serif}
.wrap{display:grid;grid-template-columns:240px 1fr;height:100vh}
aside{overflow:auto;border-right:1px solid var(--line);background:var(--panel)}
aside button{display:flex;justify-content:space-between;width:100%;padding:6px 10px;border:0;border-bottom:1px solid var(--line);background:none;color:var(--ink);font:inherit;text-align:left;cursor:pointer}
aside button.cur{background:#26303a}aside .st{font-size:12px;color:var(--muted)}aside .st.set{color:var(--keep)}aside .st.skip{color:var(--cut)}aside .low{color:var(--auto)}
main{display:grid;grid-template-rows:auto 1fr auto auto;gap:8px;padding:10px;min-width:0;overflow:hidden}
.top{display:flex;gap:14px;align-items:baseline;flex-wrap:wrap}.top b{font-size:18px}.muted{color:var(--muted)}
.vids{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;min-height:0}
.vids figure{margin:0;display:grid;grid-template-rows:1fr auto;min-height:0}
video{width:100%;height:100%;max-height:60vh;object-fit:contain;background:#000;border-radius:6px}
figcaption{font-size:12px;color:var(--muted);text-align:center}
.tl{position:relative;height:46px;background:#0b0e0f;border:1px solid var(--line);border-radius:6px;cursor:pointer}
.tl canvas{position:absolute;inset:0;width:100%;height:100%}
.bar{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
.bar button{font:inherit;padding:7px 12px;border-radius:6px;border:1px solid var(--line);background:var(--panel);color:var(--ink);cursor:pointer}
.bar button.p{background:var(--cut);border-color:var(--cut);color:#fff;font-weight:600}
.big{font-variant-numeric:tabular-nums;font-size:16px}
kbd{border:1px solid var(--line);border-bottom-width:2px;border-radius:4px;padding:0 5px;font-size:12px}
.msg{color:var(--keep)}
</style></head><body><div class="wrap"><aside id="list"></aside><main>
<div class="top"><b id="title"></b><span id="info" class="muted"></span><span id="state" class="big"></span></div>
<div class="vids" id="vids"></div>
<div class="tl" id="tl"><canvas id="cv"></canvas></div>
<div class="bar"><span class="big" id="fr"></span>
<button class="p" id="bc">Начало здесь (C)</button><button class="p" id="be" style="background:#9c3fd6;border-color:#9c3fd6">Конец здесь (E)</button><button id="br">Сбросить конец (R)</button><button id="bs">Выкинуть эпизод (S)</button><button id="bu">Всё авто (U)</button>
<button id="bp">◀ Пред (P)</button><button id="bn">След (N) ▶</button><span id="msg" class="msg"></span>
<span class="muted" style="margin-left:auto"><kbd>Space</kbd> пуск <kbd>←</kbd><kbd>→</kbd> кадр <kbd>Shift</kbd>+стрелки 10 <kbd>[</kbd><kbd>]</kbd> скорость</span></div>
</main></div>
<script>
let D,EPS,OV={},cur=0,fps=30,frame=0,rate=1,vids=[];
const $=id=>document.getElementById(id);
async function init(){D=await (await fetch('/api/episodes')).json();EPS=D.episodes;OV=D.overrides;fps=D.fps;
 const first=EPS.findIndex(e=>!(String(e.episode) in OV)); renderList(); load(first<0?0:first);}
function status(e){const o=OV[String(e.episode)];if(o&&o.start==='skip')return['выкинут','skip'];
 if(!o)return['авто '+(e.auto_cut??'—'),''];const st=o.start==='auto'?'авто '+(e.auto_cut??'—'):o.start;return[st+'–'+(o.end??'конец'),'set'];}
function renderList(){$('list').replaceChildren(...EPS.map((e,i)=>{const b=document.createElement('button');const[s,c]=status(e);
 b.className=i===cur?'cur':'';b.innerHTML=`<span>ep ${e.episode} <span class="${e.confidence==='low'?'low':'muted'}">${e.confidence==='low'?'!':''}</span></span><span class="st ${c}">${s}</span>`;b.onclick=()=>load(i);return b;}));}
function cutOf(e){const o=OV[String(e.episode)];if(!o||o.start==='auto')return e.auto_cut;return o.start==='skip'?null:+o.start;}
function endOf(e){const o=OV[String(e.episode)];return o&&o.end!=null?+o.end:e.length-1;}
const skipped=e=>OV[String(e.episode)]?.start==='skip';
function load(i){cur=i;const e=EPS[i];frame=Math.max(0,(cutOf(e)??e.auto_cut??0)-20);
 $('title').textContent=`эпизод ${e.episode}`;$('info').textContent=`${e.task} · ${e.length} кадров · авто-разрез ${e.auto_cut} · захват закрыт ~${e.close_frame??'—'} ${e.flags?'· '+e.flags:''}`;
 $('vids').replaceChildren();vids=['top','floor','arm'].map(c=>{const f=document.createElement('figure');const v=document.createElement('video');
  v.src=e.videos[c].url;v.muted=true;v.preload='auto';v.dataset.from=e.videos[c].from;v.playsInline=true;
  v.addEventListener('loadedmetadata',()=>seek(frame));f.append(v);const cap=document.createElement('figcaption');cap.textContent=c;f.append(cap);$('vids').append(f);return v;});
 vids[0].addEventListener('timeupdate',onTick);renderList();draw();showState();$('list').children[i]?.scrollIntoView({block:'nearest'});}
function seek(f){const e=EPS[cur];frame=Math.max(0,Math.min(e.length-1,Math.round(f)));for(const v of vids){v.currentTime=+v.dataset.from+frame/fps+0.5/fps;}draw();showState();}
function onTick(){const v=vids[0];if(v.paused)return;const e=EPS[cur];let f=Math.round((v.currentTime-+v.dataset.from)*fps);
 if(f>=e.length-1){f=e.length-1;pause();}frame=f;for(const w of vids.slice(1)){const t=+w.dataset.from+f/fps;if(Math.abs(w.currentTime-t)>0.08)w.currentTime=t;}draw();showState();}
function play(){for(const v of vids){v.playbackRate=rate;v.play().catch(()=>{});}}function pause(){for(const v of vids)v.pause();seek(frame);}
function showState(){const e=EPS[cur];$('fr').textContent=`кадр ${frame} / ${e.length-1} · ${(frame/fps).toFixed(2)} с · захват ${e.gripper[frame]??'—'} · ×${rate}`;
 const[s]=status(e);$('state').textContent='· '+s;}
function draw(){const cv=$('cv'),r=cv.getBoundingClientRect();cv.width=r.width*devicePixelRatio;cv.height=r.height*devicePixelRatio;const g=cv.getContext('2d');g.scale(devicePixelRatio,devicePixelRatio);
 const e=EPS[cur],W=r.width,H=r.height,x=f=>f/(e.length-1)*W;const c=cutOf(e);
 const en=endOf(e);if(c!=null&&!skipped(e)){g.fillStyle='rgba(67,196,122,.22)';g.fillRect(x(c),0,x(en)-x(c),H);} else if(skipped(e)){g.fillStyle='rgba(255,82,82,.18)';g.fillRect(0,0,W,H);}
 g.strokeStyle='#7fb2ff';g.lineWidth=1.5;g.beginPath();e.gripper.forEach((v,i)=>{const y=H-4-(v/100)*(H-8);i?g.lineTo(x(i),y):g.moveTo(x(i),y);});g.stroke();
 const mark=(f,col,w)=>{if(f==null)return;g.fillStyle=col;g.fillRect(x(f)-w/2,0,w,H);};
 mark(e.auto_cut,'#ffb300',2);mark(c,'#ff5252',3);if(en<e.length-1)mark(en,'#d16bff',3);mark(e.close_frame,'#00e5ff',1);mark(frame,'#ffffff',2);
 g.fillStyle='#93a0a1';g.font='11px system-ui';g.fillText('синяя линия: захват (вверх = открыт) · жёлтая: авто · красная: начало · фиолетовая: конец · голубая: захват закрыт',6,12);}
async function post(body){const r=await fetch('/api/cut',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({episode:EPS[cur].episode,...body})});
 OV=(await r.json()).overrides;renderList();draw();showState();$('msg').textContent='сохранено';setTimeout(()=>$('msg').textContent='',1200);}
const setCut=()=>post({cut_frame:frame}),skip=()=>post({skip:true}),undo=()=>post({clear:true}),setEnd=()=>post({end_frame:frame}),clearEnd=()=>post({clear_end:true});
const next=()=>{pause();load(Math.min(EPS.length-1,cur+1));},prev=()=>{pause();load(Math.max(0,cur-1));};
$('bc').onclick=setCut;$('be').onclick=setEnd;$('br').onclick=clearEnd;$('bs').onclick=skip;$('bu').onclick=undo;$('bn').onclick=next;$('bp').onclick=prev;
$('tl').onclick=ev=>{const r=$('tl').getBoundingClientRect();pause();seek((ev.clientX-r.left)/r.width*(EPS[cur].length-1));};
addEventListener('resize',draw);
addEventListener('keydown',ev=>{if(ev.target.tagName==='INPUT')return;const k=ev.key,n=ev.shiftKey?10:1;
 if(k===' '){ev.preventDefault();vids[0].paused?play():pause();}
 else if(k==='ArrowRight'){ev.preventDefault();pause();seek(frame+n);}else if(k==='ArrowLeft'){ev.preventDefault();pause();seek(frame-n);}
 else if(k==='c'||k==='C'||k==='с'||k==='С'){setCut();}else if(k==='e'||k==='E'||k==='у'||k==='У'){setEnd();}else if(k==='r'||k==='R'||k==='к'||k==='К'){clearEnd();}else if(k==='s'||k==='S'||k==='ы'||k==='Ы'){skip();}else if(k==='u'||k==='U'||k==='г'||k==='Г'){undo();}
 else if(k==='n'||k==='N'||k==='т'||k==='Т'){next();}else if(k==='p'||k==='P'||k==='з'||k==='З'){prev();}
 else if(k===']'||k==='ъ'){rate=Math.min(4,rate*2);for(const v of vids)v.playbackRate=rate;showState();}else if(k==='['||k==='х'){rate=Math.max(.25,rate/2);for(const v of vids)v.playbackRate=rate;showState();}});
init();
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--overrides", default=str(REPO / "configs/grasp_merge_overrides.csv"))
    ap.add_argument("--cuts", default=str(REPO / "captures/merge_review/cuts.csv"))
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    Handler.episodes = load_episodes(Path(a.cuts))
    Handler.overrides_path = Path(a.overrides)
    print(f"{len(Handler.episodes)} episodes; overrides -> {a.overrides}\nopen http://127.0.0.1:{a.port}")
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
