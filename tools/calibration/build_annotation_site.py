#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]


HTML = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>MMAUD 双鱼眼中心标注</title>
<style>
body{font-family:sans-serif;background:#171717;color:#eee;margin:20px}button{margin:4px;padding:8px 14px}
#frame{max-width:100%;cursor:crosshair;border:2px solid #666}#status{white-space:pre-wrap;margin:10px 0}
.left{color:#53d8fb}.right{color:#ffca58}.warn{color:#ff7777}
</style></head><body>
<h2>MMAUD 双鱼眼 UAV 中心标注</h2>
<p>直接点击 UAV：左半自动写入 left 的局部坐标，右半自动写入 right。红/黄十字分别表示左右标注。</p>
<div><button onclick="move(-1)">上一帧 B</button><button onclick="move(1)">下一帧 N</button>
<button onclick="markInvisible('left')">左不可见</button><button onclick="markInvisible('right')">右不可见</button>
<button onclick="clearFrame()">清除此帧</button><button onclick="exportCsv()">导出 CSV</button></div>
<div id="status"></div><canvas id="frame"></canvas>
<script>
const records=__RECORDS__; let index=0; const state=JSON.parse(localStorage.getItem('mmaud_center_state')||'{}');
const canvas=document.getElementById('frame'),ctx=canvas.getContext('2d'),status=document.getElementById('status');
function key(r,c){return r.sample_id+'|'+c} function save(){localStorage.setItem('mmaud_center_state',JSON.stringify(state))}
function draw(){const r=records[index],im=new Image(); im.onload=()=>{canvas.width=im.width;canvas.height=im.height;ctx.drawImage(im,0,0);ctx.strokeStyle='#aaa';ctx.beginPath();ctx.moveTo(im.width/2,0);ctx.lineTo(im.width/2,im.height);ctx.stroke();
 for(const c of ['left','right']){const a=state[key(r,c)];if(a&&a.visible===1){const ox=c==='right'?im.width/2:0,x=(a.u/1280)*(im.width/2)+ox,y=(a.v/960)*im.height;ctx.strokeStyle=c==='left'?'#00e5ff':'#ffd000';ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(x-12,y);ctx.lineTo(x+12,y);ctx.moveTo(x,y-12);ctx.lineTo(x,y+12);ctx.stroke()}}
 }; im.src=r.thumbnail; const l=state[key(r,'left')],rr=state[key(r,'right')];status.textContent=`${index+1}/${records.length}  ${r.sample_id}  ${r.class_name}\nleft=${JSON.stringify(l||null)}  right=${JSON.stringify(rr||null)}`}
canvas.onclick=e=>{const r=records[index],rect=canvas.getBoundingClientRect(),x=(e.clientX-rect.left)*canvas.width/rect.width,y=(e.clientY-rect.top)*canvas.height/rect.height,c=x<canvas.width/2?'left':'right',local=x%(canvas.width/2);state[key(r,c)]={visible:1,u:local/(canvas.width/2)*1280,v:y/canvas.height*960};save();draw()}
function markInvisible(c){state[key(records[index],c)]={visible:0,u:'',v:''};save();draw()}
function clearFrame(){delete state[key(records[index],'left')];delete state[key(records[index],'right')];save();draw()}
function move(d){index=Math.max(0,Math.min(records.length-1,index+d));draw()}
document.onkeydown=e=>{if(e.key.toLowerCase()==='n')move(1);if(e.key.toLowerCase()==='b')move(-1)}
function exportCsv(){let rows=['sample_id,sequence_id,class_name,image_path,image_time,camera,u,v,visible,confidence,source,calibration_split'];for(const r of records)for(const c of ['left','right']){const a=state[key(r,c)]||{};rows.push([r.sample_id,r.sequence_id,r.class_name,r.image_path,r.image_time,c,a.u??'',a.v??'',a.visible??'',1.0,'manual_browser',r.calibration_split].join(','))}const blob=new Blob([rows.join('\n')+'\n'],{type:'text/csv'}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='center_annotations.csv';a.click()}
draw();</script></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an offline browser center-annotation page")
    parser.add_argument(
        "--annotations", type=Path, default=PROJECT_ROOT / "calibration/center_annotations.csv"
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("/home/jasoncui/datasets/MMAUD/v1"))
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "calibration/annotation_site")
    args = parser.parse_args()
    with args.annotations.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    unique = {}
    for row in rows:
        unique.setdefault(
            row["sample_id"],
            {key: row[key] for key in (
                "sample_id", "sequence_id", "class_name", "image_path", "image_time",
                "calibration_split",
            )},
        )
    image_dir = args.output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for count, row in enumerate(unique.values(), start=1):
        source = args.dataset_root / row["image_path"]
        destination = image_dir / f"{row['sample_id']}.jpg"
        if not destination.exists():
            with Image.open(source) as image:
                image.load()
                image = image.convert("RGB")
                image.thumbnail((1280, 480), Image.Resampling.LANCZOS)
                image.save(destination, quality=88, optimize=True)
        record = dict(row)
        record["thumbnail"] = f"images/{destination.name}"
        records.append(record)
        if count % 50 == 0:
            print(f"prepared {count}/{len(unique)}")
    html = HTML.replace("__RECORDS__", json.dumps(records, ensure_ascii=False))
    (args.output_dir / "index.html").write_text(html, encoding="utf-8")
    print((args.output_dir / "index.html").resolve())


if __name__ == "__main__":
    main()
