"""仅补Windows文件短锁有限重试；原训练数学实现、源SHA与预算保持。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time


ROOT=Path(__file__).resolve().parents[1]


def retry_replace(original,source,target,events):
    for attempt in range(21):
        try:
            return original(source,target)
        except OSError as error:
            if getattr(error,"winerror",None) not in (5,32) or attempt==20:
                raise
            before=time.perf_counter()
            time.sleep(.05)
            events.append({"path":str(target),"winerror":error.winerror,"attempt":attempt+1,"waiting_seconds":time.perf_counter()-before})


def check():
    events=[]
    calls=[]
    def locked(source,target):
        calls.append(1)
        if len(calls)<3:
            error=PermissionError("模拟Windows短锁")
            error.winerror=32
            raise error
    retry_replace(locked,"source","target",events)
    assert len(calls)==3 and len(events)==2
    with tempfile.TemporaryDirectory(dir=ROOT/"tmp") as path:
        source,target=Path(path)/"source",Path(path)/"target"
        source.write_text("新内容",encoding="utf-8")
        target.write_text("旧内容",encoding="utf-8")
        retry_replace(os.replace,source,target,events)
        assert target.read_text(encoding="utf-8")=="新内容" and not source.exists()
    print(json.dumps({"ok":True,"simulated_short_lock_retries":2,"real_temporary_replace":True,"gpu":False}))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--check",action="store_true")
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    if args.check:
        check()
        return
    if not args.resume:
        raise ValueError("工程包装仅用于既有--resume，不创建新实验")
    from tools import train_translated_edges
    folder=ROOT/"experiments/runs/h1_translated_edges_v1/mnist_circular"
    paths=[Path(__file__),ROOT/"tools/train_translated_edges.py",ROOT/"configs/h1_translated_edges_v1.json"]
    events=[]
    record={"started_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"source_sha256":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            "rule":"仅os.replace WinError5/32，最多20次等待，每次50ms；等待计入原Budget", "events":events}
    log=folder/("engineering_io_"+time.strftime("%Y%m%d_%H%M%S",time.gmtime())+".json")
    log.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding="utf-8")
    original=os.replace
    os.replace=lambda source,target:retry_replace(original,source,target,events)
    try:
        train_translated_edges.main()
        record["completed"]=True
    finally:
        os.replace=original
        record["ended_utc"]=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())
        record["retry_count"]=len(events)
        record["waiting_seconds"]=sum(e["waiting_seconds"] for e in events)
        log.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
