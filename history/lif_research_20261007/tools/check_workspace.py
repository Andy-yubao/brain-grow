"""检查当前工作区文档链接、必需文件、配置和Python语法；不验证模型。"""
import ast
import json
import re
import tomllib
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT=Path(__file__).resolve().parents[1]
EXCLUDED={".git",".venv","__pycache__",".pytest_cache",".ruff_cache","data","artifacts","tmp","runs"}


def main():
    errors=[];links=0;python_count=0
    files=[p for base in (ROOT,ROOT/"docs",ROOT/"tools",ROOT/"tests",ROOT/"src")
           for p in (base.iterdir() if base==ROOT else base.rglob("*"))
           if p.is_file() and not any(part in EXCLUDED or part.endswith(".egg-info") for part in p.relative_to(ROOT).parts)]
    docs=[p for p in files if p.suffix==".md"]
    for path in docs:
        text=path.read_text(encoding="utf-8")
        text=re.sub(r"(?ms)^```[^\n]*\n.*?^```[^\n]*$","",text)
        text=re.sub(r"`[^`]*`","",text)
        for raw in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)",text):
            raw=raw.strip().strip("<>");url=urlsplit(raw)
            if url.scheme or url.netloc or raw.startswith("#"):continue
            target=unquote(url.path)
            if not target:continue
            links+=1
            if not (path.parent/target).resolve().exists():errors.append(f"断开链接: {path.relative_to(ROOT)} -> {raw}")
    for path in files:
        if path.suffix==".py":
            try:ast.parse(path.read_text(encoding="utf-8"),filename=str(path));python_count+=1
            except SyntaxError as exc:errors.append(str(exc))
    for required in ("AGENTS.md","README.md","docs/status.md","docs/history.md","docs/current_framework.md",
                     "docs/research_goal.md","docs/data_protocol.md","docs/experiment_plan.md","docs/long_run_prompt.md",
                     "src/braingrow/network/growing.py","src/braingrow/data/rendering.py","tests/test_framework.py",
                     "reports/data_v5_evidence.json"):
        if not (ROOT/required).is_file():errors.append(f"缺少文件: {required}")
    try:
        tomllib.loads((ROOT/"pyproject.toml").read_text(encoding="utf-8"))
        evidence=json.loads((ROOT/"reports/data_v5_evidence.json").read_text(encoding="utf-8"))
        if not evidence.get("ok") or evidence.get("version")!="v5":errors.append("缺少v5审计证据")
    except (OSError,ValueError) as exc:errors.append(str(exc))
    print(json.dumps({"ok":not errors,"documents":len(docs),"local_links":links,
        "python_files_syntax_checked":python_count,"external_links_checked":False,
        "model_validation_performed":False,"errors":errors},ensure_ascii=False,indent=2))
    if errors:raise SystemExit(1)

if __name__=="__main__":main()
