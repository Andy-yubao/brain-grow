"""轻量检查当前文档、配置和Python语法；归档仅检查入口存在，不验证模型。"""
import ast
import json
import re
import tomllib
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT=Path(__file__).resolve().parents[1]
EXCLUDED={".git",".venv","__pycache__",".pytest_cache",".ruff_cache","data","artifacts","tmp","runs","history"}


def main():
    errors=[];links=0;python_count=0
    files=[p for base in (ROOT,ROOT/"docs",ROOT/"tools",ROOT/"tests",ROOT/"src",ROOT/"configs",ROOT/"reports")
           if base.exists()
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
                     "docs/research_plan.md","docs/research_methods.md","docs/literature.md",
                     "history/README.md","history/lif_research_20261007/ARCHIVE.md"):
        if not (ROOT/required).is_file():errors.append(f"缺少文件: {required}")
    try:
        tomllib.loads((ROOT/"pyproject.toml").read_text(encoding="utf-8"))
        for path in files:
            if path.suffix==".json":json.loads(path.read_text(encoding="utf-8"))
    except (OSError,ValueError) as exc:errors.append(str(exc))
    print(json.dumps({"ok":not errors,"documents":len(docs),"local_links":links,
        "python_files_syntax_checked":python_count,"external_links_checked":False,
        "model_validation_performed":False,"historical_contents_checked":False,"errors":errors},ensure_ascii=False,indent=2))
    if errors:raise SystemExit(1)

if __name__=="__main__":main()
