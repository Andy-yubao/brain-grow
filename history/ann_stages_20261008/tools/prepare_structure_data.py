"""一次准备本轮数据；默认不提取 MNIST 探索性确认像素。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from braingrow.datasets import load_bundle  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--task", choices=("all", "synthetic_translated", "synthetic_fixed", "mnist"), default="all")
    parser.add_argument("--include-confirm", action="store_true",
                        help="仅在全部候选冻结后使用；将显式提取/读取确认数据")
    args = parser.parse_args()
    tasks = ("synthetic_translated", "synthetic_fixed", "mnist") if args.task == "all" else (args.task,)
    summary = {}
    for task in tasks:
        bundle = load_bundle(task, root=args.root, include_confirm=args.include_confirm)
        record = dict(fit_shape=list(bundle.fit_x.shape), select_shape=list(bundle.select_x.shape),
                      confirm_shape=list(bundle.confirm_x.shape), seed=bundle.manifest["seed"],
                      confirm_loaded=bundle.manifest["confirm_loaded"])
        if task == "mnist":
            splits = bundle.manifest["splits"]
            sets = [set(splits[name]) for name in ("fit", "select", "confirm")]
            if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
                raise ValueError("MNIST 原 ID 集合存在重叠")
            record["id_isolation"] = True
            record["permissions_used"] = bundle.manifest["permissions_used"]
        summary[task] = record
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
