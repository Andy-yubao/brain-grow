"""显示工作区状态；实验使用独立 module 入口。"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="BrainGrow 工作区状态与研究入口")
    parser.add_argument("--status", action="store_true", help="显示当前工作区状态")
    args = parser.parse_args()
    if args.status:
        status = Path(__file__).resolve().parents[2] / "docs" / "status.md"
        if not status.is_file():
            parser.error("找不到工作区状态文件；请在原工作区以 editable 安装使用。")
        print(status.read_text(encoding="utf-8"))
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
