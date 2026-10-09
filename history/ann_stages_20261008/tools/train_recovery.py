"""项目根的边恢复延续实验入口，不修改旧剪枝实验。"""
import os
from pathlib import Path
import sys
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from braingrow.recovery_experiment import main
if __name__ == "__main__":
    main()
