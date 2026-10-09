"""项目根运行路径组正则比较：python -m tools.train_path_sparsity。"""
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT/"src") not in sys.path:
    sys.path.insert(0, str(ROOT/"src"))

from braingrow.path_sparsity_experiment import main  # noqa: E402


if __name__ == "__main__":
    main()
