"""仅生成/校验动态MNIST数据，不调用模型或训练。"""
import argparse
import json
from pathlib import Path
from braingrow.data.dynamic import prepare
from braingrow.data.moving_mnist_v6 import prepare as prepare_v6


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices=["v5", "v6"], default="v6")
    parser.add_argument("--root", type=Path, default=Path("data"))
    args = parser.parse_args()
    data = prepare_v6(root=args.root) if args.version == "v6" else prepare(root=args.root, version="v5")
    print(json.dumps({"version": args.version, "training_performed": False,
                      "train_shape": list(data["train_x"].shape),
                      "validation_shape": list(data["validation_x"].shape),
                      "manifest": str(args.root / f"manifest_{args.version}.json")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
