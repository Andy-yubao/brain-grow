"""Inspect runtime availability without installation or model execution."""
from __future__ import annotations
import importlib.metadata
import importlib.util
import json
import platform
import sys

def main() -> None:
    packages = {}
    for module, distribution in [("numpy", "numpy"), ("PIL", "Pillow"), ("pytest", "pytest")]:
        available = importlib.util.find_spec(module) is not None
        version = None
        if available:
            try:
                version = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                pass
        packages[module] = {"available": available, "version": version}
    print(json.dumps({"python": platform.python_version(), "executable": sys.executable, "packages": packages, "scope": "runtime_inventory_only"}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
