"""只检查本轮合成缓存与确认集读取边界，不访问 MNIST 或 GPU。"""

import json

import numpy as np
import pytest

from braingrow.datasets import load_synthetic


def test_synthetic_cache_and_confirmation_boundary(tmp_path):
    first = load_synthetic(root=tmp_path)
    assert first.fit_x.shape == (10000, 64)
    assert first.fit_x.dtype == np.float32 and first.fit_y.dtype == np.int64
    assert np.bincount(first.fit_y).tolist() == [5000, 5000]
    assert first.confirm_x.shape == (0, 64) and not first.manifest["confirm_loaded"]
    frozen = load_synthetic(root=tmp_path, include_confirm=True)
    np.testing.assert_array_equal(first.fit_x, frozen.fit_x)
    assert frozen.confirm_x.shape == (2000, 64)
    path = tmp_path / "data" / "structure_synthetic_v1" / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["conditions"]["translated"]["parameter_sha256"] = "invalid"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="参数哈希不一致"):
        load_synthetic(root=tmp_path)
