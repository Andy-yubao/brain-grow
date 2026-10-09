"""用真实许可MNIST笔迹检查v6方向、像素可见度、确定性及缓存拒绝。"""
import json
from pathlib import Path

import numpy as np
import pytest

from braingrow.data.moving_mnist_v6 import CONFIG, load_cache, prepare, render, selected_images, sha
from braingrow.data.rendering import foreground, render_pose


@pytest.fixture(scope="module")
def real_digit():
    manifest = json.loads(Path("data/manifest_v5.json").read_text(encoding="utf-8"))
    record = next(r for r in manifest["samples"]["train"] if r["digit"] == 2 and r["variant"] == 0)
    raw_id = record["raw_id"]
    return selected_images([raw_id], Path("data"), manifest)[raw_id], raw_id


def test_real_video_determinism_and_motion(real_digit):
    image, raw_id = real_digit
    x, metadata = render(image, raw_id)
    repeat, repeated_metadata = render(image, raw_id)
    np.testing.assert_array_equal(x, repeat)
    assert metadata == repeated_metadata
    assert x.shape == (7, 8, 784) and x.dtype == np.float32
    assert 0 <= x.min() < x.max() <= 1
    poses = np.asarray(metadata["poses_xy_clockwise_degrees"])
    np.testing.assert_array_equal(x[0], np.broadcast_to(x[0, 0], x[0].shape))
    assert np.all((poses[:, 0, 2] >= -15) & (poses[:, 0, 2] < 15))
    for motion, axis, sign in ((1, 1, -1), (2, 1, 1), (3, 0, -1), (4, 0, 1)):
        delta = poses[motion, -1] - poses[motion, 0]
        assert 8 <= sign * delta[axis] <= 10
        assert delta[1 - axis] == 0 and delta[2] == 0
    assert 60 <= poses[5, -1, 2] - poses[5, 0, 2] <= 90
    assert 60 <= poses[6, 0, 2] - poses[6, -1, 2] <= 90
    with pytest.raises(ValueError, match="根固定"):
        render(image, raw_id, seed=12)


def test_independent_larger_canvas_visibility(real_digit):
    image, raw_id = real_digit
    x, metadata = render(image, raw_id, variant=1)
    tile, _ = foreground(image)
    poses = np.asarray(metadata["poses_xy_clockwise_degrees"])
    expected_visibility = np.asarray(metadata["ink_visible_fraction"])
    cropped = 0
    for motion in range(7):
        for frame in range(8):
            reference = render_pose(tile, metadata["scale"], poses[motion, frame], padding=20)
            center = reference[20:48, 20:48]
            np.testing.assert_array_equal(center, np.rint(x[motion, frame].reshape(28, 28) * 255).astype(np.uint8))
            actual = float(center.sum(dtype=np.uint64) / reference.sum(dtype=np.uint64))
            assert actual >= .85 and abs(actual - expected_visibility[motion, frame]) < 1e-12
            cropped += actual < 1
    assert cropped > 0, "此固定真实笔迹fixture须覆盖允许裁切分支"


def test_cache_corruption_and_incomplete_pair_preserve_files(tmp_path):
    path = tmp_path / "dynamic_v6.npz"
    np.savez_compressed(path, train_y=np.array([[2, 0]]))
    manifest = dict(config=CONFIG, cache_sha256=sha(path))
    (tmp_path / "manifest_v6.json").write_text(json.dumps(manifest), encoding="utf-8")
    np.testing.assert_array_equal(load_cache(tmp_path, members=("train_y",))["train_y"], [[2, 0]])
    path.write_bytes(path.read_bytes() + b"corrupted")
    preserved = path.read_bytes()
    with pytest.raises(ValueError, match="哈希"):
        prepare(tmp_path)
    assert path.read_bytes() == preserved
    (tmp_path / "manifest_v6.json").unlink()
    with pytest.raises(ValueError, match="不完整"):
        prepare(tmp_path)
    assert path.read_bytes() == preserved


def test_source_id_guard_rejects_sealed_pixels():
    manifest = json.loads(Path("data/manifest_v5.json").read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="封存"):
        selected_images([manifest["splits"]["sealed_test"][0]], Path("data"), manifest)
