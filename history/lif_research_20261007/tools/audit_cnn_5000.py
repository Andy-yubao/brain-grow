"""检查5000真实笔迹的数据隔离、完整元数据、旧像素与独立画布。"""
import json
import time
from pathlib import Path

import numpy as np

from braingrow.data.moving_mnist_5000 import ROOT, extract_images, load, raw_sources, select_ids, sha
from braingrow.data.moving_mnist_v6 import CONFIG, load_cache
from braingrow.data.rendering import foreground, render_pose


def main():
    started = time.perf_counter()
    fit, control, manifest = load()
    parent = json.loads(Path("data/manifest_v6.json").read_text(encoding="utf-8"))
    labels = raw_sources(parent)
    assert manifest["splits"] == select_ids(parent, labels)
    assert manifest["parent_manifest_sha256"] == sha("data/manifest_v6.json")
    assert len(manifest["remaining_sealed_ids"]) == 54900
    images = extract_images(manifest["splits"], parent)
    old = load_cache(members=("train_x", "train_y", "train_ids"))
    frames, empty, old_videos, checked_reference = 0, 0, 0, 0
    minimum_visible, cropped = 1., 0
    sample_ids = []
    for pool, arrays, per_digit in (("fit", fit, 400), ("control", control, 100)):
        x, y, ids = arrays
        np.testing.assert_array_equal(y[:, 0], labels[ids])
        np.testing.assert_array_equal(y[:, 1], np.tile(np.arange(7), len(ids) // 7))
        np.testing.assert_array_equal(np.bincount(labels[manifest["splits"][pool]], minlength=10),
                                      np.full(10, per_digit))
        poses = np.load(ROOT / f"{pool}_poses.npy", mmap_mode="r")
        visible = np.load(ROOT / f"{pool}_visible.npy", mmap_mode="r")
        assert poses.shape == (len(ids), 8, 3) and visible.shape == (len(ids), 8)
        assert np.isfinite(poses).all() and np.isfinite(visible).all()
        assert np.all(visible >= .85) and np.all(visible <= 1)
        minimum_visible = min(minimum_visible, float(visible.min()))
        cropped += int((visible < 1).sum())
        assert np.all((poses[:, 0, 2] >= -15) & (poses[:, 0, 2] <= 15))
        delta = poses[:, -1] - poses[:, 0]
        for motion, axis, sign in ((1, 1, -1), (2, 1, 1), (3, 0, -1), (4, 0, 1)):
            d = delta[y[:, 1] == motion]
            assert np.all((sign * d[:, axis] >= 8-1e-12) & (sign * d[:, axis] <= 10+1e-12))
            assert np.all(d[:, 1-axis] == 0) and np.all(d[:, 2] == 0)
        for motion, sign in ((5, 1), (6, -1)):
            d = delta[y[:, 1] == motion]
            assert np.all((sign * d[:, 2] >= 60-1e-12) & (sign * d[:, 2] <= 90+1e-12))
            assert np.all(d[:, :2] == 0)
        for start in range(0, len(x), 1024):
            block = x[start:start+1024]
            empty += int((~block.any(axis=2)).sum())
            frames += len(block) * 8
            still = block[y[start:start+1024, 1] == 0]
            assert np.array_equal(still, np.broadcast_to(still[:, :1], still.shape))
        for raw_id in parent["optimization_splits"][pool]:
            positions = np.flatnonzero(ids == raw_id)
            original = np.flatnonzero(old["train_ids"] == raw_id)
            np.testing.assert_array_equal(x[positions], np.rint(old["train_x"][original] * 255).astype(np.uint8))
            np.testing.assert_array_equal(y[positions], old["train_y"][original])
            old_videos += len(positions)
        # 各池各数字第一个新增笔迹，固定选取不按图像或成绩挑选。
        offset = 16 if pool == "fit" else 4
        for digit in range(10):
            raw_id = manifest["splits"][pool][digit * per_digit + offset]
            sample_ids.append(raw_id)
            tile, bbox = foreground(images[raw_id])
            positions = np.flatnonzero(ids == raw_id)
            for variant in range(4):
                diagonal = np.random.default_rng(np.random.SeedSequence([20261004, raw_id, variant, 6])).uniform(
                    *CONFIG["foreground_diagonal_pixels"])
                scale = diagonal / np.hypot(bbox[2]-bbox[0], bbox[3]-bbox[1])
                for position in positions[variant*7:(variant+1)*7]:
                    for frame in range(8):
                        reference = render_pose(tile, scale, poses[position, frame], padding=20)
                        assert not (reference[0].any() or reference[-1].any()
                                    or reference[:, 0].any() or reference[:, -1].any())
                        center = reference[20:48, 20:48]
                        np.testing.assert_array_equal(center.reshape(784), x[position, frame])
                        ratio = center.sum(dtype=np.uint64) / reference.sum(dtype=np.uint64)
                        assert abs(ratio - visible[position, frame]) < 1e-12
                        checked_reference += 1
    assert empty == 0 and frames == 1120000 and old_videos == 5600
    result = dict(ok=True, manifest_sha256=sha(ROOT / "manifest.json"),
        fit_original_ids=4000, control_original_ids=1000, raw_id_overlap=0,
        original_labels_verified=True, videos=140000, all_frames_checked=frames, empty_frames=empty,
        minimum_visible=minimum_visible, maximum_crop=1-minimum_visible, cropped_frames=cropped,
        old_v6_videos_pixel_identical=old_videos, independent_reference_frames=checked_reference,
        independent_reference_sample_ids=sample_ids, independent_reference_padding=20,
        remaining_sealed_ids=54900, remaining_sealed_pixels_extracted=0, official_test_reads=0,
        original_ids_extracted_for_audit=5000, model_forward_videos=0, optimizer_updates=0,
        wall_seconds=time.perf_counter()-started)
    (ROOT / "audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
