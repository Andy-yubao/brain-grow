"""逐帧独立大画布审计v6；无模型前向、训练或封存像素提取。"""
from __future__ import annotations

import gzip
import json
import time
from pathlib import Path

import numpy as np

from braingrow.data.moving_mnist_v6 import CONFIG, load_cache, selected_images, sha
from braingrow.data.rendering import foreground, render_pose


def main():
    started = time.perf_counter()
    manifest_path = Path("data/manifest_v6.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    parent = json.loads(Path("data/manifest_v5.json").read_text(encoding="utf-8"))
    assert sha("data/manifest_v5.json") == manifest["parent_v5_manifest_sha256"]
    assert sha("data/dynamic_v5.npz") == manifest["parent_v5_cache_sha256"]
    assert manifest["config"] == CONFIG and manifest["splits"] == parent["splits"]
    id_sets = [set(manifest["splits"][key]) for key in ("train", "validation", "sealed_test")]
    assert all(not id_sets[i] & id_sets[j] for i in range(3) for j in range(i))
    fit, control = (set(manifest["optimization_splits"][key]) for key in ("fit", "control"))
    assert len(fit) == 160 and len(control) == 40 and not fit & control and fit | control == id_sets[0]
    ids_to_read = manifest["splits"]["train"] + manifest["splits"]["validation"]
    images = selected_images(ids_to_read, Path("data"), parent)
    label_path = Path("data/mnist/train-labels-idx1-ubyte.gz")
    assert sha(label_path) == manifest["sources"][label_path.name]["sha256"]
    label_bytes = gzip.decompress(label_path.read_bytes())
    labels = {raw_id: label_bytes[8 + raw_id] for raw_id in ids_to_read}
    data = load_cache()
    all_visibility, all_initial_angles, all_rotation, all_translation = [], [], [], []
    per_motion_visible = [[] for _ in range(7)]
    checked, empty, opposite_reversals, rotation_same_first = 0, 0, 0, 0
    candidate_attempts, visible_pixel_counts = [], []
    for split in ("train", "validation"):
        x, y, ids = (data[split + "_" + key] for key in ("x", "y", "ids"))
        assert x.dtype == np.float32 and x.shape[1:] == (8, 784) and np.isfinite(x).all()
        assert 0 <= x.min() and x.max() <= 1 and set(ids.tolist()) == set(manifest["splits"][split])
        np.testing.assert_array_equal(y[:, 0], [labels[int(raw_id)] for raw_id in ids])
        groups = np.rint(x.reshape(-1, 7, 8, 28, 28) * 255).astype(np.uint8)
        for index, (group, record) in enumerate(zip(groups, manifest["samples"][split], strict=True)):
            positions = slice(index * 7, (index + 1) * 7)
            np.testing.assert_array_equal(y[positions, 1], np.arange(7))
            assert np.all(ids[positions] == record["raw_id"]) and record["digit"] == labels[record["raw_id"]]
            assert np.all(data[split + "_variants"][positions] == record["variant"])
            tile, _ = foreground(images[record["raw_id"]])
            poses, expected_visibility = np.asarray(record["poses_xy_clockwise_degrees"]), np.asarray(record["ink_visible_fraction"])
            np.testing.assert_array_equal(data[split + "_poses"][positions], poses)
            np.testing.assert_array_equal(data[split + "_visible"][positions], expected_visibility)
            np.testing.assert_array_equal(group[0], np.broadcast_to(group[0, 0], group[0].shape))
            assert 18 <= record["foreground_diagonal_pixels"] <= 20.7
            assert np.all((poses[:, 0, 2] >= -15) & (poses[:, 0, 2] < 15))
            all_initial_angles.append(poses[:, 0, 2])
            for motion, axis, sign in ((1, 1, -1), (2, 1, 1), (3, 0, -1), (4, 0, 1)):
                delta = poses[motion, -1] - poses[motion, 0]
                distance = sign * delta[axis]
                assert 8 - 1e-12 <= distance <= 10 + 1e-12 and delta[1 - axis] == 0 and delta[2] == 0
                all_translation.append(float(distance))
            for motion, sign in ((5, 1), (6, -1)):
                angle = sign * (poses[motion, -1, 2] - poses[motion, 0, 2])
                assert 60 - 1e-12 <= angle <= 90 + 1e-12
                np.testing.assert_array_equal(poses[motion, :, :2], np.broadcast_to(poses[motion, 0, :2], (8, 2)))
                all_rotation.append(float(angle))
            for first, second in ((1, 2), (3, 4), (5, 6)):
                opposite_reversals += int(np.array_equal(group[first, ::-1], group[second]))
            rotation_same_first += int(np.array_equal(group[5, 0], group[6, 0]))
            for motion in range(7):
                for frame in range(8):
                    reference = render_pose(tile, record["scale"], poses[motion, frame], padding=20)
                    assert not (reference[0].any() or reference[-1].any() or reference[:, 0].any() or reference[:, -1].any())
                    center = reference[20:48, 20:48]
                    np.testing.assert_array_equal(center, group[motion, frame])
                    visible = float(center.sum(dtype=np.uint64) / reference.sum(dtype=np.uint64))
                    assert abs(visible - expected_visibility[motion, frame]) < 1e-12 and visible >= .85
                    all_visibility.append(visible)
                    per_motion_visible[motion].append(visible)
                    visible_pixel_counts.append(int((center > 25).sum()))
                    empty += int(not center.any())
                    checked += 1
            candidate_attempts.extend(record["candidate_attempts_by_motion"])
            if (index + 1) % 200 == 0:
                print(f"v6独立逐帧审计 {split}：{index + 1}/{len(groups)}条轨迹。", flush=True)
    visibility = np.asarray(all_visibility)
    assert checked == 56000 and empty == 0 and (visibility < 1).any()
    angles = np.asarray(all_initial_angles)
    result = dict(ok=True, version="v6", data_seed=20261004, training_performed=False,
        cache_sha256=sha("data/dynamic_v6.npz"), manifest_sha256=sha(manifest_path),
        renderer_source_sha256=manifest["renderer_source_sha256"], source_v5_cache_unchanged=True,
        source_v5_manifest_unchanged=True, train_videos=5600, fit_videos=4480, control_videos=1120,
        validation_videos=1400, train_raw_ids=200, fit_raw_ids=160, control_raw_ids=40, validation_raw_ids=100,
        original_ids_extracted=300, raw_id_overlap=0, original_labels_verified=True,
        reference_frames_checked=checked, independent_reference_padding_pixels=20,
        cropped_frames=int((visibility < 1).sum()), cropped_frame_fraction=float((visibility < 1).mean()),
        cropped_more_than_10_percent_frames=int((visibility < .9).sum()),
        minimum_ink_visible_fraction=float(visibility.min()), maximum_ink_crop_fraction=float(1 - visibility.min()),
        empty_frames=empty, median_visible_ink_pixels_above_gray25=float(np.median(visible_pixel_counts)),
        visibility_quantiles=np.quantile(visibility, [0, .01, .05, .5, 1]).tolist(),
        cropping_by_motion=[dict(motion=i, frames=len(v), cropped_frames=int((np.asarray(v) < 1).sum()),
                                minimum_visible=float(min(v))) for i, v in enumerate(per_motion_visible)],
        translation_range=[min(all_translation), max(all_translation)], rotation_magnitude_range=[min(all_rotation), max(all_rotation)],
        initial_angle_range=[float(angles.min()), float(angles.max())],
        opposite_exact_pixel_reversal_pairs=opposite_reversals, rotation_identical_first_frames=rotation_same_first,
        candidate_proposals=int(sum(candidate_attempts)), accepted_motion_videos=len(candidate_attempts),
        maximum_attempts_observed=int(max(candidate_attempts)), generation_reference_render_calls=manifest["generation_reference_render_calls"],
        audit_reference_render_calls=checked, generation_wall_seconds=manifest["generation_wall_seconds"],
        audit_wall_seconds=time.perf_counter() - started, sealed_test_pixels_extracted=0, sealed_test_generated=False,
        validation_model_evaluations=0, model_forward_videos=0, model_backward_videos=0, optimizer_updates=0,
        limitation="裁切比例不代表可识别性；可见度拒绝可能产生位置线索，尚未训练或证明准确率。")
    destination = Path("reports/data_v6_evidence.json")
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
