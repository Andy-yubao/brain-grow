"""并行生成用户授权5000笔迹，保留原v6缓存；不运行模型。"""
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from braingrow.data.moving_mnist_5000 import ROOT, extract_images, load, raw_sources, select_ids, sha
from braingrow.data.moving_mnist_v6 import CONFIG, render


def render_id(task):
    raw_id, image = task
    xs, poses, visible = [], [], []
    calls = 0
    for variant in range(4):
        x, meta = render(image, raw_id, variant)
        xs.append(np.rint(x * 255).astype(np.uint8))
        poses.append(np.asarray(meta["poses_xy_clockwise_degrees"]))
        visible.append(np.asarray(meta["ink_visible_fraction"]))
        calls += meta["generation_reference_render_calls"]
    return raw_id, np.concatenate(xs), np.concatenate(poses), np.concatenate(visible), calls


def prepare(workers):
    if ROOT.exists():
        load()
        print("5000笔迹缓存已存在且哈希通过，未重新生成。", flush=True)
        return
    pending = ROOT.with_name(ROOT.name + "_pending")
    if pending.exists():
        raise ValueError("扩展缓存存在未完成现场，拒绝覆盖")
    parent_path = Path("data/manifest_v6.json")
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    labels = raw_sources(parent)
    splits = select_ids(parent, labels)
    images = extract_images(splits, parent)
    pending.mkdir()
    arrays = {}
    for pool, ids in splits.items():
        n = len(ids) * 28
        for member, dtype, shape in (("x", np.uint8, (n, 8, 784)),
                                     ("y", np.int64, (n, 2)), ("ids", np.int64, (n,)),
                                     ("poses", np.float64, (n, 8, 3)),
                                     ("visible", np.float64, (n, 8))):
            arrays[pool, member] = np.lib.format.open_memmap(
                pending / f"{pool}_{member}.npy", mode="w+", dtype=dtype, shape=shape)
    tasks = [(i, images[i]) for ids in splits.values() for i in ids]
    locations = {i: (pool, index) for pool, ids in splits.items() for index, i in enumerate(ids)}
    started = time.perf_counter()
    calls = 0
    minimum_visible = 1.
    cropped = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for count, (raw_id, x, poses, visible, n_calls) in enumerate(executor.map(render_id, tasks, chunksize=4), 1):
            pool, index = locations[raw_id]
            sl = slice(index * 28, (index + 1) * 28)
            arrays[pool, "x"][sl] = x
            arrays[pool, "y"][sl] = np.column_stack((np.full(28, labels[raw_id]), np.tile(np.arange(7), 4)))
            arrays[pool, "ids"][sl] = raw_id
            arrays[pool, "poses"][sl] = poses
            arrays[pool, "visible"][sl] = visible
            minimum_visible = min(minimum_visible, float(visible.min()))
            cropped += int((visible < 1).sum())
            calls += n_calls
            if count % 250 == 0:
                print(f"5000笔迹：{count}/5000，耗时{time.perf_counter()-started:.1f}秒。", flush=True)
    for array in arrays.values():
        array.flush()
    arrays.clear()
    # 关闭Windows内存映射，之后目录原子更名。
    del array
    files = {p.name: sha(p) for p in pending.glob("*.npy")}
    manifest = dict(version="v6_5000", config=CONFIG, seed=20261004, splits=splits,
        held_validation_ids=parent["splits"]["validation"],
        remaining_sealed_ids=sorted(set(parent["splits"]["sealed_test"]) - set(splits["fit"]) - set(splits["control"])),
        stage_transition="用户于2026-10-05授权扩展到5000原图，4800个旧封存训练池ID重划为此次拟合/控制；剩余ID继续封存",
        original_ids_extracted=5000, newly_authorized_previous_sealed_ids=4800,
        remaining_sealed_pixels_extracted=0, official_test_reads=0,
        fit_videos=112000, control_videos=28000, variants=4, motions=7,
        storage="uint8原像素；输入GPU时转float32/255，与旧v6逐像素相同",
        parent_manifest_sha256=sha(parent_path), sources=parent["sources"], file_sha256=files,
        renderer_sha256=sha("src/braingrow/data/moving_mnist_v6.py"),
        generator_sha256=sha(__file__), generation_wall_seconds=time.perf_counter()-started,
        generation_reference_render_calls=calls, frames=1120000, cropped_frames=cropped,
        minimum_visible=minimum_visible, maximum_crop=1-minimum_visible, workers=workers,
        model_forward_videos=0, optimizer_updates=0)
    (pending / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    pending.replace(ROOT)
    print("5000笔迹生成完成；原v6数据未改动。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    prepare(args.workers)
