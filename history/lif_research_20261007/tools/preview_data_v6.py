"""从v6拟合缓存导出真实像素、8帧图和动画，并逐块回读核对。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from braingrow.data.moving_mnist_v6 import MOTIONS, load_cache, sha


def export(root=Path("data"), output=Path("reports/data_v6_samples")):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((root / "manifest_v6.json").read_text(encoding="utf-8"))
    data = load_cache(root, members=("train_x", "train_y", "train_ids", "train_variants", "train_visible"))
    x, y, ids, variants, visible = (data["train_" + key] for key in ("x", "y", "ids", "variants", "visible"))
    fit_mask = np.isin(ids, manifest["optimization_splits"]["fit"])
    pixels = np.rint(x.reshape(-1, 8, 28, 28) * 255).astype(np.uint8)
    font = Path("C:/Windows/Fonts/msyh.ttc")
    title, label, small = [ImageFont.truetype(str(font), size) for size in (22, 20, 16)]
    tile, step = 112, 122
    overview = Image.new("RGB", (670, 390), "#f2f4f7")
    draw = ImageDraw.Draw(overview)
    draw.text((18, 12), "v6实际输入：数字0–9首个拟合笔迹", font=title, fill="#172a42")
    draw.text((18, 45), "28×28原像素放大4倍；variant 0、静止首帧", font=small, fill="#405572")
    overview_records = []
    for digit in range(10):
        index = int(np.flatnonzero(fit_mask & (y[:, 0] == digit) & (y[:, 1] == 0) & (variants == 0))[0])
        left, top = 20 + (digit % 5) * 130, 92 + (digit // 5) * 150
        overview.paste(Image.fromarray(pixels[index, 0]).resize((tile, tile), Image.Resampling.NEAREST).convert("RGB"), (left, top))
        draw.text((left + 45, top - 25), str(digit), font=label, fill="#172a42")
        overview_records.append(dict(digit=digit, raw_id=int(ids[index]), index=index, pixel_position=[left, top]))
    overview_path = output / "digits_0_to_9.png"
    overview.save(overview_path)
    with Image.open(overview_path) as saved:
        for record in overview_records:
            left, top = record["pixel_position"]
            recovered = np.asarray(saved.crop((left, top, left + tile, top + tile)).convert("L"))[::4, ::4]
            np.testing.assert_array_equal(recovered, pixels[record["index"], 0])
    groups = []
    for digit in (2, 5, 9):
        raw_id = int(ids[np.flatnonzero(fit_mask & (y[:, 0] == digit))[0]])
        indices = np.flatnonzero((ids == raw_id) & (variants == 0))
        indices = indices[np.argsort(y[indices, 1])]
        assert len(indices) == 7 and fit_mask[indices].all()
        canvas = Image.new("RGB", (1130, 1030), "#f2f4f7")
        draw = ImageDraw.Draw(canvas)
        draw.text((20, 12), f"v6 数字{digit} · MNIST ID {raw_id} · 拟合集", font=title, fill="#172a42")
        draw.text((20, 47), "前景18–20.7像素；平移8–10像素；旋转总幅度60–90°；允许最多15%笔画出画", font=small, fill="#405572")
        for frame in range(8):
            draw.text((130 + frame * step + 33, 78), f"帧{frame + 1}", font=small, fill="#405572")
        for motion in range(7):
            top = 110 + motion * 126
            draw.text((20, top + 42), MOTIONS[motion], font=label, fill="#172a42")
            for frame in range(8):
                image = Image.fromarray(pixels[indices[motion], frame]).resize((tile, tile), Image.Resampling.NEAREST)
                canvas.paste(image.convert("RGB"), (130 + frame * step, top))
        draw.text((20, 1000), "实际缓存像素最近邻放大4倍；标签保留原MNIST身份；初始角±15°", font=small, fill="#405572")
        png = output / f"digit_{digit}_frames.png"
        canvas.save(png)
        with Image.open(png) as saved:
            for motion in range(7):
                for frame in range(8):
                    left, top = 130 + frame * step, 110 + motion * 126
                    recovered = np.asarray(saved.crop((left, top, left + tile, top + tile)).convert("L"))[::4, ::4]
                    np.testing.assert_array_equal(recovered, pixels[indices[motion], frame])
        animations = []
        for frame in range(8):
            animation = Image.new("L", (880, 220), 242)
            draw = ImageDraw.Draw(animation)
            draw.text((12, 8), f"v6 数字{digit} · 帧{frame + 1}/8 · 所有运动", font=title, fill=20)
            for motion in range(7):
                left = 12 + motion * step
                draw.text((left + 15, 43), MOTIONS[motion], font=small, fill=45)
                animation.paste(Image.fromarray(pixels[indices[motion], frame]).resize((tile, tile), Image.Resampling.NEAREST), (left, 70))
            draw.text((12, 193), "250ms/帧仅为展示速度；输入仍是实际28×28像素", font=small, fill=45)
            animations.append(animation)
        gif = output / f"digit_{digit}_motions.gif"
        animations[0].save(gif, save_all=True, append_images=animations[1:], duration=250, loop=0, disposal=2, optimize=False)
        with Image.open(gif) as saved:
            assert saved.n_frames == 8
            for frame in range(8):
                saved.seek(frame)
                for motion in range(7):
                    left = 12 + motion * step
                    recovered = np.asarray(saved.crop((left, 70, left + tile, 70 + tile)).convert("L"))[::4, ::4]
                    np.testing.assert_array_equal(recovered, pixels[indices[motion], frame])
        groups.append(dict(digit=digit, raw_id=raw_id, indices=indices.tolist(), png=png.name, gif=gif.name,
                           per_frame_visible=visible[indices].tolist()))
    candidate = np.flatnonzero(fit_mask & (visible.min(1) < .9))
    if not len(candidate):
        candidate = np.flatnonzero(fit_mask & (visible.min(1) < 1))
    index = int(candidate[0])
    crop_canvas = Image.new("RGB", (1020, 220), "#f2f4f7")
    draw = ImageDraw.Draw(crop_canvas)
    draw.text((18, 10), f"实际出画样例：数字{y[index, 0]} · {MOTIONS[y[index, 1]]} · ID {ids[index]} · variant {variants[index]}", font=title, fill="#172a42")
    for frame in range(8):
        left = 18 + frame * step
        draw.text((left + 24, 45), f"帧{frame + 1}", font=small, fill="#405572")
        crop_canvas.paste(Image.fromarray(pixels[index, frame]).resize((tile, tile), Image.Resampling.NEAREST).convert("RGB"), (left, 72))
        draw.text((left + 1, 190), f"出画{1-visible[index, frame]:.1%}", font=small, fill="#405572")
    crop_canvas.save(output / "partial_out_of_frame.png")
    with Image.open(output / "partial_out_of_frame.png") as saved:
        for frame in range(8):
            left = 18 + frame * step
            np.testing.assert_array_equal(np.asarray(saved.crop((left, 72, left + tile, 184)).convert("L"))[::4, ::4], pixels[index, frame])
    result = dict(version="v6", data_seed=manifest["seed"], source_cache_sha256=sha(root / "dynamic_v6.npz"),
                  source_manifest_sha256=sha(root / "manifest_v6.json"), config=manifest["config"], split="fit",
                  selection="每类首个拟合ID；variant0；出画样例按缓存顺序首个裁切>=10%的拟合视频",
                  overview=overview_records, groups=groups,
                  cropped_example=dict(index=index, raw_id=int(ids[index]), digit=int(y[index, 0]),
                                       motion=int(y[index, 1]), variant=int(variants[index]), visible=visible[index].tolist()),
                  png_and_gif_pixel_roundtrip_exact=True, gif_frames=8, gif_duration_ms=250,
                  model_forward_videos=0, optimizer_updates=0,
                  display_timing_note="GIF速度只是展示速度，不代表数据物理帧率或神经仿真时间")
    (output / "preview_manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / ".gitattributes").write_text("*.json -text\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "groups": groups, "pixels_verified": True}, ensure_ascii=False))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("reports/data_v6_samples"))
    args = parser.parse_args()
    export(args.root, args.output)


if __name__ == "__main__":
    main()
