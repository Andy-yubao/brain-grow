"""从已生成缓存导出实际输入帧；无模型、训练或重新抽样。"""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices=["v5", "v6"], default="v6")
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.version == "v6":
        from tools.preview_data_v6 import export
        export(args.root, args.output or Path("reports/data_v6_samples"))
        return
    args.output = args.output or Path("tmp/dynamic_mnist_v5_preview")
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    if not font_path.exists():
        raise ValueError("请配置支持中文的预览字体")
    title, label, small = [ImageFont.truetype(str(font_path), size) for size in (23, 20, 16)]
    info = json.loads((args.root / f"manifest_{args.version}.json").read_text(encoding="utf-8"))
    motions = ("静止", "上移", "下移", "左移", "右移", "顺时针", "逆时针")
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    with np.load(args.root / f"dynamic_{args.version}.npz") as cache:
        x, y, ids = cache["train_x"], cache["train_y"], cache["train_ids"]
        size = int(np.sqrt(x.shape[-1]))
        tile = size * 4
        step, row_step = tile + 10, tile + 12
        width, height = 130 + 8 * step + 20, 110 + 7 * row_step + 52
        for group, digit in enumerate((2, 5, 9), 1):
            raw_id = int(ids[np.flatnonzero(y[:, 0] == digit)[0]])
            indices = np.flatnonzero(ids == raw_id)
            indices = indices[cache["train_variants"][indices] == 0]
            indices = indices[np.argsort(y[indices, 1])]
            assert len(indices) == 7
            pixels = np.rint(x[indices].reshape(7, 8, size, size) * 255).astype(np.uint8)
            canvas = Image.new("RGB", (width, height), "#f2f4f7")
            draw = ImageDraw.Draw(canvas)
            draw.text((20, 12), f"第{group}组：数字 {digit}   |   原始 MNIST ID {raw_id}   |   训练集 · {args.version}", font=title, fill="#172a42")
            draw.text((20, 48), f"真实{size}×{size}输入，最近邻放大4倍；从左到右第1–8帧。", font=small, fill="#405572")
            for frame in range(8):
                draw.text((130 + frame * step + 30, 78), f"帧 {frame + 1}", font=small, fill="#405572")
            for motion in range(7):
                draw.text((20, 110 + motion * row_step + tile // 3), motions[motion], font=label, fill="#172a42")
                for frame in range(8):
                    image = Image.fromarray(pixels[motion, frame]).resize((tile, tile), Image.Resampling.NEAREST)
                    canvas.paste(image.convert("RGB"), (130 + frame * step, 110 + motion * row_step))
            caption = "v5：各类别共享全圆随机姿态；旋转105°，平移14.1–14.4像素；显示增强轨迹0。"
            draw.text((20, height - 38), caption, font=small, fill="#405572")
            png = args.output / f"group_{group}_digit_{digit}.png"
            canvas.save(png)
            animations = []
            for frame in range(8):
                anim = Image.new("RGB", (450, 100 + 7 * row_step), "#f2f4f7")
                d = ImageDraw.Draw(anim)
                d.text((16, 12), f"{args.version} · 数字 {digit} · 帧{frame+1}/8", font=title, fill="#172a42")
                for motion in range(7):
                    top = 60 + motion * row_step
                    d.text((16, top + tile // 3), motions[motion], font=label, fill="#172a42")
                    image = Image.fromarray(pixels[motion, frame]).resize((tile, tile), Image.Resampling.NEAREST)
                    anim.paste(image.convert("RGB"), (155, top))
                animations.append(anim)
            gif = args.output / f"group_{group}_digit_{digit}.gif"
            animations[0].save(gif, save_all=True, append_images=animations[1:], duration=250, loop=0, disposal=2)
            with Image.open(png) as saved:
                for motion in range(7):
                    for frame in range(8):
                        left, top = 130 + frame * step, 110 + motion * row_step
                        recovered = np.asarray(saved.crop((left, top, left + tile, top + tile)).convert("L"))[::4, ::4]
                        np.testing.assert_array_equal(recovered, pixels[motion, frame])
            records.append({"digit": digit, "raw_id": raw_id, "cache_indices": indices.tolist(),
                            "png": str(png.resolve()), "gif": str(gif.resolve())})
    (args.output / "preview_manifest.json").write_text(json.dumps({
        "version": args.version, "data_seed": info["seed"], "source": str(args.root / f"dynamic_{args.version}.npz"),
        "split": "train", "selection": "first cached raw ID for digits2/5/9", "groups": records,
        "display": "nearest-neighbor4x; GIF250ms/frame is display timing only"}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "groups": records}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
