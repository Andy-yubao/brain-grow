"""全量v5几何/像素审计；不训练、不使用封存样本。"""
import json
from pathlib import Path
import numpy as np
from braingrow.data.moving_mnist_v5 import prepare
from braingrow.data.dynamic import load_mnist
from braingrow.data.rendering import foreground, render_pose, pose_bounds
def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


def main():
    data = prepare()
    manifest = json.loads(Path("data/manifest_v5.json").read_text(encoding="utf-8"))
    images, labels, _ = load_mnist(Path("data/mnist"))
    checked, cropped, empty, reversals, shared = 0, 0, 0, 0, 0
    angles, translations, margin = [], [], 28.
    idsets = [set(manifest["splits"][k]) for k in ("train","validation","sealed_test")]
    assert all(not idsets[i]&idsets[j] for i in range(3) for j in range(i))
    for split in ("train","validation"):
        np.testing.assert_array_equal(data[split+"_y"][:,0], labels[data[split+"_ids"]])
        groups = data[split+"_x"].reshape(-1,7,8,28,28)
        for group, record in zip(groups, manifest["samples"][split], strict=True):
            tile, bbox = foreground(images[record["raw_id"]])
            poses = np.asarray(record["poses_xy_clockwise_degrees"])
            bounds = pose_bounds(bbox, record["scale"], poses)
            margin = min(margin, float(bounds[...,:2].min()), float(28-bounds[...,2:].max()))
            angles.append(poses[:,0,2])
            np.testing.assert_array_equal(group[0], np.broadcast_to(group[0,0],group[0].shape))
            assert np.isclose(poses[5,-1,2]-poses[5,0,2],105)
            assert np.isclose(poses[6,-1,2]-poses[6,0,2],-105)
            shared += int(np.array_equal(group[5,0],group[6,0]))
            for a,b in ((1,2),(3,4),(5,6)):
                reversals += int(np.array_equal(group[a,::-1],group[b]))
            for motion,axis,sign in ((1,1,-1),(2,1,1),(3,0,-1),(4,0,1)):
                delta = sign*(poses[motion,-1,axis]-poses[motion,0,axis])
                assert 14.1-1e-9 <= delta <= 14.4+1e-9
                translations.append(float(delta))
            for pixels,pose in zip(group.reshape(-1,28,28), poses.reshape(-1,3), strict=True):
                reference = render_pose(tile,record["scale"],pose,padding=8)
                center = reference[8:36,8:36]
                np.testing.assert_array_equal(center,np.rint(pixels*255).astype(np.uint8))
                cropped += int(reference.sum(dtype=np.int64)!=center.sum(dtype=np.int64))
                empty += int(not center.any())
                checked += 1
    assert cropped == empty == reversals == shared == 0 and margin >= 1-1e-8
    angles = np.asarray(angles)
    result = {"ok":True,"version":"v5","data_seed":20261004,"training_performed":False,
              "train_videos":len(data["train_x"]),"validation_videos":len(data["validation_x"]),
              "train_raw_ids":len(idsets[0]),"validation_raw_ids":len(idsets[1]),"raw_id_overlap":0,
              "reference_frames_checked":checked,"cropped_frames":cropped,"empty_frames":empty,
              "opposite_exact_reversal_pairs":reversals,"rotation_shared_first_frames":shared,
              "minimum_geometry_margin_pixels":margin,"translation_range":[min(translations),max(translations)],
              "start_angle_range_by_motion":[[float(angles[:,i].min()),float(angles[:,i].max())] for i in range(7)],
              "positive_initial_angle_fraction_by_motion":(angles>0).mean(axis=0).tolist(),
              "sealed_test_generated":False,"limitation":"全圆姿态不保证无静态捷径；大幅净平移的端点位置约束仍存在；旋转6/9身份可能有内在歧义"}
    write_json(Path("reports/data_v5_evidence.json"),result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
