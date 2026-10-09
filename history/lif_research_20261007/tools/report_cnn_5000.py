"""冻结后比较新旧CNN的共同控制/验证视频，复核真实成本与预测。"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import json
import gzip
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from braingrow.data.moving_mnist_5000 import ROOT, load, sha
from braingrow.data.moving_mnist_v6 import load_cache
from braingrow.evaluation.metrics import evaluate
from braingrow.network.ann_controls import make_model


OLD = Path("experiments/runs/ann_baselines_v6/stacked_cnn")
NEW = Path("experiments/runs/cnn_5000_v6/stacked_cnn")
EVIDENCE = Path("reports/cnn_5000_20261005_evidence")
REPORT = Path("reports/cnn_5000_20261005.json")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def metrics(logits, y):
    result = evaluate(logits, y)
    result["videos"] = len(y)
    predicted = np.column_stack((logits[:, :10].argmax(1), logits[:, 10:].argmax(1)))
    joint = np.zeros((70, 70), dtype=np.int64)
    np.add.at(joint, (y[:, 0] * 7 + y[:, 1], predicted[:, 0] * 7 + predicted[:, 1]), 1)
    result["joint_confusion"] = joint.tolist()
    return result


@torch.no_grad()
def forward(model, x, uint8=False):
    outputs = []
    for start in range(0, len(x), 128):
        block = torch.tensor(x[start:start+128], device="cuda", dtype=torch.float32)
        if uint8:
            block.div_(255)
        output = model(block)
        assert torch.isfinite(output).all()
        outputs.append(output.cpu().numpy())
    return np.concatenate(outputs)


def scalar(m):
    return {k: m[k] for k in ("loss", "digit_accuracy", "motion_accuracy", "joint_accuracy", "videos")}


def main():
    # 二次调用不重复评价；重要预测应原样保留，而非重新消费模型前向。
    if REPORT.exists():
        result = read(REPORT)
        for path, digest in result["evidence_sha256"].items():
            assert sha(path) == digest
        print("报告已存在且重要证据哈希通过，未重新前向。", flush=True)
        return
    old_result, new_result = read(OLD / "result.json"), read(NEW / "result.json")
    assert old_result["status"] in ("completed", "early_stopped")
    assert new_result["status"] in ("completed", "early_stopped")
    old_config, new_config = read(OLD / "config.json"), read(NEW / "config.json")
    assert old_config["model_seed"] == new_config["model_seed"] == 11
    assert old_config["resources"] == new_config["resources"]
    for config in (old_config, new_config):
        assert all(sha(path) == digest for path, digest in config["source_sha256"].items())
    fit, control, manifest = load()
    old_epoch, new_epoch = old_result["best_control"]["epoch"], new_result["best_control"]["epoch"]
    initial_hashes = {str(path): sha(path) for directory in (OLD, NEW)
                      for path in (directory / "best.pt", directory / "resume.pt", directory / "config.json")}
    with np.load(NEW / "evaluations" / f"control_{new_epoch:03d}.npz") as saved:
        new_control = saved["logits"].copy()
        np.testing.assert_array_equal(saved["y"], control[1])
    # 同一扩展控制集上，新模型直接复用已保存预测；只有旧冻结模型新增前向。
    EVIDENCE.mkdir()
    (EVIDENCE / ".gitattributes").write_text(".gitattributes -text\n*.json -text\n*.jsonl -text\n", encoding="utf-8")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    model = make_model("stacked_cnn").cuda().eval()
    started = time.perf_counter()
    model.load_state_dict(torch.load(OLD / "best.pt", map_location="cpu", weights_only=False)["model"])
    old_control = forward(model, control[0], uint8=True)
    validation = load_cache(members=("validation_x", "validation_y", "validation_ids"))
    old_validation = forward(model, validation["validation_x"])
    model.load_state_dict(torch.load(NEW / "best.pt", map_location="cpu", weights_only=False)["model"])
    new_validation = forward(model, validation["validation_x"])
    old_metrics, new_metrics = metrics(old_control, control[1]), metrics(new_control, control[1])
    for key in ("loss", "digit_accuracy", "motion_accuracy", "joint_accuracy"):
        assert abs(new_metrics[key] - new_result["best_control"]["metrics"][key]) < 1e-12
    old_manifest = read("data/manifest_v6.json")
    common = np.isin(control[2], old_manifest["optimization_splits"]["control"])
    fresh = ~common
    assert common.sum() == 1120 and fresh.sum() == 26880
    common_old, common_new = metrics(old_control[common], control[1][common]), metrics(new_control[common], control[1][common])
    for key in ("digit_accuracy", "motion_accuracy", "joint_accuracy"):
        assert common_old[key] == old_result["best_control"]["metrics"][key]
    fresh_old, fresh_new = metrics(old_control[fresh], control[1][fresh]), metrics(new_control[fresh], control[1][fresh])
    validation_old, validation_new = metrics(old_validation, validation["validation_y"]), metrics(new_validation, validation["validation_y"])
    np.savez_compressed(EVIDENCE / "paired_predictions.npz", control_ids=control[2], control_y=control[1],
        old_control_logits=old_control, new_control_logits=new_control,
        validation_ids=validation["validation_ids"], validation_y=validation["validation_y"],
        old_validation_logits=old_validation, new_validation_logits=new_validation)
    for name, directory, config, result in (("old", OLD, old_config, old_result), ("new", NEW, new_config, new_result)):
        (EVIDENCE / f"{name}_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        (EVIDENCE / f"{name}_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        with (directory / "cost_journal.jsonl").open("rb") as source:
            with (EVIDENCE / f"{name}_cost_journal.jsonl.gz").open("wb") as target:
                with gzip.GzipFile(fileobj=target, mode="wb", mtime=0) as compressed:
                    shutil.copyfileobj(source, compressed)
        rows = [read(p) for p in sorted(directory.glob("epoch_*.json"))]
        (EVIDENCE / f"{name}_epochs.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        with np.load(directory / "evaluations" / f"selected_fit_{result['best_control']['epoch']:03d}.npz") as saved:
            verified = evaluate(saved["logits"], saved["y"])
            for key in ("loss", "digit_accuracy", "motion_accuracy", "joint_accuracy"):
                assert abs(verified[key] - result["best_control_fit"][key]) < 1e-12
        costs = result["costs"]
        expected_videos = 4480 if name == "old" else 112000
        assert costs["train_forward_videos"] == costs["train_backward_videos"] == result["completed_epochs"] * expected_videos
        assert costs["optimizer_updates"] == result["completed_epochs"] * (expected_videos // 128)
        checkpoint = torch.load(directory / "resume.pt", map_location="cpu", weights_only=False)
        assert checkpoint["state"]["completed_epochs"] == result["completed_epochs"]
        assert checkpoint["state"]["costs"] == costs
        assert {int(v["step"]) for v in checkpoint["optimizer"]["state"].values()} == {costs["optimizer_updates"]}
        # 独立累计逐批成本账，不能只相信最终result。
        previous = {k: 0 for k in costs}
        for line in (directory / "cost_journal.jsonl").read_text(encoding="utf-8").splitlines():
            cumulative = json.loads(line)["cumulative_costs"]
            assert all(cumulative[k] >= previous[k] for k in previous)
            previous = cumulative
        assert previous == costs
    shutil.copy2(ROOT / "manifest.json", EVIDENCE / "data_manifest.json")
    shutil.copy2(ROOT / "audit.json", EVIDENCE / "data_audit.json")
    assert all(sha(path) == digest for path, digest in initial_hashes.items())
    result = dict(completed=True, old_best_epoch=old_epoch, new_best_epoch=new_epoch,
        old_completed_epochs=old_result["completed_epochs"], new_completed_epochs=new_result["completed_epochs"],
        new_stop_reason=new_result["status"], old_training_original_ids=160, new_training_original_ids=4000,
        new_total_original_ids=5000, expanded_control=dict(old=old_metrics, new=new_metrics),
        old_40_control=dict(old=common_old, new=common_new),
        newly_added_960_control=dict(old=fresh_old, new=fresh_new),
        frozen_validation=dict(old=validation_old, new=validation_new),
        fit_at_selected_best=dict(old=old_result["best_control_fit"], new=new_result["best_control_fit"]),
        control_gain_percentage_points={key: 100*(new_metrics[key]-old_metrics[key]) for key in
                                       ("digit_accuracy", "motion_accuracy", "joint_accuracy")},
        old_training_costs=old_result["costs"], new_training_costs=new_result["costs"],
        post_freeze_costs=dict(old_control_forward_videos=28000, old_validation_forward_videos=1400,
            new_validation_forward_videos=1400, model_metric_evaluations=3,
            optimizer_updates=0, wall_seconds=time.perf_counter()-started,
            note="原40与新增960控制子集从保存预测切片复算，无额外前向"),
        checkpoint_source_sha256=initial_hashes, data_manifest_sha256=sha(ROOT / "manifest.json"),
        interpretation="同架构/初始化根/渲染与反馈规则；每轮训练视频25倍，非等更新数纯ID消融。控制参与选模；原验证历史曾被探索。最终测试仍封存。",
        remaining_sealed_pixels_extracted=0, official_test_reads=0,
        evidence_sha256={str(p): sha(p) for p in EVIDENCE.iterdir() if p.is_file()})
    REPORT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({name: {k: scalar(v) for k,v in result[name].items()}
        for name in ("expanded_control", "old_40_control", "newly_added_960_control", "frozen_validation")}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
