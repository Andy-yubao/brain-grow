"""在查看GRU控制成绩前收束本轮预算，完整保留旧配置和训练状态。"""
from __future__ import annotations
import json
import shutil
from pathlib import Path
import torch
from tools.run_layered_gradient import write_json
from tools.train_ann_comparison import sha, nested_equal


def main():
    directory = Path("experiments/runs/ann_comparison_v1/cnn_gru")
    config_path = directory / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    if result["status"] != "stopped" or not (directory / "STOP").exists():
        raise ValueError("必须先在完整遍边界安全停止本对照")
    if result["costs"]["control_metric_evaluations"] != 0 or result["completed_epochs"] >= 5:
        raise ValueError("本修订须在查看任何GRU控制成绩之前登记")
    if (directory / "config_before_budget_cap.json").exists():
        raise ValueError("已修订，不重复覆盖证据")
    shutil.copy2(config_path, directory / "config_before_budget_cap.json")
    shutil.copy2(directory / "result.json", directory / "result_before_budget_cap.json")
    checkpoint_hashes = {}
    for name in ("resume.pt", "resume_previous.pt"):
        path = directory / name
        if not path.exists():
            continue
        checkpoint_hashes[name] = sha(path)
        shutil.copy2(path, directory / (path.stem + "_before_budget_cap.pt"))
        saved = torch.load(path, map_location="cpu", weights_only=False)
        saved["state"]["maximum_epochs"] = 10
        pending = path.with_name(path.name + ".pending")
        torch.save(saved, pending)
        restored = torch.load(pending, map_location="cpu", weights_only=False)
        if not nested_equal(saved, restored):
            raise RuntimeError("预算字段修订后的模型/优化器/随机状态保存复核失败")
        pending.replace(path)
    amendment = dict(original_epochs=100, new_epochs=10,
        completed_epochs_at_decision=result["completed_epochs"],
        control_evaluations_at_decision=0, checkpoint_sha256_before=checkpoint_hashes,
        reason="共享GPU的另一个任务不得干预；以三模型第10遍共同预算作为本轮主要对照，已完成的两臂100遍结果仅作扩展补充。",
        state_change="只改管理状态maximum_epochs；模型/优化器/随机流/已完成遍数/所有训练参数与执行配置保持。",
        unexecuted_epochs=90, checkpoint_full_payload_serialization_exact=True,
        continuation_note="本轮完成不表示已运行或已收敛100遍；未来扩展须另登记。")
    config["maximum_epochs"] = 10
    config["maximum_train_forward_videos"] = 44800
    config["maximum_train_backward_videos"] = 44800
    config["maximum_control_forward_videos"] = 2240
    config["maximum_fit_forward_videos"] = 4480
    config["budget_amendment"] = amendment
    write_json(config_path, config)
    write_json(directory / "budget_cap.json", amendment)
    (directory / "STOP").unlink()
    print(f"已登记本轮10遍预算，从第{result['completed_epochs']+1}遍继续；原100遍配置和检查点保留。", flush=True)


if __name__ == "__main__":
    main()
