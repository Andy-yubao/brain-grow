"""真实CNN＋GRU的批量执行诊断；不更新或覆盖训练检查点。"""
from __future__ import annotations
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import hashlib
import time
from pathlib import Path
import numpy as np
import torch
from braingrow.network.ann_controls import make_model
from tools.run_layered_gradient import loss_for, training_data, write_json


def main():
    directory = Path("experiments/runs/ann_comparison_v1/cnn_gru")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(11)
    checkpoint_path = directory / "resume.pt"
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = make_model("cnn_gru").cuda()
    model.load_state_dict(saved["model"])
    fit, _, _ = training_data()
    indices = np.random.default_rng(11).permutation(len(fit[0]))[:128]
    x, y = (torch.as_tensor(a[indices], device="cuda") for a in fit[:2])
    observations = []
    gradients = {}
    logits = {}
    for micro in (32, 128):
        model.train()
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        started = time.perf_counter()
        outputs = []
        loss_sum = 0.0
        for start in range(0, len(x), micro):
            output = model(x[start:start + micro])
            loss = loss_for(output, y[start:start + micro])
            count = len(output)
            (loss * (count / len(x))).backward()
            loss_sum += float(loss.detach()) * count
            outputs.append(output.detach())
        torch.cuda.synchronize()
        wall = time.perf_counter() - started
        gradients[micro] = {name: p.grad.detach().cpu().clone() for name, p in model.named_parameters()}
        logits[micro] = torch.cat(outputs).cpu()
        observations.append(dict(micro_batch=micro, effective_batch=128, loss=loss_sum / len(x),
            wall_seconds=wall, forward_videos=128, backward_videos=128, optimizer_updates=0))
        print(f"微批量{micro}：同128视频前向反传耗时{wall:.3f}秒，损失{loss_sum/len(x):.6f}", flush=True)
    errors = {}
    for name in gradients[32]:
        errors[name] = float((gradients[32][name] - gradients[128][name]).abs().max())
        torch.testing.assert_close(gradients[32][name], gradients[128][name], rtol=1e-3, atol=2e-6)
    torch.testing.assert_close(logits[32], logits[128], rtol=3e-5, atol=3e-6)
    assert abs(observations[0]["loss"] - observations[1]["loss"]) < 1e-5
    write_json(directory / "microbatch_probe.json", dict(ok=True, observations=observations,
        gradient_max_abs_error_by_parameter=errors,
        logits_max_abs_error=float((logits[32] - logits[128]).abs().max()),
        checkpoint_completed_epochs=saved["state"]["completed_epochs"],
        checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        gpu=torch.cuda.get_device_name(0), forward_videos=256, backward_videos=256,
        optimizer_updates=0, sealed_test_reads=0, validation_forward_evaluations=0,
        note="工程探针仅用拟合视频，不改训练状态；共享GPU和首次内核开销使耗时只能作本次工程线索。"))


if __name__ == "__main__":
    main()
