"""仅一次小CPU真实导出、固定删边及候选完整恢复检查。"""
import random
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from tools.train_capacity_pruning import MaskedFC, capture, restore, export_model, load_export, choose_delete


class CapacityPruningTest(unittest.TestCase):
    def test_export_and_rejected_transaction(self):
        torch.manual_seed(11)
        model = MaskedFC([5, 9, 4, 3], .2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        ema = [torch.ones_like(l.weight) for l in model.layers]
        x, y = torch.randn(8, 5), torch.arange(8) % 3
        F.cross_entropy(model(x), y).backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        saved = capture(model, optimizer, ema)
        expected = torch.rand(4)
        choose_delete(model, ema, [5, 5, 2])
        F.cross_entropy(model(x), y).backward()
        for layer, mask in zip(model.layers, model.masks):
            layer.weight.grad.mul_(mask)
        optimizer.step()
        model.enforce(optimizer)
        for layer, mask in zip(model.layers, model.masks):
            self.assertTrue(torch.equal(layer.weight[~mask], torch.zeros_like(layer.weight[~mask])))
        model.eval()
        temporary_root = Path(__file__).resolve().parents[1] / "tmp"
        temporary_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as folder:
            path = Path(folder) / "tiny.bgm"
            export_model(model, path)
            loaded = load_export(path).eval()
            self.assertTrue(torch.equal(model(x), loaded(x)))
        restore(model, optimizer, ema, saved)
        self.assertTrue(torch.equal(torch.rand(4), expected))
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, saved["model"][key]))
        restored = capture(model, optimizer, ema)
        for index, row in saved["optimizer"]["state"].items():
            for key, value in row.items():
                self.assertTrue(torch.equal(value, restored["optimizer"]["state"][index][key]))
        for actual, expected_score in zip(ema, saved["ema"]):
            self.assertTrue(torch.equal(actual, expected_score))


if __name__ == "__main__":
    unittest.main()
