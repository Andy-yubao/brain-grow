"""一次小CPU真实CSR前向检查，不启GPU。"""
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from tools.measure_capacity_deployment import CSRFC, csr_layers, model_storage


class CapacityDeploymentTest(unittest.TestCase):
    def test_int32_csr_matches_dense(self):
        torch.set_num_threads(2)
        rng = np.random.default_rng(11)
        rows, weights, biases = [], [], []
        for output, inputs in ((9, 5), (4, 9), (3, 4)):
            alive = rng.random((output, inputs)) > .6
            values = rng.normal(size=int(alive.sum())).astype(np.float32)
            bias = rng.normal(size=output).astype(np.float32)
            dense = np.zeros((output, inputs), dtype=np.float32)
            dense[alive] = values
            rows.append({"alive": alive, "values": values, "bias": bias, "shape": [output, inputs]})
            weights.append(torch.from_numpy(dense))
            biases.append(torch.from_numpy(bias))
        sparse = CSRFC(csr_layers(rows)).eval()
        inputs = torch.from_numpy(rng.normal(size=(7, 5)).astype(np.float32))
        expected = inputs
        for i, (weight, bias) in enumerate(zip(weights, biases)):
            expected = F.linear(expected, weight, bias)
            if i < 2:
                expected = F.relu(expected)
        with torch.inference_mode():
            actual = sparse(inputs)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-5, rtol=1e-5))
        self.assertEqual(sparse.weight_0.crow_indices().dtype, torch.int32)
        self.assertGreater(model_storage(sparse)["index_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
