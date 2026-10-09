"""唯一小CPU配额与真实删边行为检查。"""
import unittest

import torch

from tools.train_capacity_edge_curve import allocation
from tools.train_capacity_pruning import MaskedFC, choose_delete


class CapacityEdgeCurveTest(unittest.TestCase):
    def test_fixed_largest_remainder_and_real_mask(self):
        initial = [2014106,8056420,328835,1050]
        previous = initial
        for target in (4336800,2168400,867360,433680):
            actual = allocation(initial,target)
            self.assertEqual(sum(actual),target)
            self.assertTrue(all(k<=n for k,n in zip(actual,previous)))
            previous = actual
        torch.manual_seed(11)
        model = MaskedFC([5,9,4,3],.2)
        scores = [torch.rand_like(l.weight) for l in model.layers]
        before = [int(m.sum()) for m in model.masks]
        keep = allocation(before,25)
        choose_delete(model,scores,[n-k for n,k in zip(before,keep)])
        self.assertEqual([int(m.sum()) for m in model.masks],keep)
        self.assertLessEqual(sum(int(torch.count_nonzero(l.weight)) for l in model.layers),25)


if __name__ == "__main__":
    unittest.main()
