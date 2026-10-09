"""一次真实CPU图清理：间接断路、非零bias、精确零mask槽及原ID。"""
import unittest
import numpy as np
import torch
from tools.train_capacity_pruning import MaskedFC
from tools.clean_capacity_reachability import clean_graph,structural_info


class ReachabilityTest(unittest.TestCase):
    def test_backward_cleanup_preserves_function_and_source(self):
        torch.set_num_threads(2); torch.manual_seed(11)
        model=MaskedFC([3,4,4,3,2],.2)
        with torch.no_grad():
            for l in model.layers:
                l.weight.zero_(); l.bias.fill_(.4)
            model.layers[0].weight[0,0]=.3; model.layers[0].weight[2,1]=-.6
            model.layers[0].weight[1,2]=.5; model.layers[0].weight[3,0]=.8
            model.layers[1].weight[0,0]=.7; model.layers[1].weight[2,2]=.9
            model.layers[1].weight[1,1]=.5; model.layers[1].weight[3,3]=.9
            model.layers[2].weight[0,0]=.6; model.layers[2].weight[2,2]=.2
            model.layers[2].weight[1,1]=.3
            model.layers[3].weight[0,0]=.7; model.layers[3].weight[1,2]=.8
        frozen={k:v.clone() for k,v in model.state_dict().items()}
        x=torch.randn(24,3)
        logits=model.eval()(x).detach()
        cleaned,certificate=clean_graph(model,[[2,5,8,11],[0,3,6,9],[2,7,11]])
        self.assertTrue(torch.allclose(cleaned.eval()(x),logits,atol=1e-7,rtol=1e-7))
        self.assertEqual([r["direct_zero_outgoing_count"] for r in certificate],[0,1,1])
        self.assertEqual([r["indirect_unreachable_count"] for r in certificate],[2,1,0])
        self.assertEqual([r["discarded_nonzero_bias_count"] for r in certificate],[2,2,1])
        self.assertEqual(certificate[0]["surviving_original_ids"],[2,8])
        self.assertEqual(structural_info(cleaned)["widths"],[3,2,2,2,2])
        self.assertLess(sum(structural_info(cleaned)["nonzero_edges"]),sum(structural_info(model)["nonzero_edges"]))
        for k,v in frozen.items(): self.assertTrue(torch.equal(model.state_dict()[k],v))


if __name__=="__main__": unittest.main()
