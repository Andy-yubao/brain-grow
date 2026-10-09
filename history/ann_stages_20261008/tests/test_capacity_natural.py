"""一次小CPU检查真实补偿、物理缩宽与拒绝状态恢复。"""
import unittest

import torch

from tools.train_capacity_natural import NaturalFC, apply_nodes, snapshot, recover, structure, train_batches, propose


class NaturalMechanismTest(unittest.TestCase):
    def test_compensation_and_full_rejected_state(self):
        torch.set_num_threads(2)
        torch.manual_seed(11)
        model = NaturalFC([4,7,6,5,10], .2)
        with torch.no_grad():
            model.layers[0].weight[0].copy_(torch.tensor([1.,0.,0.,0.]))
            model.layers[0].weight[1].copy_(torch.tensor([0.,1.,0.,0.]))
            model.layers[0].weight[2].copy_(torch.tensor([1.,1.,0.,0.]))
            model.layers[0].bias[:3].zero_()
        x,y = torch.rand(12,4),torch.arange(12)%10
        event = {"layer":0,"deleted":[2],"basis":[0,1],"coefficients":[[1.],[1.]]}
        expected = model.eval()(x).detach()
        reduced,_ = apply_nodes(model,event)
        self.assertTrue(torch.allclose(reduced.eval()(x),expected,atol=1e-6,rtol=1e-6))
        self.assertEqual(reduced.layers[0].out_features,6)
        self.assertEqual(reduced.node_ids_0.tolist(),[0,1,3,4,5,6])
        self.assertLess(sum(structure(reduced)["nonzero_edges"]),sum(structure(model)["nonzero_edges"]))
        cfg={"learning_rate":.001,"weight_decay":.0001,"sensitivity_ema":.9,"evaluation_batch_size":8,
             "edge_group_sizes":[1,2,4],"model_seed":11,"probe_loss_increase":1e9,"probe_accuracy_drop":1.,
             "probe_kl":1e9,"proposal_seconds_per_slot":60}
        ema=[torch.ones_like(l.weight) for l in model.layers]
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        ledger={"attempted_updates":0,"optimization_samples":0,"proposal_function_samples":0}
        train_batches(model,optimizer,ema,[(x,y)],cfg,ledger)
        saved=snapshot(model,optimizer,ema)
        next_random=torch.rand(5)
        proposal=propose(model,ema,"edge",1,x,x,y,cfg,ledger)
        self.assertTrue(proposal["deleted"])
        for key,value in model.state_dict().items(): self.assertTrue(torch.equal(value,saved["model"][key]))
        candidate,new_ema=apply_nodes(model,event,ema)
        candidate_optimizer=torch.optim.AdamW(candidate.parameters(),lr=.001,weight_decay=.0001)
        train_batches(candidate,candidate_optimizer,new_ema,[(x,y)],cfg,ledger)
        restored,opt,scores=recover(candidate,candidate_optimizer,new_ema,saved,cfg)
        self.assertTrue(torch.equal(torch.rand(5),next_random))
        actual=snapshot(restored,opt,scores)
        for key,value in saved["model"].items(): self.assertTrue(torch.equal(value,actual["model"][key]))
        for old,new in zip(saved["gradients"],actual["gradients"]): self.assertTrue(torch.equal(old,new))
        for old,new in zip(saved["ema"],actual["ema"]): self.assertTrue(torch.equal(old,new))
        for i,row in saved["optimizer"]["state"].items():
            for key,value in row.items(): self.assertTrue(torch.equal(value,actual["optimizer"]["state"][i][key]))


if __name__=="__main__": unittest.main()
