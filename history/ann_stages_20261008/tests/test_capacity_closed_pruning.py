"""一次小CPU真实事务：删边触发间接闭合，跨宽度拒绝恢复全部状态。"""
import unittest
import torch
from tools.train_capacity_natural import NaturalFC,train_batches,apply_edges
from tools.train_capacity_closed_pruning import close_candidate,snapshot,recover,propose_edges


class ClosedPruningTest(unittest.TestCase):
    def test_closed_candidate_and_rejection(self):
        torch.set_num_threads(2); torch.manual_seed(11)
        model=NaturalFC([4,6,5,4,10],.2)
        with torch.no_grad():
            model.layers[1].weight.zero_()
            for row,cols in enumerate(([1],[0,2],[3],[4],[5])):
                model.layers[1].weight[row,cols]=.2
            model.masks[1].copy_(model.layers[1].weight!=0)
        cfg={"learning_rate":.001,"weight_decay":.0001,"sensitivity_ema":.9,"evaluation_batch_size":8,
             "logit_atol":1e-4,"logit_rtol":1e-5,"model_seed":11,"edge_group_sizes":[1,2,4],
             "probe_loss_increase":.01,"probe_accuracy_drop":.005,"probe_kl":.002,"proposal_seconds_per_slot":60}
        x,y=torch.randn(12,4),torch.arange(12)%10
        ema=[torch.ones_like(l.weight) for l in model.layers]
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        ledger={"attempted_updates":0,"optimization_samples":0,"closure_evaluation_calls":0,"closure_evaluation_samples":0,"proposal_function_samples":0}
        train_batches(model,optimizer,ema,[(x,y)],cfg,ledger)
        model.eval()
        saved=snapshot(model,optimizer,ema)
        next_random=torch.rand(6)
        event=propose_edges(model,ema,1,x,y,cfg,ledger)
        self.assertTrue(all(model.layers[0].weight.flatten()[i]!=0 for i in event["deleted"]))
        for key,value in saved["model"].items(): self.assertTrue(torch.equal(model.state_dict()[key],value))
        apply_edges(model,{"layer":2,"deleted":[0,5,10,15]})
        closed,new_ema,closure=close_candidate(model,ema,x,cfg,ledger)
        self.assertTrue(closure["comparison"]["pass"])
        self.assertEqual([r["direct_zero_outgoing_count"] for r in closure["certificate"]],[0,1,0])
        self.assertEqual([r["indirect_unreachable_count"] for r in closure["certificate"]],[1,0,0])
        self.assertEqual(closed.node_ids_0.tolist(),[0,2,3,4,5])
        self.assertEqual(closed.node_ids_1.tolist(),[1,2,3,4])
        candidate_optimizer=torch.optim.AdamW(closed.parameters(),lr=.001,weight_decay=.0001)
        train_batches(closed,candidate_optimizer,new_ema,[(x,y)],cfg,ledger)
        restored,opt,scores=recover(closed,candidate_optimizer,new_ema,saved,cfg)
        self.assertFalse(restored.training)
        self.assertTrue(torch.equal(torch.rand(6),next_random))
        actual=snapshot(restored,opt,scores)
        for key,value in saved["model"].items(): self.assertTrue(torch.equal(value,actual["model"][key]))
        for old,new in zip(saved["gradients"],actual["gradients"]): self.assertTrue(torch.equal(old,new))
        for old,new in zip(saved["ema"],actual["ema"]): self.assertTrue(torch.equal(old,new))
        for i,row in saved["optimizer"]["state"].items():
            for key,value in row.items(): self.assertTrue(torch.equal(value,actual["optimizer"]["state"][i][key]))


if __name__=="__main__": unittest.main()
