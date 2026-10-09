"""B真实离散编辑、母图隔离与固定输入输出边界。"""
import pickle
import numpy as np
from braingrow.network.discrete import initial_network
from braingrow.network.communities import born_candidate,offspring,behavior_score,select_candidates


def test_full_candidate_edits_are_private_and_output_control_freezes_hidden():
    mother=initial_network();before=pickle.dumps(mother)
    c=born_candidate(mother,0);saved=pickle.dumps(c)
    kinds=set()
    for proposal_id in range(6,78):
        full=offspring(c,proposal_id,True);control=offspring(c,proposal_id,False)
        kinds.update(a["kind"] for a in full.actions)
        for name in ("win","rec","alpha","threshold","bias"):
            np.testing.assert_array_equal(getattr(control.net,name),getattr(c.net,name))
        assert not full.net.rec[128:,:128].any()
        assert len(full.stable_ids)==full.net.n and len(set(full.stable_ids))==full.net.n
    assert kinds=={"input","internal","dynamics","output","node"}
    assert pickle.dumps(c)==saved and pickle.dumps(mother)==before


def test_behavior_selection_keeps_complete_graphs_and_checkpoint():
    baseline=np.array([[1,1],[1,0],[0,1],[0,0]])
    y=np.array([[0,1],[0,0],[0,1],[1,0]]);ids=np.array([10,11,12,13])
    candidates=[]
    for proposal_id in range(12):
        c=born_candidate(initial_network(),proposal_id)
        c.prediction=baseline.copy()
        if proposal_id%2:c.prediction[0,0]=0
        behavior_score(c,baseline,y,ids,(0,0,1));candidates.append(c)
    rng=np.random.default_rng(11);selected=select_candidates(candidates,rng)
    assert len(selected)==6 and any(c.score["target_repairs"]==1 for c in selected)
    restored=pickle.loads(pickle.dumps({"candidates":selected,"rng":rng}))
    assert restored["rng"].integers(100)==rng.integers(100)
    for a,b in zip(selected,restored["candidates"]):
        np.testing.assert_array_equal(a.net.win,b.net.win)
