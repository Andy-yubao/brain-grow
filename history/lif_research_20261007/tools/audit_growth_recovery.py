"""只读仿真图审计并补存B私有RNG；不执行LIF、不改训练结果。"""
import pickle
from pathlib import Path
import numpy as np
from braingrow.network.growing import GrowingNetwork
from braingrow.network.communities import born_candidate,offspring
from tools.run_discrete_growth import save,write_json


def main():
    directory=Path("experiments/runs/community_growth_d1")
    state=pickle.loads((directory/"resume.pkl").read_bytes())
    rng_states={};audited=0
    for arm in ("full","output_only"):
        candidates={}
        for row in (r for r in state["lineage"] if r["arm"]==arm):
            pid=row["proposal_id"]
            candidate=born_candidate(state["initial"],pid) if row["parent_id"] is None else offspring(candidates[row["parent_id"]],pid,full=arm=="full")
            recorded=GrowingNetwork.load(directory/f"{arm}_graph{pid:03d}.npz")
            for key in ("win","rec","out","alpha","bias","threshold","input_mask","rec_mask","output_mask"):
                np.testing.assert_array_equal(getattr(candidate.net,key),getattr(recorded,key))
            assert candidate.stable_ids==row["stable_ids"] and candidate.actions==row["actions"]
            candidates[pid]=candidate;rng_states[(arm,pid)]=candidate.proposal_rng_state;audited+=1
        for frozen in directory.glob(arm+"_frozen*.pkl"):
            candidate=pickle.loads(frozen.read_bytes())
            candidate.proposal_rng_state=rng_states[(arm,candidate.proposal_id)];save(frozen,candidate)
    for candidate in state["starting_candidates"]:candidate.proposal_rng_state=rng_states[("full",candidate.proposal_id)]
    arm="full" if state["arm_index"]==0 else "output_only"
    for candidate in state["candidates"]:candidate.proposal_rng_state=rng_states[(arm,candidate.proposal_id)]
    save(directory/"proposal_rng_states.pkl",rng_states);save(directory/"resume.pkl",state)
    result={"ok":True,"actual_graphs_exactly_reconstructed":audited,"LIF_simulation_videos":0,
        "note":"事后用本轮固定根11/提议ID重建并逐数组相等核对，补存私有RNG末态；未筛选模型或改变成绩"}
    write_json("reports/growth_recovery_audit_20261004.json",result);print(result)


if __name__=="__main__":main()
