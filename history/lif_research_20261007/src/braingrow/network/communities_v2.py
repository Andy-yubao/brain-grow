"""B2：先非支配，再原始ID/视频多样性；两探索槽独立于任务槽。"""
def select_candidates(candidates, rng):
    nondominated = []
    for c in candidates:
        s = c.score
        dominated = any(o.score["target_repairs"] >= s["target_repairs"] and o.score["damage"] <= s["damage"] and
            (o.score["target_repairs"] > s["target_repairs"] or o.score["damage"] < s["damage"]) for o in candidates if o is not c)
        if not dominated: nondominated.append(c)
    selected = []; covered_ids = set(); covered_videos = set()
    remaining = list(candidates)
    for _ in range(min(4, len(candidates))):
        preferred = [c for c in nondominated if c in remaining]
        pool = preferred or remaining
        best = min(pool, key=lambda c: (-len(set(c.score["repair_ids"]) - covered_ids),
            -len(set(c.score["signature"]) - covered_videos), -c.score["target_repairs"],
            c.score["damage"], -c.score["total_repairs"], c.proposal_id))
        selected.append(best); covered_ids.update(best.score["repair_ids"]); covered_videos.update(best.score["signature"])
        remaining = [c for c in remaining if c is not best]
    for index in rng.choice(len(remaining), min(2, len(remaining)), False): selected.append(remaining[int(index)])
    return selected
