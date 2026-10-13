"""Unit tests for crop_containment_reward (v11 route ②). Stdlib only; run directly."""
import importlib.util
import os

_MOD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crop_containment_reward.py")
_spec = importlib.util.spec_from_file_location("ccr", _MOD)
ccr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ccr)


def _msg(*tool_call_jsons):
    """Build an assistant message whose content holds the given <tool_call> blocks."""
    body = "<think>x</think>" + "".join(f"<tool_call>{j}</tool_call>" for j in tool_call_jsons)
    return {"role": "assistant", "content": body}


def _crop(bbox):
    import json
    return json.dumps({"name": "crop_and_search",
                       "arguments": {"image_id": "IMG", "bbox": bbox, "goal": "g"}})


GT = [400, 400, 500, 500]           # 100x100 box, area 10000
APPROX = 1e-6
_n = _p = 0


def check(name, cond):
    global _n, _p
    _n += 1
    ok = bool(cond)
    _p += ok
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")


# --- geometry ---
check("full containment = 1.0", abs(ccr.gt_containment([300, 300, 600, 600], GT) - 1.0) < APPROX)
check("no overlap = 0.0", ccr.gt_containment([0, 0, 100, 100], GT) == 0.0)
# crop covers left half of GT (x 400-450) -> half the GT area inside
check("half containment ~0.5", abs(ccr.gt_containment([0, 0, 450, 1000], GT) - 0.5) < APPROX)
check("looseness doesn't reduce (huge crop still 1.0)",
      abs(ccr.gt_containment([0, 0, 1000, 1000], GT) - 1.0) < APPROX)
check("degenerate GT -> 0.0", ccr.gt_containment([0, 0, 100, 100], [500, 500, 500, 500]) == 0.0)
check("inverted crop corners normalized", abs(ccr.gt_containment([600, 600, 300, 300], GT) - 1.0) < APPROX)

# --- extraction ---
msgs = [_msg(_crop([300, 300, 600, 600]))]
check("parse single crop", ccr.parse_crop_bboxes(msgs) == [[300, 300, 600, 600]])
msgs_multi = [_msg(_crop([0, 0, 100, 100])), _msg(_crop([300, 300, 600, 600]))]
check("parse across turns", len(ccr.parse_crop_bboxes(msgs_multi)) == 2)
# bbox as array-of-boxes in one call
arr = [{"role": "assistant", "content": f"<tool_call>{ccr.json.dumps({'name':'crop_and_search','arguments':{'image_id':'I','bbox':[[0,0,50,50],[300,300,600,600]]}})}</tool_call>"}]
check("parse array-of-bboxes", len(ccr.parse_crop_bboxes(arr)) == 2)
check("malformed json skipped", ccr.parse_crop_bboxes([_msg("{not json")]) == [])
check("non-crop tool skipped",
      ccr.parse_crop_bboxes([_msg('{"name":"search","arguments":{"bbox":[1,2,3,4]}}')]) == [])
check("bad bbox length skipped",
      ccr.parse_crop_bboxes([_msg(_crop([1, 2, 3]))]) == [])
check("user msgs ignored",
      ccr.parse_crop_bboxes([{"role": "user", "content": _crop([0, 0, 1, 1])}]) == [])

# --- bonus ---
res_good = {"messages": [_msg(_crop([300, 300, 600, 600]))]}
res_bad = {"messages": [_msg(_crop([0, 0, 100, 100]))]}
res_none = {"messages": [{"role": "assistant", "content": "<think>no tools</think><answer>[1,2,3,4]</answer>"}]}
check("bonus full containment = lam", abs(ccr.crop_containment_bonus(res_good, GT, lam=0.2, task="grounding_bbox") - 0.2) < APPROX)
check("bonus no containment = 0", ccr.crop_containment_bonus(res_bad, GT, lam=0.2, task="grounding_bbox") == 0.0)
check("bonus no crops = 0", ccr.crop_containment_bonus(res_none, GT, lam=0.2, task="grounding_bbox") == 0.0)
check("best-of-crops picks the good one",
      abs(ccr.crop_containment_bonus({"messages": res_bad["messages"] + res_good["messages"]}, GT, lam=0.2, task="grounding_bbox") - 0.2) < APPROX)
check("non-grounding task gated to 0",
      ccr.crop_containment_bonus(res_good, GT, lam=0.2, task="spot_diff") == 0.0)
check("missing GT -> 0", ccr.crop_containment_bonus(res_good, None, lam=0.2, task="grounding_bbox") == 0.0)
# list-of-GT boxes: crop covers only one of two GTs -> mean containment 0.5 -> bonus 0.1
two_gts = [[400, 400, 500, 500], [0, 0, 100, 100]]
check("list-of-GT mean containment",
      abs(ccr.crop_containment_bonus(res_good, two_gts, lam=0.2, task="grounding_bbox") - 0.1) < APPROX)
check("lam scales bonus", abs(ccr.crop_containment_bonus(res_good, GT, lam=0.5, task="grounding_bbox") - 0.5) < APPROX)

# ==========================================================================
# v14 hardened phi = containment * sat_zoom * context_ok
# ==========================================================================
import json as _json

def _crop_orig(boxes, image_id="ORIGINAL"):
    return _json.dumps({"name": "crop", "arguments": {"image_id": image_id, "boxes": boxes}})

def _msg_orig(*crop_jsons):
    return {"role": "assistant", "content": "<think>x</think>" + "".join(f"<tool_call>{j}</tool_call>" for j in crop_jsons)}

print("\n-- v14 sat_zoom --")
check("whole-image sat_zoom = 0", ccr.sat_zoom([0, 0, 1000, 1000]) == 0.0)
check("2x-area-shrink < 1", ccr.sat_zoom([0, 0, 700, 1000]) < 1.0)   # ~1.43x zoom
check("~8x zoom saturates ~1", abs(ccr.sat_zoom([0, 0, 354, 354]) - 1.0) < 5e-3)  # (1000/354)^2 ≈ 7.98
check("beyond-8x still 1 (saturated)", ccr.sat_zoom([0, 0, 100, 100]) == 1.0)   # 100x zoom
check("degenerate crop sat_zoom = 0", ccr.sat_zoom([500, 500, 500, 500]) == 0.0)

print("-- v14 context_ok --")
G2 = [400, 400, 500, 500]          # 100x100, area 10000
check("GT fills crop (fill=1) -> 0", ccr.context_ok(G2, G2) == 0.0)
check("GT is 50% of crop -> 1.0", abs(ccr.context_ok([400, 400, 500, 600], G2) - 1.0) < APPROX)  # crop area 20000, fill 0.5
check("GT is 25% of crop -> 1.0 (fill<0.5 saturates)", abs(ccr.context_ok([400, 400, 600, 600], G2) - 1.0) < APPROX)  # fill 0.25
check("GT is 75% of crop -> 0.5", abs(ccr.context_ok([400, 400, 500, 700] if False else [400, 400, 533, 500], G2) - (1 - (10000/((533-400)*100)))/0.5) < 1e-2)

print("-- v14 phi (composed) --")
# GOOD crop: GT + ~40% margin, contains GT, ~zoomed, has context -> phi high
good = [340, 340, 560, 560]        # area 220^2=48400, zoom ~20x, fill 10000/48400≈0.21 -> ctx 1, contain 1
check("good crop phi high (>0.9)", ccr.phi(good, G2) > 0.9)
check("whole-image phi = 0 (zoom kills it)", ccr.phi([0, 0, 1000, 1000], G2) == 0.0)
check("crop missing GT phi = 0 (containment kills it)", ccr.phi([0, 0, 100, 100], G2) == 0.0)
check("tiny-hug phi low (context kills it)", ccr.phi([400, 400, 500, 500], G2) == 0.0)  # fill=1 -> ctx 0

print("-- v14 ORIGINAL-frame filtering --")
boxes, n_nested = ccr.parse_crop_bboxes_original([_msg_orig(_crop_orig([1, 2, 3, 4]))])
check("ORIGINAL crop kept", boxes == [[1, 2, 3, 4]] and n_nested == 0)
boxes, n_nested = ccr.parse_crop_bboxes_original([_msg_orig(_crop_orig([1, 2, 3, 4], image_id="IMG_007"))])
check("nested crop skipped + counted", boxes == [] and n_nested == 1)
# missing image_id defaults to ORIGINAL
miss = [{"role": "assistant", "content": f"<tool_call>{_json.dumps({'name':'crop','arguments':{'boxes':[5,6,7,8]}})}</tool_call>"}]
boxes, n_nested = ccr.parse_crop_bboxes_original(miss)
check("missing image_id -> treated ORIGINAL", boxes == [[5, 6, 7, 8]] and n_nested == 0)

print("-- v14 crop_shaping_terms --")
res_good = {"messages": [_msg_orig(_crop_orig(good))]}
t = ccr.crop_shaping_terms(res_good, G2, task="grounding/localization")
check("terms applicable for grounding+GT", t["applicable"] is True)
check("phi_best high for good crop", t["phi_best"] > 0.9)
check("phi_final == phi_best (single crop, no wander)", abs(t["phi_final"] - t["phi_best"]) < APPROX and t["wander"] == 0.0)
check("n_crop counted", t["n_crop"] == 1 and t["n_crop_nested"] == 0)
# find-then-wander: good crop THEN whole-image drift
res_wander = {"messages": [_msg_orig(_crop_orig(good)), _msg_orig(_crop_orig([0, 0, 1000, 1000]))]}
tw = ccr.crop_shaping_terms(res_wander, G2, task="grounding/localization")
check("wander detected (best high, final 0)", tw["phi_best"] > 0.9 and tw["phi_final"] == 0.0 and tw["wander"] > 0.9)
# non-grounding gated
check("non-grounding -> not applicable", ccr.crop_shaping_terms(res_good, G2, task="spot_diff")["applicable"] is False)
# missing GT
check("missing GT -> not applicable", ccr.crop_shaping_terms(res_good, None, task="grounding")["applicable"] is False)
# zero-crop trajectory
t0 = ccr.crop_shaping_terms({"messages": [{"role": "assistant", "content": "<answer>[1,2,3,4]</answer>"}]}, G2, task="grounding")
check("zero-crop -> applicable but phi_best=0", t0["applicable"] is True and t0["phi_best"] == 0.0)

print("-- v14 compose_crop_shaped_reward --")
# R_out=0 (failed), good crop found: R = 0 + 0.3*phi_best*1 - 0 = 0.3*phi_best
r, info = ccr.compose_crop_shaped_reward(0.0, t, lam=0.3, beta=0.2)
check("failed+good-crop gets positive shaped reward", r > 0.27 and info["crop_bonus"] > 0.27 and info["crop_penalty"] == 0.0)
# soft gate: already-solved (R_out=1) -> bonus fades to 0
r1, info1 = ccr.compose_crop_shaped_reward(1.0, t, lam=0.3, beta=0.2)
check("solved (R_out=1) -> shaping bonus ~0 (soft gate)", abs(r1 - 1.0) < APPROX and info1["crop_bonus"] < APPROX)
# wander penalized but net still >= R_out for small R_out
rw, infow = ccr.compose_crop_shaped_reward(0.0, tw, lam=0.3, beta=0.2)
check("found-then-wander: penalty>0 but net still > 0 (found beats never-found)", infow["crop_penalty"] > 0 and rw > 0.0)
# zero-crop -> neutral (R = R_out)
rz, infoz = ccr.compose_crop_shaped_reward(0.4, t0, lam=0.3, beta=0.2)
check("zero-crop -> shaped == R_out (neutral)", abs(rz - 0.4) < APPROX)
# non-applicable -> passthrough, empty info
rn, infon = ccr.compose_crop_shaped_reward(0.4, {"applicable": False}, lam=0.3, beta=0.2)
check("non-applicable -> passthrough R_out, empty info", rn == 0.4 and infon == {})
# clamp to [0,1]
rc, _ = ccr.compose_crop_shaped_reward(0.95, t, lam=0.3, beta=0.2)
check("shaped clamped <= 1.0", rc <= 1.0)

print("-- v14 hack-resistance (the whole point) --")
res_whole = {"messages": [_msg_orig(_crop_orig([0, 0, 1000, 1000]))]}
check("whole-image crop -> phi_best 0 -> no bonus",
      ccr.compose_crop_shaped_reward(0.0, ccr.crop_shaping_terms(res_whole, G2, task="grounding"), lam=0.3)[0] == 0.0)
res_tiny = {"messages": [_msg_orig(_crop_orig(G2))]}  # crop == GT (tiny-hug)
check("tiny-hug crop -> phi_best 0 -> no bonus",
      ccr.compose_crop_shaped_reward(0.0, ccr.crop_shaping_terms(res_tiny, G2, task="grounding"), lam=0.3)[0] == 0.0)

print(f"\n{_p}/{_n} passed")
raise SystemExit(0 if _p == _n else 1)
