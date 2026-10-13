"""Crop-shaping reward (v14 exp1) — STANDALONE, env-gated, pure/testable.

Lineage:
  v11 route ② (oracle-gate finding, see RL_EXPERIMENT_LOG.md / memory
  vdr-rl-oracle-crop-capability-gate): given a crop that CONTAINS the GT, the SFT
  model localizes fine (committed IoU ~0.55-0.59). The 0.22 eval ceiling is mode-A =
  the agent crops the WRONG region, and the crop SEARCH diverges on hard samples with
  no intermediate signal (memory vdr-rl-toolcall-doomloop-difficulty-proxy). So the
  reward must give a PROCESS signal on crop quality.

  Naive containment-only bonus is HACKABLE: crop the whole image → containment=1. The
  v14 hardened potential (validated offline on real v13 GRPO groups, crop_reward_gateA.py
  + crop_reward_groupval.py) is:

      phi(C) = containment(GT ⊂ C) × sat_zoom(C) × context_ok(GT, C)

  * containment    = area(GT ∩ C)/area(GT)  — is the target inside the crop (reward
                     CONTAINMENT not tightness; oracle M5≈M10 → looseness barely hurts).
  * sat_zoom       = min(1, ln(A_orig/A_crop)/ln(8))  — rewards zooming in, saturates at
                     8×; whole-image crop (zoom 1) → 0 → kills the whole-image hack.
  * context_ok     = clamp((1 - fill)/(1 - 0.5)), fill = area(GT)/area(C) — penalizes
                     over-tight crops that leave no context; kills the tiny-hug hack.

  Combination with the terminal outcome reward R_out (∈[0,1]; (DIoU+1)/2 grounding):
      R = R_out + λ·φ_best·(1 − R_out)  −  β·wander
      φ_best  = max_t φ(C_t)       (best crop the agent ever produced)
      φ_final = φ(C_last)          (φ of the LAST crop; 0 if no crops)
      wander  = max(0, φ_best − φ_final)   (found the target then drifted off it)
  Soft gate (1 − R_out): shaping fades as the outcome is already solved (no
  double-paying, avoids the 0.5-boundary discontinuity of a hard (1−hit) gate).
  Defaults λ=0.3, β=0.2 (both env-tunable). Zero-crop trajectory → φ_best=φ_final=0 →
  R = R_out (neutral; direct-answer easy samples untouched).

Coordinate space: crop `boxes` and reward_task["bbox_1000_xyxy"] are BOTH /1000 xyxy
relative to the ORIGINAL image, so A_orig = 1000*1000 and no image files are needed —
BUT only for ORIGINAL-frame crops. A nested crop (image_id=IMG_xxx) uses /1000 relative
to a SUB-image, so its area/coords are in a different frame and φ would be wrong. Nested
crops are 0.6% of real v13 crops (measured); the reward scores ORIGINAL-frame crops only
and logs the nested count (see parse_crop_bboxes_original).

This module has NO training-framework imports. Wiring into deepresearch_workflow.py is a
small edit (see `INTEGRATION` docstring at bottom), gated behind MM_REWARD_CROP_SHAPING.
"""
from __future__ import annotations

import json
import math
import os
import re
from typing import Any, List, Optional, Sequence, Tuple

_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
CROP_TOOL_NAMES = {"crop_and_search", "crop_image", "crop"}

Box = List[float]

# ---- hardened-phi constants ----------------------------------------------
A_ORIG = 1000.0 * 1000.0     # crop boxes are /1000 relative to the ORIGINAL frame
ZOOM_TARGET = 8.0            # sat_zoom saturates at 8× zoom (matches oracle-gate probe)
FILL_OK, FILL_MAX = 0.5, 1.0  # context_ok: full reward for fill<=0.5, →0 at fill=1.0

# ---- v15 constants: put the RANGE where the policy actually lives ---------
# v14 post-mortem (measured on 11k real rollouts): phi was at ceiling for 59% of
# trajectories, sat_zoom at ceiling 93.6%, context_ok 88.0% -> phi collapsed to
# ~containment and could not tell a 0.009-fill crop from a 0.056-fill crop (all three
# crops of a textbook 3-step zoom-in scored phi=1.000). Meanwhile, CONDITIONAL on
# containment, tightness swings hit@0.5 from 0.169 (fill<0.02) to 0.481 (fill>=0.08),
# and that survives controlling for target size (pooled within-band +0.343).
# So v15 scores TIGHTNESS on a log ramp across the regime the policy occupies
# (observed fill: p25 0.007 / median 0.032 / 30.6% >= 0.08).
FILL_FLOOR = 0.001    # below this the crop is ~the whole image -> no credit
FILL_TARGET = 0.08    # oracle-tight; ramp saturates here (no gain past it: 0.402 vs 0.423)
WHOLE_IMAGE_ZOOM = 4.0  # degenerate-crop guard: binds ONLY near whole-image, not a score axis

# Relative weights, set from the WITHIN-TASK causal test (the only control that
# actually holds difficulty fixed -- controlling for target-size BAND does not, and
# an earlier size-band analysis over-stated tightness at +0.343). Measured on mixed
# groups (those where hit varies, 30% of groups; all-fail groups contribute a
# structural zero and dilute any contrast):
#     containment split -> +0.195 hit@0.5 / +0.150 IoU
#     tightness  split -> +0.036 hit@0.5 / +0.024 IoU     (~5x weaker)
# So containment carries the reward and tightness is a tie-breaker, NOT a co-equal.
# FINAL WEIGHTS — chosen by ranking candidate designs on the ONLY criterion that
# matters for a process reward: does it order FAILED rollouts by how close they came?
# (62% of groups are all-fail. Those groups are NOT wasted -- under soft DIoU every one
# of them already has non-zero reward std, so they carry gradient; the job of phi is to
# make that gradient point somewhere sensible.) Measured on 446 all-fail groups,
# score = coverage x mean within-group Spearman(phi, continuous IoU):
#     A containment only              cover 69.1%  rho +0.262  score 0.181   <-- WINNER
#     B + approach floor              cover 80.5%  rho +0.155  score 0.124
#     C 0.85*frame + 0.15*tight       cover 99.1%  rho +0.105  score 0.104
#     E 0.85*frame + 0.15*tight, no approach       rho +0.180  score 0.158
# Both embellishments buy coverage and pay for it in ordering quality. A phi that is
# flat in 31% of groups is harmless there (R_out still supplies gradient); a phi that
# orders every group but points the wrong way is actively harmful.
#
# Tightness is 0 by default because it does not survive its own control:
#     Spearman(tightness, IoU | contained, all-fail) = -0.071   (NEGATIVE)
#     within-task tightness split on mixed groups   = +0.036    (vs containment +0.195)
# i.e. "contain the target" is supported; "and be as tight as possible" is not.
# Both knobs are kept env-tunable so the claim can be re-tested, but the defaults are
# what the data supports.
W_TIGHT = float(os.getenv("MM_REWARD_CROP_W_TIGHT", "0.0"))
APPROACH_W = float(os.getenv("MM_REWARD_CROP_APPROACH_W", "0.0"))
W_FRAME = 1.0 - W_TIGHT
IMG_DIAG = math.sqrt(2.0) * 1000.0


# ---- geometry (pure) ------------------------------------------------------

def _area(b: Sequence[float]) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _intersection_area(a: Sequence[float], b: Sequence[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def gt_containment(crop: Sequence[float], gt: Sequence[float]) -> float:
    """Fraction of the GT box's area that lies inside `crop`. 1.0 == GT fully contained.

    Reward-relevant quantity: saturates at 1.0 once the GT is fully in view (looseness
    beyond that not penalised), degrades smoothly as the crop clips the GT.
    """
    gt_area = _area(gt)
    if gt_area <= 0.0:
        return 0.0
    return _intersection_area(_normalize(crop), _normalize(gt)) / gt_area


def sat_zoom(crop: Sequence[float]) -> float:
    """Saturating zoom factor of a crop relative to the ORIGINAL image.

    whole-image crop (area==A_orig, zoom 1) → 0; ≥8× zoom → 1. Kills the whole-image hack.
    """
    ac = _area(_normalize(crop))
    if ac <= 0.0:
        return 0.0
    z = A_ORIG / ac
    if z <= 1.0:
        return 0.0
    return min(1.0, math.log(z) / math.log(ZOOM_TARGET))


def context_ok(crop: Sequence[float], gt: Sequence[float]) -> float:
    """Penalize over-tight crops where the GT FILLS the crop (no context left).

    fill = area(GT)/area(crop). Full reward for fill<=FILL_OK, linearly to 0 at
    fill=FILL_MAX. Kills the tiny-hug hack; a GT+margin crop keeps context_ok≈1.
    """
    ac = _area(_normalize(crop))
    if ac <= 0.0:
        return 0.0
    fill = _area(_normalize(gt)) / ac
    return max(0.0, min(1.0, (FILL_MAX - fill) / (FILL_MAX - FILL_OK)))


def phi(crop: Sequence[float], gt: Sequence[float]) -> float:
    """v14 crop-quality potential for a single GT box, in [0,1]. SATURATED -- see
    FILL_FLOOR comment; kept so v8..v14 runs stay reproducible."""
    return gt_containment(crop, gt) * sat_zoom(crop) * context_ok(crop, gt)


def gt_fill(crop: Sequence[float], gt: Sequence[float]) -> float:
    """area(GT)/area(crop): how much of the crop the target occupies. The single
    strongest controllable predictor of a correct final box (see FILL_FLOOR note)."""
    ac = _area(_normalize(crop))
    return (_area(_normalize(gt)) / ac) if ac > 0 else 0.0


def zoom_ramp(fill: float) -> float:
    """Log ramp over the fill regime the policy actually occupies: 0 at FILL_FLOOR
    (~whole image), 1 at FILL_TARGET (oracle-tight). This is the term that carries the
    v15 gradient -- it is the axis v14 was flat on."""
    if fill <= FILL_FLOOR:
        return 0.0
    return min(1.0, math.log(fill / FILL_FLOOR) / math.log(FILL_TARGET / FILL_FLOOR))


def hug_guard(crop: Sequence[float], gt: Sequence[float]) -> float:
    """Kill the tiny-hug hack: a crop that IS the GT leaves no context. Full credit for
    fill<=FILL_OK, ->0 at fill=FILL_MAX. Binds on ~6.5% of real crops."""
    fill = gt_fill(crop, gt)
    return max(0.0, min(1.0, (FILL_MAX - fill) / (FILL_MAX - FILL_OK)))


def whole_image_guard(crop: Sequence[float]) -> float:
    """Kill the whole-image hack. Saturates at a 4x area reduction, so it is 1.0 for
    essentially every real crop and only bites the degenerate 'crop the whole image'
    case. Deliberately NOT a scoring axis (that was v14's sat_zoom mistake)."""
    ac = _area(_normalize(crop))
    if ac <= 0.0:
        return 0.0
    z = A_ORIG / ac
    if z <= 1.0:
        return 0.0
    return min(1.0, math.log(z) / math.log(WHOLE_IMAGE_ZOOM))


def approach(crop: Sequence[float], gt: Sequence[float]) -> float:
    """Graded 'how close is this crop to framing the target', in [0,1], defined even
    when the crop misses entirely. Center distance normalised by the image diagonal."""
    c, g = _normalize(crop), _normalize(gt)
    cx, cy = (c[0] + c[2]) / 2.0, (c[1] + c[3]) / 2.0
    gx, gy = (g[0] + g[2]) / 2.0, (g[1] + g[3]) / 2.0
    d = math.hypot(cx - gx, cy - gy)
    return max(0.0, 1.0 - min(1.0, d / IMG_DIAG))


def frame_quality(crop: Sequence[float], gt: Sequence[float]) -> float:
    """Containment, optionally floored by graded approach credit (APPROACH_W, default 0).

    The approach floor was intended to give gradient where containment is a flat 0. It
    does raise coverage across all-fail groups from 69.1% to 80.5% -- but it degrades the
    ordering from rho +0.262 to +0.155, a net loss. Even restricted to the hardest groups
    (no rollout contained the target at all) approach ordered failures at rho +0.018,
    i.e. noise. Off by default; keep the knob so the claim stays falsifiable."""
    cont = gt_containment(crop, gt)
    if APPROACH_W <= 0.0:
        return cont
    return max(cont, APPROACH_W * approach(crop, gt))


def phi_v15(crop: Sequence[float], gt: Sequence[float]) -> float:
    """v15 crop-quality potential.

        phi = (W_FRAME*frame_quality + W_TIGHT*tightness_if_contained)
              * hug_guard * whole_image_guard

    With the data-backed defaults (W_TIGHT=0, APPROACH_W=0) this reduces to

        phi = containment * hug_guard * whole_image_guard

    which is deliberately CLOSE to v14. That is the finding, not an oversight: v14 was
    already measuring the right quantity. Its failures were elsewhere --
      (1) -beta*wander taxed retrying (wander>0 needs >=2 crops), and
      (2) lambda=0.3 gave the term only 23% of within-group variance.
    The one real change here is that sat_zoom/context_ok are demoted from SCORE AXES to
    pure guards: as score axes they sat at ceiling 93.6%/88.0%, so they never separated
    anything, but they did attenuate containment. whole_image_guard also saturates at 4x
    rather than 8x so it bites only the degenerate crop.

    Both W_TIGHT and APPROACH_W are env-tunable (MM_REWARD_CROP_W_TIGHT /
    MM_REWARD_CROP_APPROACH_W) for re-testing; see the weight block above for why each
    defaults to 0.
    """
    contained = gt_containment(crop, gt) >= 0.99
    tight = zoom_ramp(gt_fill(crop, gt)) if (contained and W_TIGHT > 0.0) else 0.0
    base = W_FRAME * frame_quality(crop, gt) + W_TIGHT * tight
    return base * hug_guard(crop, gt) * whole_image_guard(crop)


def active_phi_variant() -> str:
    """'v15' (tightness-aware) or 'v14' (legacy, saturated). Env: MM_REWARD_CROP_PHI."""
    v = os.getenv("MM_REWARD_CROP_PHI", "v14").strip().lower()
    return "v15" if v == "v15" else "v14"


def _phi_one(crop: Sequence[float], gt: Sequence[float]) -> float:
    return phi_v15(crop, gt) if active_phi_variant() == "v15" else phi(crop, gt)


def _phi_multi(crop: Sequence[float], gts: Sequence[Sequence[float]]) -> float:
    """Mean phi over GT boxes (single-GT grounding => just phi(crop, gt))."""
    if not gts:
        return 0.0
    return sum(_phi_one(crop, g) for g in gts) / len(gts)


def _normalize(b: Sequence[float]) -> Box:
    """Coerce to floats and fix inverted corners; return [x1,y1,x2,y2]."""
    x1, y1, x2, y2 = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


def _valid_box(b: Any) -> bool:
    return (isinstance(b, (list, tuple)) and len(b) == 4
            and all(isinstance(v, (int, float)) for v in b))


def _boxes_from_args(args: Any) -> List[Box]:
    """Extract crop box(es) from a tool_call's arguments dict.

    Real live schema uses `boxes`; older/test payloads use `bbox`. Accepts a single
    [x1,y1,x2,y2] or an array-of-boxes. Returns [] on anything malformed.
    """
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except (ValueError, TypeError):
            return []
    if not isinstance(args, dict):
        return []
    bx = args.get("boxes")
    if bx is None:
        bx = args.get("bbox")
    if bx is None:
        return []
    if _valid_box(bx):
        return [_normalize(bx)]
    if isinstance(bx, (list, tuple)):
        return [_normalize(b) for b in bx if _valid_box(b)]
    return []


# ---- extraction of crop bboxes from a trajectory --------------------------

def _iter_crop_calls(messages: Sequence[dict]):
    """Yield (call_dict, args) for every crop tool_call in assistant turns.

    Tolerant: skips malformed tool_calls / non-crop calls rather than raising.
    """
    for msg in messages or []:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if not isinstance(content, str) or "<tool_call>" not in content:
            continue
        for raw in _TOOL_CALL_RE.findall(content):
            try:
                call = json.loads(raw.strip())
            except (ValueError, TypeError):
                continue
            if not isinstance(call, dict) or call.get("name") not in CROP_TOOL_NAMES:
                continue
            yield call, call.get("arguments")


def parse_crop_bboxes(messages: Sequence[dict]) -> List[Box]:
    """Every crop bbox (single or array), ANY frame. Legacy helper for crop_containment_bonus."""
    boxes: List[Box] = []
    for _call, args in _iter_crop_calls(messages):
        boxes.extend(_boxes_from_args(args))
    return boxes


def parse_crop_bboxes_original(messages: Sequence[dict]) -> Tuple[List[Box], int]:
    """Crop bboxes on the ORIGINAL frame only, plus a count of skipped NESTED boxes.

    A crop call with image_id in {None, "ORIGINAL"} is original-frame (missing key
    defaults to ORIGINAL — the tool's default target image). image_id="IMG_xxx" is a
    nested crop in a DIFFERENT coordinate frame (0.6% of real crops); its coords/area
    are not comparable to the /1000 ORIGINAL frame, so we skip it (φ would be wrong)
    and count it for monitoring. Returns (original_boxes, n_nested_skipped).
    """
    boxes: List[Box] = []
    n_nested = 0
    for _call, args in _iter_crop_calls(messages):
        parsed = _boxes_from_args(args)
        if not parsed:
            continue
        image_id = args.get("image_id") if isinstance(args, dict) else None
        if image_id in (None, "ORIGINAL"):
            boxes.extend(parsed)
        else:
            n_nested += len(parsed)
    return boxes, n_nested


def _gt_boxes(gt: Any) -> List[Box]:
    """reward_task['bbox_1000_xyxy'] -> list of GT boxes (handles single or list-of-boxes)."""
    if _valid_box(gt):
        return [_normalize(gt)]
    if isinstance(gt, (list, tuple)):
        return [_normalize(b) for b in gt if _valid_box(b)]
    return []


# ---- the hardened crop-shaping terms + composition ------------------------

def crop_shaping_terms(
    result: dict,
    gt_bbox: Any,
    task: Optional[str] = None,
    grounding_only: bool = True,
) -> dict:
    """Compute the raw crop-shaping potential structure (NO λ/β applied) for a rollout.

    Returns a dict always containing:
      applicable   : bool  — grounding task with a valid GT (else shaping is a no-op)
      phi_best     : float — max_t phi(C_t) over ORIGINAL-frame crops (0 if none)
      phi_final    : float — phi of the LAST ORIGINAL-frame crop (0 if none)
      wander       : float — max(0, phi_best - phi_final)
      n_crop       : int   — # ORIGINAL-frame crops scored
      n_crop_nested: int   — # nested crops skipped (monitor: nested surge)
    """
    out = {"applicable": False, "phi_best": 0.0, "phi_final": 0.0,
           "wander": 0.0, "n_crop": 0, "n_crop_nested": 0,
           # --- v15 behavioural fields (diagnostics; do NOT enter the reward) ---
           "phi_first": 0.0, "phi_gain_from_retry": 0.0, "first_crop_bad": 0.0,
           "retried": 0.0, "contained": 0.0, "tight": 0.0, "fill_best": 0.0,
           "phi_at_ceiling": 0.0}
    if grounding_only and task is not None and "ground" not in str(task).lower():
        return out
    gts = _gt_boxes(gt_bbox)
    if not gts:
        return out
    out["applicable"] = True
    messages = result.get("messages", []) if isinstance(result, dict) else []
    crops, n_nested = parse_crop_bboxes_original(messages)
    out["n_crop"] = len(crops)
    out["n_crop_nested"] = n_nested
    if not crops:
        return out  # zero-crop -> phi_best=phi_final=0 -> neutral (R = R_out)
    phis = [_phi_multi(c, gts) for c in crops]
    out["phi_best"] = max(phis)
    out["phi_final"] = phis[-1]
    out["wander"] = max(0.0, out["phi_best"] - out["phi_final"])

    # --- behavioural instrumentation -------------------------------------
    # PRIMARY metric: does the policy retry after a bad first crop? Measured at
    # 51.3% pre-v15; v14's -beta*wander taxed exactly this (wander>0 requires >=2
    # crops, so committing after one crop guaranteed a zero penalty).
    out["phi_first"] = phis[0]
    out["phi_gain_from_retry"] = max(0.0, out["phi_best"] - phis[0])
    # "bad first crop" is judged on CONTAINMENT (stage-1 failure, 63.3% of rollouts),
    # independent of the phi variant so the metric stays comparable across runs.
    cont = [sum(gt_containment(c, g) for g in gts) / len(gts) for c in crops]
    fills = [sum(gt_fill(c, g) for g in gts) / len(gts) for c in crops]
    out["first_crop_bad"] = 1.0 if cont[0] < 0.99 else 0.0
    out["retried"] = 1.0 if len(crops) >= 2 else 0.0
    out["contained"] = 1.0 if max(cont) >= 0.99 else 0.0

    # --- GATED wander: penalise ABANDONING a found target, not retrying -------
    # "Found it then left" is genuinely bad and survives the within-task control:
    #   kept vs abandoned = +0.394 vs +0.086 hit@0.5 pooled; +0.075 hit / +0.086 IoU
    #   WITHIN group (n=201). ~26% of found-trajectories abandon.
    # But raw wander is structurally biased: a single-crop trajectory CANNOT incur it
    # (phi_best == phi_final by construction) while 47.9% of multi-crop trajectories did.
    # Retrying therefore carried risk that committing did not -- which is what collapsed
    # crop_n 3.28 -> 1.11 in v14prod, not the penalty's intent.
    # Gating on "the target was actually found" removes exactly that bias: the 36.9% of
    # multi-crop trajectories that never contained the target become EXEMPT, so retrying
    # from a bad start is free, while abandoning a good crop still costs.
    out["found_then_left"] = 1.0 if (out["contained"] >= 1.0 and cont[-1] < 0.99) else 0.0
    out["wander_gated"] = out["wander"] if out["contained"] >= 1.0 else 0.0
    out["tight"] = 1.0 if any(cn >= 0.99 and fl >= FILL_TARGET
                              for cn, fl in zip(cont, fills)) else 0.0
    # fill of the best-phi crop (raw + interpretable; compare to FILL_TARGET)
    out["fill_best"] = fills[max(range(len(phis)), key=lambda i: phis[i])]
    # HEADROOM MONITOR: if this climbs toward 1.0 the reward has saturated again and
    # v15 has degenerated into v14. Watch it every step.
    out["phi_at_ceiling"] = 1.0 if out["phi_best"] >= 0.99 else 0.0
    return out


def compose_crop_shaped_reward(
    r_out: float,
    terms: dict,
    lam: float = 0.3,
    beta: float = 0.2,
) -> Tuple[float, dict]:
    """R = R_out + λ·φ_best·(1−R_out) − β·wander, clamped to [0,1].

    Returns (shaped_reward, info) where info logs φ_best / φ_final / wander / bonus /
    penalty / crop counts SEPARATELY (so we can tell which term moved when tuning β).
    Non-applicable (non-grounding / no GT) -> returns (r_out, {}) unchanged.
    """
    if not terms.get("applicable"):
        return r_out, {}
    phi_best = float(terms.get("phi_best", 0.0))
    # MM_REWARD_CROP_WANDER_GATE=1 charges the wander penalty ONLY when the target was
    # actually framed and then abandoned, which removes the commit-vs-retry asymmetry
    # that collapsed crop_n in v14prod. Default 0 = legacy ungated behaviour.
    gated = os.getenv("MM_REWARD_CROP_WANDER_GATE", "0") == "1"
    wander = float(terms.get("wander_gated" if gated else "wander", 0.0))
    bonus = lam * phi_best * (1.0 - r_out)
    penalty = beta * wander
    shaped = max(0.0, min(1.0, r_out + bonus - penalty))
    info = {
        "crop_phi_best": phi_best,
        "crop_phi_final": float(terms.get("phi_final", 0.0)),
        "crop_wander": wander,
        "crop_bonus": bonus,
        "crop_penalty": penalty,
        "crop_n": int(terms.get("n_crop", 0)),
        "crop_n_nested": int(terms.get("n_crop_nested", 0)),
        "reward_out": float(r_out),
        "reward_shaped": shaped,
        # --- v15 behavioural metrics (diagnostic only, not in the reward) ---
        "crop_phi_first": float(terms.get("phi_first", 0.0)),
        "crop_phi_gain_from_retry": float(terms.get("phi_gain_from_retry", 0.0)),
        "crop_first_bad": float(terms.get("first_crop_bad", 0.0)),
        "crop_retried": float(terms.get("retried", 0.0)),
        "crop_contained": float(terms.get("contained", 0.0)),
        "crop_tight": float(terms.get("tight", 0.0)),
        "crop_fill_best": float(terms.get("fill_best", 0.0)),
        "crop_phi_at_ceiling": float(terms.get("phi_at_ceiling", 0.0)),
        "crop_found_then_left": float(terms.get("found_then_left", 0.0)),
        "crop_wander_gated": float(terms.get("wander_gated", 0.0)),
    }
    return shaped, info


# ---- legacy v11 containment-only bonus (kept for compat; HACKABLE) ---------

def crop_containment_bonus(
    result: dict,
    gt_bbox: Any,
    lam: float = 0.2,
    task: Optional[str] = None,
    grounding_only: bool = True,
) -> float:
    """λ * best-containment bonus (containment ONLY — no zoom/context; whole-image-hackable).

    Superseded by crop_shaping_terms + compose_crop_shaped_reward. Retained so existing
    tests / callers keep working. Scores crops in ANY frame (legacy behavior).
    """
    if grounding_only and task is not None and "ground" not in str(task).lower():
        return 0.0
    gts = _gt_boxes(gt_bbox)
    if not gts:
        return 0.0
    crops = parse_crop_bboxes(result.get("messages", []) if isinstance(result, dict) else [])
    if not crops:
        return 0.0
    best = 0.0
    for crop in crops:
        best = max(best, _phi_multi_containment(crop, gts))
    return lam * best


def _phi_multi_containment(crop: Sequence[float], gts: Sequence[Sequence[float]]) -> float:
    return sum(gt_containment(crop, g) for g in gts) / len(gts) if gts else 0.0


# ---- INTEGRATION (apply only on user go) ----------------------------------
# In deepresearch_workflow.py, in the `else:` branch after outcome_reward is computed
# (~line 358-360), BEFORE the `for trajectory in episode.trajectories:` loop:
#
#     from .crop_containment_reward import crop_shaping_terms, compose_crop_shaped_reward
#     if os.getenv("MM_REWARD_CROP_SHAPING", "0") == "1":
#         terms = crop_shaping_terms(
#             result, reward_task.get("bbox_1000_xyxy"), task=reward_task.get("task"),
#         )
#         outcome_reward, crop_info = compose_crop_shaped_reward(
#             outcome_reward, terms,
#             lam=float(os.getenv("MM_REWARD_CROP_LAMBDA", "0.3")),
#             beta=float(os.getenv("MM_REWARD_CROP_BETA", "0.2")),
#         )
#         if crop_info:
#             episode.info["crop_shaping"] = crop_info
#             # per-term metrics land on the swanlab curve (φ_best/wander logged separately)
#             episode.metrics.update({f"crop/{k}": v for k, v in crop_info.items()})
#
# reward_task is available in this scope (line 319). Env-gated -> no-op until enabled;
# λ/β tunable without code change.
