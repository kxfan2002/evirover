"""Aggregation of per-sample results into domain-standard metrics.

Hierarchy:
  subcategory : recognition, localization        (only under grounding)
  category    : grounding, spot_diff, segmentation, counting
  overall     : coverage/parse info only

There is no single "main score". Each task reports the metrics standard to its
domain, side by side, because they are not commensurable across tasks:
  grounding    : IoU + R@0.5          (REC / RefCOCO convention)
  spot_diff    : P/R + F1 (macro) + F1_micro   (multi-box detection)
  segmentation : gIoU + cIoU          (RES convention)
  counting     : accuracy + MAE
We deliberately do not fold these into one number or a cross-task average.

spot_diff records physically live in grounding/localization.jsonl (they share
the grounding family for loading/paths), but their metric is a different
domain (multi-target detection F1, not single-box IoU), so build_summary lifts
them out to a top-level category and keeps grounding/localization as pure
grounding_bbox.
"""
from collections import defaultdict, OrderedDict, Counter
from typing import Dict, List

from . import config


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _metrics_for_task(task: str, rows: List[dict]) -> dict:
    """Task-appropriate detail metrics for a bucket of rows (all same task)."""
    out = {}
    if task in ("grounding_bbox",):
        # REC convention: IoU (mean box IoU) + R@0.5 (fraction with IoU>=0.5).
        out["IoU"] = round(_mean([r["components"].get("iou", 0.0) for r in rows]), 4)
        out["R@0.5"] = round(_mean([r["components"].get("hit@0.5", 0.0) for r in rows]), 4)
    elif task == "spot_diff":
        # macro: per-image precision/recall/F1 averaged (each image weighted
        # equally, regardless of how many differences it contains).
        out["precision"] = round(_mean([r["components"].get("precision", 0.0) for r in rows]), 4)
        out["recall"] = round(_mean([r["components"].get("recall", 0.0) for r in rows]), 4)
        out["f1"] = round(_mean([r["components"].get("f1", 0.0) for r in rows]), 4)
        # micro: pool TP / predicted / GT across all images, then compute once
        # (images with more differences weigh more). Mirrors seg's gIoU vs cIoU.
        tp = sum(r["components"].get("tp", 0) for r in rows)
        n_pred = sum(r["components"].get("n_pred", 0) for r in rows)
        n_gt = sum(r["components"].get("n_gt", 0) for r in rows)
        p_mi = tp / n_pred if n_pred else 0.0
        r_mi = tp / n_gt if n_gt else 0.0
        out["f1_micro"] = round(2 * p_mi * r_mi / (p_mi + r_mi), 4) if (p_mi + r_mi) else 0.0
    elif task == "segmentation":
        # RES convention: gIoU (mean per-image IoU) + cIoU (Σinter / Σunion).
        out["gIoU"] = round(_mean([r["components"].get("iou", 0.0) for r in rows]), 4)
        tot_inter = sum(r["components"].get("inter", 0) for r in rows)
        tot_union = sum(r["components"].get("union", 0) for r in rows)
        out["cIoU"] = round(tot_inter / tot_union, 4) if tot_union > 0 else 0.0
        out["iou@0.5"] = round(_mean([r["components"].get("hit@0.5", 0.0) for r in rows]), 4)
        skipped = sum(1 for r in rows if r["components"].get("sam_skipped"))
        if skipped:
            out["sam_skipped"] = skipped
    elif task == "counting":
        out["accuracy"] = round(_mean([r["components"].get("correct", 0.0) for r in rows]), 4)
        errs = [r["components"]["abs_err"] for r in rows if "abs_err" in r["components"]]
        out["mae"] = round(_mean(errs), 4)
    return out


def _base_stats(rows: List[dict]) -> dict:
    n = len(rows)
    parse_ok = sum(1 for r in rows if r["parse_ok"])
    finish = defaultdict(int)
    for r in rows:
        finish[r.get("finish_reason") or "unknown"] += 1
    return {
        "n": n,
        "parse_ok": parse_ok,
        "parse_fail_rate": round(1 - parse_ok / n, 4) if n else 0.0,
        "mean_score": round(_mean([r["score"] for r in rows]), 4),
        "finish_reasons": dict(finish),
    }


def _bucket(rows: List[dict]) -> dict:
    """Base stats plus per-task detail metrics (buckets may mix tasks)."""
    out = _base_stats(rows)
    by_task = defaultdict(list)
    for r in rows:
        by_task[r.get("task")].append(r)
    # If the bucket is a single task, inline its metrics; else nest per task.
    if len(by_task) == 1:
        out.update(_metrics_for_task(next(iter(by_task)), rows))
    else:
        out["by_task"] = {t: {**_base_stats(rs), **_metrics_for_task(t, rs)}
                          for t, rs in sorted(by_task.items())}
    return out


def _collapse_repeats(rows: List[dict]) -> List[dict]:
    """With --repeat R there are R rows per question; average them before scoring.

    The R rows must not be fed to the statistics as R samples -- that is
    pseudo-replication: n grows R-fold and SE appears to shrink by sqrt(R) while
    between-question variance is unchanged, so the interval is fiction. Repeats
    only cancel within-question sampling noise. Collapse them to one row
    (averaging score and the numeric fields of components) and leave n alone.

    `parse_ok` takes the majority and `finish_reason` the mode; both are display
    only. Non-numeric components take the first value.

    The grouping key must be (file, id), not id: three ids appear in both
    grounding/recognition and segmentation (one image, two different questions).
    Keying on id alone averages a box IoU together with a mask IoU, corrupting
    both categories, and quietly drops n from 688 to 685.
    """
    if not any(int(r.get("rep", 0) or 0) for r in rows):
        return rows                      # --repeat not used; nothing to do
    by_id: "OrderedDict[tuple, list]" = OrderedDict()
    for r in rows:
        by_id.setdefault((r.get("file"), r["id"]), []).append(r)
    out = []
    for rid, grp in by_id.items():
        if len(grp) == 1:
            out.append(grp[0]); continue
        base = dict(grp[0])
        base["score"] = _mean([float(r.get("score", 0.0) or 0.0) for r in grp])
        base["parse_ok"] = sum(1 for r in grp if r.get("parse_ok")) * 2 >= len(grp)
        fr = Counter(r.get("finish_reason") or "unknown" for r in grp)
        base["finish_reason"] = fr.most_common(1)[0][0]
        keys = {k for r in grp for k in (r.get("components") or {})}
        comp = {}
        for k in keys:
            vals = [(r.get("components") or {}).get(k) for r in grp]
            nums = [float(v) for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
            comp[k] = (sum(nums) / len(nums)) if len(nums) == len(grp) else \
                next((v for v in vals if v is not None), None)
        base["components"] = comp
        base["n_reps"] = len(grp)
        base.pop("trajectory", None)      # no single trajectory owns a collapsed row
        out.append(base)
    return out


def build_summary(rows: List[dict]) -> dict:
    """Return the three-tier summary: subcategory / category / overall."""
    rows = _collapse_repeats(rows)
    # spot_diff is loaded under the grounding family but scored as its own
    # domain (multi-box F1), so lift it out to a top-level category. Grounding
    # subcategories are then computed from the remaining (grounding_bbox) rows.
    spot_rows = [r for r in rows if r.get("task") == "spot_diff"]
    ground_rows = [r for r in rows if r.get("task") != "spot_diff"]

    by_subcat = defaultdict(list)   # (family, subcat) -> rows
    by_category = defaultdict(list)
    for r in ground_rows:
        by_category[r["family"]].append(r)
        if r["family"] == "grounding":
            by_subcat[r.get("subcategory", "")].append(r)

    # Subcategory scores (grounding only).
    subcategories = {
        sub: _bucket(by_subcat[sub])
        for sub in config.GROUNDING_SUBCATEGORIES if by_subcat.get(sub)
    }

    # Category detail. No synthetic "main score": each category/subcategory
    # exposes its domain-standard metrics (grounding: IoU + R@0.5; spot_diff:
    # P/R/F1 + F1_micro; seg: gIoU + cIoU; counting: accuracy + MAE) and they
    # are reported side by side.
    categories = {}
    for cat in config.CATEGORIES:
        if cat == "spot_diff":
            if spot_rows:
                categories["spot_diff"] = _bucket(spot_rows)
            continue
        crows = by_category.get(cat)
        if not crows:
            continue
        if cat == "grounding":
            categories[cat] = {
                **_base_stats(crows),
                "subcategories": subcategories,
            }
        else:
            categories[cat] = _bucket(crows)

    # Overall carries only coverage/parse info — no cross-category aggregate
    # score (metrics are not commensurable across tasks, so we don't fold them).
    overall = {
        "n": len(rows),
        "parse_fail_rate": round(
            1 - sum(1 for r in rows if r["parse_ok"]) / len(rows), 4
        ) if rows else 0.0,
    }
    return {"categories": categories, "overall": overall}


# Descriptive/coverage keys that are not domain metrics — hidden from metric lists.
_HIDE = ("n", "parse_ok", "mean_score", "score", "parse_fail_rate",
         "finish_reasons", "by_task", "subcategories", "macro_of")


def _metrics_str(d: dict) -> str:
    return " ".join(f"{k}={v}" for k, v in d.items() if k not in _HIDE)


def format_summary(summary: dict) -> str:
    lines = []
    lines.append("=" * 72)
    lines.append("CATEGORIES (domain-standard metrics, reported side by side)")
    lines.append("=" * 72)
    for cat, m in summary["categories"].items():
        lines.append(f"  {cat:14s} n={m['n']:<3d} parse_fail={m['parse_fail_rate']}")
        if cat == "grounding":
            for sub, sm in m["subcategories"].items():
                if "by_task" in sm:
                    for t, tm in sm["by_task"].items():
                        lines.append(f"      └ {sub}/{t:14s} n={tm['n']:<3d} "
                                     f"{_metrics_str(tm)}")
                else:
                    lines.append(f"      └ {sub:12s} n={sm['n']:<3d} {_metrics_str(sm)}")
        else:
            extras = _metrics_str(m)
            if extras:
                lines.append(f"      {extras}")
            if "by_task" in m:
                for t, tm in m["by_task"].items():
                    lines.append(f"      └ {t:14s} n={tm['n']:<3d} {_metrics_str(tm)}")
    o = summary["overall"]
    lines.append("=" * 72)
    lines.append(f"COVERAGE  n={o['n']}  parse_fail_rate={o['parse_fail_rate']}")
    lines.append("=" * 72)
    return "\n".join(lines)
