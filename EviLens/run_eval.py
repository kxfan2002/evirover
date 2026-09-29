#!/usr/bin/env python3
"""Single-turn, no-tool benchmark evaluation.

For each sample: build (system prompt, question, image) -> call an OpenAI-compatible
model -> extract <answer> -> score by task family -> aggregate.

Grounding/counting/spot_diff run concurrently over the API. Segmentation additionally
runs SAM3 locally (serial, single GPU) to turn the model's box+points into a mask.

Examples:
  python run_eval.py --base-url http://host:8000/v1 --api-key sk-xxx --model my-vlm
  python run_eval.py --dry-run                       # exercise pipeline with a mock model
  python run_eval.py --files counting grounding --limit 5
  python run_eval.py --no-sam                        # skip SAM; seg scored parse-only
"""
import argparse
import copy
import json
import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

from eval import config, dataset, parsing, prompts
from eval.client import ChatClient, MockClient
from eval.scorers import get_scorer
from eval import report

# Abort segmentation if more than this fraction of SAM calls fail. A bad image or
# degenerate box fails one sample; a broken SAM install fails essentially all of
# them, so the threshold separates the two cleanly.
SAM_ERROR_ABORT_FRAC = float(os.getenv("EVAL_SAM_ABORT_FRAC", "0.05"))

ANSWER_RESERVE_TURNS = 4


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--files", nargs="*", default=None, help="benchmark files (default: all)")
    p.add_argument("--base-url", default=config.DEFAULT_BASE_URL)
    p.add_argument("--api-key", default=config.DEFAULT_API_KEY)
    p.add_argument("--model", default=config.DEFAULT_MODEL)
    p.add_argument("--max-workers", type=int, default=config.DEFAULT_MAX_WORKERS)
    p.add_argument("--max-tokens", type=int, default=config.DEFAULT_MAX_TOKENS)
    p.add_argument("--timeout", type=int, default=config.DEFAULT_REQUEST_TIMEOUT,
                   help="per-request read timeout in seconds")
    p.add_argument("--temperature", type=float, default=config.DEFAULT_TEMPERATURE,
                   help="sampling temperature (pass a NEGATIVE value to omit the "
                        "field, for endpoints that reject it)")
    p.add_argument("--top-p", type=float, default=config.DEFAULT_TOP_P,
                   help="nucleus sampling top_p (negative to omit)")
    p.add_argument("--top-k", type=int, default=config.DEFAULT_TOP_K,
                   help="top_k sampling; vLLM/qwen extension (negative to omit for "
                        "strict OpenAI endpoints)")
    p.add_argument("--limit", type=int, default=None, help="max samples per file")
    p.add_argument("--ids", default="",
                   help="run only these sample ids: comma-separated, or a path to a "
                        "file with one id per line. An unknown id is a hard error, "
                        "never silently skipped.")
    p.add_argument("--out", default=config.RESULTS_DIR, help="results directory")
    p.add_argument("--no-sam", action="store_true", help="disable SAM3 (seg scored parse-only)")
    p.add_argument("--no-reasoning-parser", action="store_true",
                   help="[--model-path] serve vLLM without --reasoning-parser qwen3. "
                        "Required for non-thinking variants: the parser assumes output "
                        "starts inside a thinking block and splits on </think>, so a "
                        "model that never emits </think> has its entire output dropped "
                        "-- empty content and reasoning_content, with HTTP 200 and "
                        "finish_reason=stop, i.e. an all-zero run that looks healthy.")
    p.add_argument("--max-image-pixels", type=int, default=None,
                   help="downscale images above this pixel count before sending, "
                        "for endpoints with a hard pixel cap. GT is normalized so "
                        "scoring is unaffected. Default: send originals unchanged.")
    p.add_argument("--dry-run", action="store_true", help="use a mock model (no API calls)")
    p.add_argument("--no-resume", action="store_true", help="ignore existing results and re-run")
    p.add_argument("--coord-order", choices=["auto", "xy", "yx"], default="auto",
                   help="bbox/point coordinate order in model answers. 'auto' (default) "
                        "uses config.MODEL_COORD_ORDER by model name (gemini->yx, else xy). "
                        "Override with 'xy'/'yx' to force.")
    p.add_argument("--coord-space", choices=["norm1000", "pixel"], default="norm1000",
                   help="coordinate SCALE convention, orthogonal to --coord-order's axis "
                        "order. norm1000 (default) = the model follows the prompt and "
                        "emits 0-1000 normalized coordinates. pixel = the model emits "
                        "absolute pixels in the vision encoder's resized image space, "
                        "converted back to 0-1000 before scoring. A model in the wrong "
                        "space raises no error -- keys, parse_ok and finish_reason all "
                        "look healthy while IoU collapses to ~0. Implemented for "
                        "segmentation only; other families raise.")
    # Agent mode (multi-turn, with tools) vs default QA (single-turn, no tools).
    p.add_argument("--mode", choices=["qa", "agent"], default="qa",
                   help="qa: single-turn no-tool (default). agent: multi-turn tool loop.")
    # The tool budget is the one knob; --max-turns 0 derives itself from it so the
    # two can't drift into the silent failure where they are equal (see
    # ANSWER_RESERVE_TURNS). Pass --max-turns explicitly only to override.
    p.add_argument("--max-turns", type=int, default=0,
                   help=f"[agent] max conversation turns; 0 = max_tool_calls + {ANSWER_RESERVE_TURNS}")
    p.add_argument("--max-tool-calls", type=int, default=30, help="[agent] max total tool calls")
    p.add_argument("--final-recap", choices=["off", "rescue", "always"], default="off",
                   help="[agent] commit phase: re-present the model's own candidate regions "
                        "before scoring. off=legacy; rescue=only when no answer was produced; "
                        "always=also let an answered rollout revise.")
    p.add_argument("--pplx-key", default=os.environ.get("PPLX_API_KEY", ""),
                   help="[agent] Perplexity Search API key (text_search)")
    p.add_argument("--serper-key", default=os.environ.get("SERPER_API_KEY", ""),
                   help="[agent] Serper API key (image tools)")
    p.add_argument("--serper-budget", type=int, default=30,
                   help="[agent] max total Serper calls this run (scarce quota). "
                        "Ignored when --serper-per-sample > 0.")
    p.add_argument("--serper-per-sample", type=int, default=0,
                   help="[agent] per-sample Serper call budget. >0 enables per-sample "
                        "capping (fair, deterministic; each sample gets its own allotment) "
                        "and disables the run-wide --serper-budget cap.")
    p.add_argument("--repeat", type=int, default=1,
                   help="run R rollouts per question; the report averages them per "
                        "question before computing metrics, so n is unchanged. Use it "
                        "for categories too small to grow -- spot_diff has only 15 "
                        "questions and a single rollout is too noisy to rank models on.")
    p.add_argument("--repeat-tasks", default="",
                   help="comma-separated task names (e.g. spot_diff) to repeat; empty "
                        "= all. Filtering is by task because spot_diff records live "
                        "inside grounding/localization.jsonl and cannot be selected "
                        "with --files alone.")
    p.add_argument("--allow-missing-search-key", action="store_true",
                   help="[agent] start even without a search key. Refused by default: "
                        "every search then returns 'unavailable: no API key', the model "
                        "re-sends the same query until the tool budget is gone, and the "
                        "run produces a complete-looking but meaningless result.")
    p.add_argument("--allow-broken-browse", action="store_true",
                   help="[agent] start even when browse cannot fetch wikipedia. Refused "
                        "by default: behind a proxy, browse can fail for every page "
                        "while text_search keeps working, so the result looks normal.")
    p.add_argument("--allow-broken-summary", action="store_true",
                   help="[agent] start even when the summary endpoint probe fails. "
                        "Refused by default: browse silently falls back to raw page text "
                        "when the summarizer is unreachable, which is a different "
                        "protocol from a run that has one.")
    p.add_argument("--allow-broken-image-search", action="store_true",
                   help="[agent] start even when object storage is unconfigured. "
                        "Refused by default: Serper Lens only accepts a public URL, so "
                        "image_search must upload each crop; without it the tool fails "
                        "every time while the score is still computed.")
    p.add_argument("--text-search-backend", choices=["perplexity", "serper"],
                   default="perplexity",
                   help="[agent] provider for text_search. 'perplexity' (default) or "
                        "'serper' (Serper /search, uncapped — does not consume the image "
                        "budget). Lets you A/B the search API's effect on results.")

    p.add_argument("--summary-base-url", default=os.environ.get(
        "SUMMARY_BASE_URL", "https://"),
                   help="[agent] OpenAI-compatible base-url for the browse summarizer")
    p.add_argument("--summary-model", default=os.environ.get(
        "SUMMARY_MODEL", "qwen/qwen3.5-27b"),
                   help="[agent] model used to summarize browsed pages ('' disables -> raw text)")
    p.add_argument("--summary-key", default=(
        os.environ.get("SUMMARY_API_KEY")
        or os.environ.get("NOVITA_API_KEY")
        or os.environ.get("ANTHROPIC_AUTH_TOKEN", "")),
                   help="[agent] API key for the browse summarizer endpoint")
    p.add_argument("--browse-max-tokens", type=int, default=24000,
                   help="[agent] cap fetched page text at this many tokens before "
                        "summarization (token-based when a tokenizer is available)")
    p.add_argument("--browse-tokenizer", default="",
                   help="[agent] HF tokenizer path/id for token-based browse "
                        "truncation (default: --model-path if serving locally, else char cap)")
    # vLLM auto-launch: serve a local checkpoint and point the eval at it.
    p.add_argument("--model-path", default="",
                   help="local model dir to auto-serve with vLLM (sets base-url/model)")
    p.add_argument("--serve-gpu", default="",
                   help="[--model-path] CUDA device(s) for the vLLM server, e.g. '0' or '0,1'")
    p.add_argument("--serve-port", type=int, default=8000, help="[--model-path] vLLM port")
    args = p.parse_args()
    if args.max_turns <= 0:
        args.max_turns = args.max_tool_calls + ANSWER_RESERVE_TURNS
    return args


def load_done_ids(path):
    """Return {(id, rep): row} for completed samples, so a rerun resumes without redoing them.

    api_error rows are treated as NOT done — they were our-side failures (timeout /
    exhausted retries), not model answers, so a rerun retries them. Later lines for
    the same id win (a retried sample appends a fresh line below its old api_error).

    The key is (id, rep), not id: with --repeat R a question has R rows, and keying
    on id alone would make resume stop after the first one. Rows without `rep`
    default to 0, so older result files still load.
    """
    done = {}
    if os.path.isfile(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                key = (r["id"], int(r.get("rep", 0) or 0))
                if r.get("finish_reason") == "api_error":
                    done.pop(key, None)   # do not treat as done; supersede any prior
                    continue
                done[key] = r
    return done


def run_inference(sample, client):
    """Call the model and return (resp_dict, answer_text) or raise.

    resp_dict has content / reasoning_content / text. The answer is extracted from
    `text` (content, plus reasoning_content when the <answer> lives only there).
    """
    n_diff = len(sample.gt.get("bboxes") or []) if sample.task == "spot_diff" else None
    user_text = prompts.build_user_text(sample.description, sample.family, n_diff)
    resp = client.complete(prompts.SYSTEM_PROMPT, user_text, sample.image_path)
    return resp, parsing.extract_answer(resp["text"])


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    # SAM context (built lazily inside sam_backend on first seg sample).
    sam_ctx = None
    if not args.no_sam:
        # Pin SAM to this job's GPU. The main process does not set
        # CUDA_VISIBLE_DEVICES (only the vLLM child does), so without this every
        # parallel job would load its own SAM3 onto GPU 0.
        if args.serve_gpu and not os.environ.get("EVAL_SAM_GPU"):
            os.environ["EVAL_SAM_GPU"] = str(args.serve_gpu).split(",")[0].strip()
        from eval import sam_backend
        sam_ctx = {"segment": sam_backend.segment, "load_gt_mask": dataset.load_gt_mask}
    else:
        sam_ctx = {"segment": None, "load_gt_mask": dataset.load_gt_mask}

    # Optionally launch a local vLLM server for a checkpoint dir.
    server = None
    base_url, api_key, model = args.base_url, args.api_key, args.model
    if args.model_path and not args.dry_run:
        from eval import serve
        server = serve.VLLMServer(
            model_path=args.model_path, port=args.serve_port, gpu=args.serve_gpu,
            reasoning_parser=not args.no_reasoning_parser,
        )
        base_url, model = server.start()
        api_key = api_key or "EMPTY"
        print(f"[serve] vLLM up at {base_url} (model={model})")

    # Resolve bbox/point coordinate convention (auto by model name, or forced).
    coord_order = args.coord_order if args.coord_order != "auto" else config.coord_order_for(model)
    sam_ctx["coord_order"] = coord_order
    if coord_order != "xy":
        print(f"[coord] model '{model}' -> coordinate order '{coord_order}' "
              f"(box/points will be normalized to xyxy)")

    # `pixel` needs the vision encoder's actual resize parameters, which only the
    # model's own preprocessor_config.json defines. Read every field, never
    # default one: a wrong guess raises nothing and just skews IoU.
    sam_ctx["coord_space"] = args.coord_space
    if args.coord_space == "pixel":
        if not args.model_path:
            raise SystemExit("--coord-space pixel needs --model-path to read preprocessor_config.json; "
                             "a hosted API does not expose its resize convention")
        pp_path = os.path.join(args.model_path, "preprocessor_config.json")
        if not os.path.exists(pp_path):
            raise SystemExit(f"--coord-space pixel, but {pp_path} was not found")
        with open(pp_path) as fh:
            pp = json.load(fh)
        missing = [k for k in ("min_pixels", "patch_size", "merge_size") if k not in pp]
        if missing:
            raise SystemExit(f"{pp_path} is missing {missing}; cannot reproduce smart_resize")
        # env wins: it is the value actually passed to vLLM.
        env_mp = os.environ.get("VDR_EVAL_MM_MAX_PIXELS")
        if env_mp:
            max_pixels = int(env_mp)
        elif "max_pixels" in pp:
            max_pixels = int(pp["max_pixels"])
        else:
            raise SystemExit(f"VDR_EVAL_MM_MAX_PIXELS is unset and {pp_path} has no max_pixels")
        sam_ctx["resize_cfg"] = {
            "max_pixels": max_pixels,
            "min_pixels": int(pp["min_pixels"]),
            "factor": int(pp["patch_size"]) * int(pp["merge_size"]),
        }
        print(f"[coord] coord_space=pixel, smart_resize cfg={sam_ctx['resize_cfg']} "
              f"(max_pixels from: {'VDR_EVAL_MM_MAX_PIXELS' if env_mp else 'preprocessor_config'})")

    # Sampling params: pinned explicitly and sent to EVERY model (no silent
    # inheritance from each model's generation_config). A negative CLI value
    # means "omit this field" (for endpoints that reject it).
    def _omit_if_neg(v):
        return None if (v is None or v < 0) else v
    temperature = _omit_if_neg(args.temperature)
    top_p = _omit_if_neg(args.top_p)
    top_k = _omit_if_neg(args.top_k)

    # Client.
    if args.dry_run:
        if args.mode == "agent":
            from eval.agent.client import MockAgentClient
            client = MockAgentClient()
        else:
            client = MockClient()
        print(f"[dry-run] using mock client (no API calls), mode={args.mode}")
    elif args.mode == "agent":
        from eval.agent.client import AgentChatClient
        client = AgentChatClient(
            base_url=base_url, api_key=api_key, model=model,
            max_tokens=args.max_tokens, temperature=temperature,
            top_p=top_p, top_k=top_k,
            timeout=args.timeout,
        )
    else:
        client = ChatClient(
            base_url=base_url, api_key=api_key, model=model,
            max_tokens=args.max_tokens, temperature=temperature,
            top_p=top_p, top_k=top_k,
            timeout=args.timeout, max_image_pixels=args.max_image_pixels,
        )
    print(f"[sampling] temperature={temperature} top_p={top_p} top_k={top_k} "
          f"(None = field omitted)")

    # Agent-mode tools (built once; shared Serper budget across the run).
    agent_tools = None
    if args.mode == "agent":
        # Hard failure, not a warning: without a key the model burns its tool budget
        # on searches that always fail, and the result looks like a real regression.
        # Agent mode only.
        _need_serper = args.text_search_backend == "serper"
        _missing = []
        if _need_serper and not args.serper_key:
            _missing.append("SERPER_API_KEY (--serper-key; text_search backend=serper)")
        if args.text_search_backend == "perplexity" and not args.pplx_key:
            _missing.append("PPLX_API_KEY (--pplx-key; text_search backend=perplexity)")
        if _missing:
            print("[agent] Refusing to start: required search key(s) missing. Running without "
                  "them silently produces an invalid result.",
                  flush=True)
            for m in _missing:
                print(f"          - {m}", flush=True)
            print("        Override (knowing the tools are dead): --allow-missing-search-key", flush=True)
            if not args.allow_missing_search_key:
                sys.exit(3)
            print("[agent] WARNING: --allow-missing-search-key given; continuing. These numbers "
                  "cannot be tabulated against a run that had keys.", flush=True)

        from eval.agent.tools import build_tools, SerperBudget
        segment_fn = sam_ctx.get("segment") if sam_ctx else None
        agent_tools = build_tools(
            pplx_key=args.pplx_key, serper_key=args.serper_key,
            serper_budget=(None if args.serper_per_sample > 0
                           else SerperBudget(args.serper_budget)),
            segment_fn=segment_fn,
            summary_base_url=args.summary_base_url,
            summary_api_key=args.summary_key,
            summary_model=args.summary_model,
            browse_tokenizer_path=(args.browse_tokenizer or args.model_path or ""),
            browse_max_tokens=args.browse_max_tokens,
            text_search_backend=args.text_search_backend,
        )
        summ = f"{args.summary_model}" if (args.summary_model and args.summary_base_url) else "OFF (raw text)"
        btok = (args.browse_tokenizer or args.model_path or "")
        budget_desc = (f"per-sample={args.serper_per_sample}" if args.serper_per_sample > 0
                       else f"run-wide={args.serper_budget}")
        print(f"[agent] tools: {', '.join(sorted(agent_tools))} | "
              f"text_search={args.text_search_backend} "
              f"pplx={'set' if args.pplx_key else 'MISSING'} "
              f"serper={'set' if args.serper_key else 'MISSING'} "
              f"budget={budget_desc} | browse-summarizer={summ} "
              f"(key={'set' if args.summary_key else 'MISSING'}) | "
              f"browse-cap={args.browse_max_tokens}tok "
              f"({'tokenizer' if btok else 'char-fallback'})")

        # Same for browse's fetch egress: behind a proxy, search can work while every
        # page fetch fails, and the run still reports a plausible score.
        if "browse" in agent_tools:
            from eval.agent.tools.browse import BrowseTool
            _probe_url = "https://en.wikipedia.org/wiki/Nobel_Prize"
            # summarizer=None: probe the fetch egress only, without touching the
            # summary API.
            _probe = BrowseTool(summarizer=None).call({"url": _probe_url, "query": "probe"}, None)
            if _probe.ok:
                print(f"[agent] browse egress OK ({_probe_url}, {len(_probe.text)} chars)", flush=True)
            else:
                _why = next((l for l in _probe.text.splitlines() if "fetch failed" in l),
                            _probe.text[:200])
                print(f"[agent] Refusing to start: browse cannot fetch {_probe_url}: {_why}", flush=True)
                print("        Usually a missing http_proxy. A working text_search proves nothing "
                      "here -- Serper is reached directly, without the proxy.", flush=True)
                print("        Override (knowing browse is dead): --allow-broken-browse", flush=True)
                if not args.allow_broken_browse:
                    sys.exit(4)
                print("[agent] WARNING: --allow-broken-browse given; continuing. These numbers "
                      "cannot be tabulated against a run with working browse.", flush=True)

        # Same for image_search's object storage. Serper's Lens endpoint only accepts
        # a public URL, so each local crop must be uploaded first. Unconfigured, every
        # call fails while the run still produces a complete score.
        if "image_search" in agent_tools:
            from eval.agent.tools import oss_upload
            _oss_missing = [k for k, v in (
                ("ALIBABA_CLOUD_ACCESS_KEY_ID", oss_upload.OSS_ACCESS_KEY_ID),
                ("ALIBABA_CLOUD_ACCESS_KEY_SECRET", oss_upload.OSS_ACCESS_KEY_SECRET),
                ("OSS_ENDPOINT", oss_upload.OSS_ENDPOINT),
                ("OSS_BUCKET_NAME", oss_upload.OSS_BUCKET_NAME),
            ) if not v]
            if _oss_missing:
                print("[agent] Refusing to start: image_search needs object storage "
                      "(Serper Lens only accepts a public URL). Missing env:", flush=True)
                for m in _oss_missing:
                    print(f"          - {m}", flush=True)
                print("        Override (knowing image_search is dead): --allow-broken-image-search",
                      flush=True)
                if not args.allow_broken_image_search:
                    sys.exit(6)
                print("[agent] WARNING: --allow-broken-image-search given; continuing. These numbers "
                      "cannot be tabulated against a run with working image_search.", flush=True)
            else:
                print("[agent] image_search object storage configured "
                      f"(bucket={oss_upload.OSS_BUCKET_NAME})", flush=True)

        # The summarizer is validated the same way. browse silently falls back to raw
        # page text when a summary call fails, so a dead endpoint or expired key
        # downgrades the protocol without any visible symptom. Probe it for real.
        if not (args.summary_base_url and args.summary_model and args.summary_key):
            print("[agent] Refusing to start: summarizer not fully configured (needs "
                  "--summary-base-url + --summary-model + --summary-key).", flush=True)
            print("        Override (knowing browse falls back to raw text): --allow-broken-summary", flush=True)
            if not args.allow_broken_summary:
                sys.exit(5)
            print("[agent] WARNING: --allow-broken-summary given; continuing.", flush=True)
        else:
            from eval.agent.summarizer import SummaryClient
            try:
                _probe_sum = SummaryClient(
                    base_url=args.summary_base_url, api_key=args.summary_key,
                    model=args.summary_model, max_tokens=8, timeout=30, max_retries=1,
                ).summarize("Reply with the single word: ok.")
                print(f"[agent] summarizer OK ({args.summary_model}): {_probe_sum[:40]!r}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[agent] Refusing to start: summarizer probe failed: {type(e).__name__}: {e}", flush=True)
                print("        Usually an expired key or exhausted quota, or a wrong base-url/model.", flush=True)
                print("        Override (knowing browse falls back to raw text): --allow-broken-summary", flush=True)
                if not args.allow_broken_summary:
                    sys.exit(5)
                print("[agent] WARNING: --allow-broken-summary given; continuing. These numbers "
                      "cannot be tabulated against a run with a working summarizer.", flush=True)

    samples = dataset.load_samples(args.files, limit=args.limit)
    if args.ids:
        want = [x.strip() for x in args.ids.split(",") if x.strip()] \
            if not os.path.isfile(args.ids) else \
            [l.strip() for l in open(args.ids) if l.strip()]
        have = {s.id for s in samples}
        unknown = [i for i in want if i not in have]
        if unknown:
            # Silently dropping a mistyped id would shrink the denominator.
            sys.exit(f"[ids] these ids are not in the selected --files: {unknown}")
        samples = [s for s in samples if s.id in set(want)]
        print(f"[ids] running {len(samples)} selected samples only")
    files = sorted({s.file for s in samples})
    print(f"loaded {len(samples)} samples across {len(files)} files")

    _no_img = sorted({s.image_path for s in samples if not os.path.exists(s.image_path)})
    _no_mask = sorted({m for s in samples if (m := s.gt.get("mask_path"))
                       and not os.path.exists(m)})
    if _no_img or _no_mask:
        print(f"[data] Refusing to start: {len(_no_img)} image(s) and {len(_no_mask)} GT mask(s) missing:",
              flush=True)
        for p in (_no_img + _no_mask)[:10]:
            print(f"          - {os.path.relpath(p, config.EVAL_DIR)}", flush=True)
        if len(_no_img) + len(_no_mask) > 10:
            print(f"          ... and {len(_no_img) + len(_no_mask) - 10} more", flush=True)
        if _no_img:
            print("        Images are not in the git repo; fetch them first:\n"
                  "          python3 scripts/download_images.py", flush=True)
        sys.exit(7)

    # Only the segmentation scorer implements the pixel conversion. Reject up front:
    # the scorer call site is wrapped in `except Exception`, so raising there would be
    # swallowed into a score=0 row.
    if args.coord_space == "pixel":
        bad = sorted({config.TASK_SCORER[s.task] for s in samples})
        bad = [f for f in bad if f != "segmentation"]
        if bad:
            sys.exit(f"[coord] --coord-space pixel is not implemented for these scorers: {bad}. "
                     f"Select only segmentation with --files, or implement them.")

    all_rows = []
    for fname in files:
        fsamples = [s for s in samples if s.file == fname]
        out_path = os.path.join(args.out, f"{fname}.jsonl")
        done = {} if args.no_resume else load_done_ids(out_path)

        # --repeat: expand each sample into R shallow copies carrying `_rep`.
        rep_tasks = {t.strip() for t in args.repeat_tasks.split(",") if t.strip()}
        def _reps_for(s):
            if args.repeat <= 1:
                return 1
            return args.repeat if (not rep_tasks or s.task in rep_tasks) else 1
        expanded = []
        for s in fsamples:
            for k in range(_reps_for(s)):
                c = copy.copy(s)
                c._rep = k                      # noqa: SLF001
                expanded.append(c)
        todo = [s for s in expanded if (s.id, s._rep) not in done]
        n_rep = sum(1 for s in expanded if s._rep > 0)
        print(f"\n[{fname}] {len(fsamples)} samples"
              + (f" (+{n_rep} repeats -> {len(expanded)} rollouts)" if n_rep else "")
              + f", {len(done)} done, {len(todo)} to run")

        rows = list(done.values())
        family = fsamples[0].family
        is_seg = family == "segmentation"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        # Each result is scored and written the moment its inference completes, so
        # progress is durable — a slow/failed call can't stall the rest of the file,
        # and re-running resumes from whatever was already written.
        n_done = [0]

        def handle(s, resp, ans, err, fout):
            base = {
                "id": s.id, "rep": getattr(s, "_rep", 0),
                "file": fname, "family": family,
                "subcategory": s.subcategory, "task": s.task,
            }
            if err is not None:
                row = {
                    **base,
                    "score": 0.0, "parse_ok": False,
                    "components": {"api_error": err[:200]},
                    "pred": None, "answer": None,
                    "content": None, "reasoning_content": None,
                    "finish_reason": "api_error",
                }
            else:
                # Scorer is chosen per-sample by task (localization mixes
                # grounding_bbox and spot_diff in one file).
                scorer = get_scorer(config.TASK_SCORER[s.task])
                try:
                    res = scorer(ans, s, ctx=sam_ctx)
                except Exception as e:  # noqa: BLE001
                    traceback.print_exc()
                    res = type("R", (), {})()
                    res.score, res.parse_ok = 0.0, False
                    res.components, res.pred = {"scorer_error": str(e)[:200]}, None
                row = {
                    **base,
                    "score": float(res.score), "parse_ok": bool(res.parse_ok),
                    "components": res.components, "pred": res.pred,
                    "answer": ans,
                    "content": resp.get("content"),
                    "reasoning_content": resp.get("reasoning_content"),
                    "finish_reason": resp.get("finish_reason"),
                    "gt": s.gt,
                }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()
            rows.append(row)
            n_done[0] += 1
            if n_done[0] % 10 == 0 or n_done[0] == len(todo):
                print(f"  [{fname}] {n_done[0]}/{len(todo)}")

        def handle_agent_row(s, row, err, fout):
            """Write a row produced by the agent runner (already scored)."""
            if err is not None:
                row = {
                    "id": s.id, "rep": getattr(s, "_rep", 0),
                    "file": fname, "family": family,
                    "subcategory": s.subcategory, "task": s.task,
                    "score": 0.0, "parse_ok": False,
                    "components": {"api_error": err[:200]},
                    "pred": None, "answer": None, "content": None,
                    "reasoning_content": None, "finish_reason": "api_error",
                }
            row["rep"] = getattr(s, "_rep", 0)
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()
            rows.append(row)
            n_done[0] += 1
            if n_done[0] % 5 == 0 or n_done[0] == len(todo):
                print(f"  [{fname}] {n_done[0]}/{len(todo)}")

        with open(out_path, "a") as fout:
            if args.mode == "agent":
                from eval.agent.runner import run_agent_sample
                traj_dir = os.path.join(args.out, "trajectories", fname)
                os.makedirs(traj_dir, exist_ok=True)

                def run_one(s):
                    return run_agent_sample(
                        s, client, agent_tools, output_dir=traj_dir,
                        max_turns=args.max_turns, max_tool_calls=args.max_tool_calls,
                        sam_ctx=sam_ctx, serper_per_sample=args.serper_per_sample,
                        final_recap=args.final_recap,
                    )
                if args.max_workers <= 1:
                    for s in todo:
                        try:
                            handle_agent_row(s, run_one(s), None, fout)
                        except Exception as e:  # noqa: BLE001
                            handle_agent_row(s, None, str(e), fout)
                else:
                    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
                        fut = {ex.submit(run_one, s): s for s in todo}
                        for f in as_completed(fut):
                            s = fut[f]
                            try:
                                handle_agent_row(s, f.result(), None, fout)
                            except Exception as e:  # noqa: BLE001
                                handle_agent_row(s, None, str(e), fout)
            # All families run concurrently, segmentation included: SAM inference is
            # already serialized inside sam_backend, so only the API call would be
            # serialized here.
            elif args.max_workers <= 1:
                for s in todo:
                    try:
                        resp, ans = run_inference(s, client)
                        handle(s, resp, ans, None, fout)
                    except Exception as e:  # noqa: BLE001
                        handle(s, None, None, str(e), fout)
            else:
                with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
                    fut = {ex.submit(run_inference, s, client): s for s in todo}
                    for f in as_completed(fut):
                        s = fut[f]
                        try:
                            resp, ans = f.result()
                            handle(s, resp, ans, None, fout)
                        except Exception as e:  # noqa: BLE001
                            handle(s, None, None, str(e), fout)

        # SAM failure has to be loud. Its signature is score=0 with parse_ok=True,
        # exit code 0, and a summary reporting 0.0 -- nothing raises anywhere. Check
        # after each family and abort rather than emit a plausible all-zero table.
        if is_seg and rows:
            n_sam_err = sum(1 for r in rows
                            if "sam_error" in str((r.get("components") or {})))
            frac = n_sam_err / len(rows)
            if frac > SAM_ERROR_ABORT_FRAC:
                ex = next((str((r.get("components") or {}).get("sam_error"))
                           for r in rows
                           if "sam_error" in str((r.get("components") or {}))), "?")
                sys.exit(
                    f"\n[{fname}] ABORT: SAM failed on {n_sam_err}/{len(rows)} = "
                    f"{frac:.1%} (threshold {SAM_ERROR_ABORT_FRAC:.0%}). The "
                    f"segmentation scores from this run are not trustworthy.\n"
                    f"   First error: {ex}\n"
                    f"   Predictions are already in {out_path}; once SAM works, "
                    f"re-scoring them is enough -- no need to re-run the rollouts."
                )
            if n_sam_err:
                print(f"  [{fname}] WARNING: SAM failed on {n_sam_err}/{len(rows)} = "
                      f"{frac:.1%} (below threshold {SAM_ERROR_ABORT_FRAC:.0%}; continuing)")

        all_rows.extend(rows)

    summary = report.build_summary(all_rows)
    summary_path = os.path.join(args.out, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\n" + report.format_summary(summary))
    print(f"\nper-sample results: {args.out}/<file>.jsonl")
    print(f"summary: {summary_path}")

    if server is not None:
        server.stop()


if __name__ == "__main__":
    main()
