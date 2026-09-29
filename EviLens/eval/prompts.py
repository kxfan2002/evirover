"""System prompt for single-turn, no-tool evaluation.

Derived from the SFT system prompt (identical to unified_system_prompt.txt) by
removing everything tied to agentic tool calling:
  - the "Use the provided tools" clause in the intro
  - Output Format branches (1) and (3) (tool_call turns)
  - the "# Tool Use Policy" section
  - the <tools>...</tools> block
Everything else (answer format rules, 0-1000 normalization, target notes) is kept
verbatim so the model answers in exactly the format the scorers expect.
"""

SYSTEM_PROMPT = """You are a multimodal agent. Analyze the given image, then answer the question.

# Output Format (strict)

Your response must be reasoning followed by a final answer:
<think>
...
</think>
<answer>
...
</answer>

Rules:
- The final answer must be inside <answer>...</answer>.
- Coordinates are normalized to 0-1000.
- Bounding boxes use xyxy format: [x_min, y_min, x_max, y_max].
- Points use [x, y].
- For grounding tasks, the answer is a bounding box: <answer>[x_min, y_min, x_max, y_max]</answer>.
- For segmentation tasks, the answer is a JSON object: <answer>{"boxes":[x_min,y_min,x_max,y_max],"positive_points":[[x,y],[x,y],[x,y]],"negative_points":[[x,y],[x,y],[x,y]]}</answer>.
- For counting tasks, the answer is an integer: <answer>N</answer>.
- For spot-the-difference tasks, the answer is a list of bounding boxes on the LEFT panel: <answer>[[x1,y1,x2,y2], [x1,y1,x2,y2], ...]</answer>.

# Important Notes About the Target

- The target is often SMALL and easy to miss. Pay attention to fine details, small text, subtle attributes, relative position, precise boundaries, and local context.
- Your bounding box must tightly fit the target — avoid overly large boxes. A good bbox should contain the target and little else.
- For spot-the-difference tasks: differences are often SMALL and subtle (colors, shapes, missing/added elements). All bounding boxes must be on the LEFT panel only (x coordinates in range 0-500).
"""

# Per-family reminder appended to the user turn, nailing down the exact answer shape.
# Union answer-format hints. These are byte-identical to the tails used in the SFT
# user turns and in eval/agent/prompts.py, so single-turn eval, agent eval, and
# training all state the answer format the same way. The spot_diff hint carries an
# {n} placeholder filled from the ground-truth difference count (see build_user_text).
FAMILY_ANSWER_HINT = {
    "grounding": (
        "Answer with a single bounding box for the target, normalized to 0-1000, "
        "in the format:\n<answer>[x_min, y_min, x_max, y_max]</answer>"
    ),
    "segmentation": (
        "Answer with a segmentation prompt as a JSON object containing one bounding box, "
        "3 positive points (clearly INSIDE the target), and 3 negative points "
        "(clearly OUTSIDE the target), all normalized to 0-1000, in the format:\n"
        '<answer>{"boxes":[x_min,y_min,x_max,y_max],"positive_points":[[x,y],[x,y],[x,y]],'
        '"negative_points":[[x,y],[x,y],[x,y]]}</answer>'
    ),
    "counting": (
        "Answer with a single integer in the format:\n<answer>N</answer>"
    ),
    "spot_diff": (
        "Answer with a list of bounding boxes on the LEFT panel only "
        "(x coordinates in 0-500), normalized to 0-1000, with one bounding box per "
        "difference (each box tightly around a single difference) and exactly {n} boxes, "
        "in the format:\n<answer>[[x1,y1,x2,y2], [x1,y1,x2,y2], ...]</answer>"
    ),
}


def build_user_text(description: str, family: str, n: int | None = None) -> str:
    """Compose the user turn text: the question plus a family-specific format hint.

    For spot_diff, pass `n` (the ground-truth difference count) so the hint states
    "exactly {n} boxes", matching the SFT/agent wording. Falls back to a generic
    plural if n is unavailable.
    """
    hint = FAMILY_ANSWER_HINT.get(family, "")
    if family == "spot_diff":
        hint = hint.format(n=n) if n else hint.replace(" and exactly {n} boxes", "")
    return f"{description.strip()}\n\n{hint}" if hint else description.strip()
