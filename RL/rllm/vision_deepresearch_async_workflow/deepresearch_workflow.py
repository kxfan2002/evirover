from io import BytesIO
import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any, List, Optional

from PIL import Image

from vision_deepresearch_async_workflow.deepresearch_agent import DeepResearchAgent
from vision_deepresearch_async_workflow.crop_containment_reward import (
    crop_shaping_terms,
    compose_crop_shaped_reward,
)
from rllm.agents.agent import Action, Episode, Step, Trajectory
from rllm.engine.rollout import RolloutEngine
from rllm.rewards.reward_fn import RewardFunction
from rllm.workflows.workflow import TerminationReason, Workflow

import base64


# Ground-truth fields are JSON-encoded strings inside extra_info (see
# data_prepare/register_rl_dataset.py). The verl data path hands extra_info to
# the workflow verbatim (agent_workflow_engine reads batch.non_tensor_batch
# ["extra_info"].tolist()), bypassing the dataset-level decode in
# train_deepresearch_workflow._decode_json_fields. The reward reads
# task_info["bbox_1000_xyxy"]/["answer"] as numeric lists, so a raw string GT
# makes iou_2d reject it and every grounding/spot_diff reward collapses to 0.
# Decode these fields for the reward call (a shallow copy keeps the agent's view
# of `task` — e.g. the True-Answer display — unchanged).
_GT_JSON_FIELDS = ("bbox_1000_xyxy", "answer")


def _decode_gt_fields(task: dict) -> dict:
    decoded = dict(task)
    for k in _GT_JSON_FIELDS:
        v = decoded.get(k)
        if isinstance(v, str):
            try:
                decoded[k] = json.loads(v)
            except (ValueError, TypeError):
                pass
    return decoded


def as_pil_image(image: Any) -> Image.Image | None:
    if isinstance(image, Image.Image):
        return image

    if isinstance(image, str) and image.startswith("data:image/"):
        try:
            _, encoded = image.split(",", 1)
            image_bytes = base64.b64decode(encoded)
            return Image.open(BytesIO(image_bytes)).convert("RGB")
        except Exception:  # noqa: BLE001
            return None

    if isinstance(image, dict):
        if "bytes" in image and image["bytes"] is not None:
            try:
                return Image.open(BytesIO(image["bytes"])).convert("RGB")
            except Exception:  # noqa: BLE001
                return None
        data_str = None
        if "data" in image and isinstance(image["data"], str):
            data_str = image["data"]
        elif "path" in image and isinstance(image["path"], str):
            data_str = image["path"]
        elif "url" in image and isinstance(image["url"], str):
            data_str = image["url"]
        if data_str:
            if data_str.startswith("data:image/"):
                try:
                    _, encoded = data_str.split(",", 1)
                    image_bytes = base64.b64decode(encoded)
                    return Image.open(BytesIO(image_bytes)).convert("RGB")
                except Exception:  # noqa: BLE001
                    return None
            try:
                return Image.open(data_str).convert("RGB")
            except Exception:  # noqa: BLE001
                return None

    if isinstance(image, str):
        try:
            return Image.open(image).convert("RGB")
        except Exception:  # noqa: BLE001
            return None

    return None


def _extract_action_from_response(response: str) -> Action:
    if "<tool_call>" in response and "</tool_call>" in response:
        tool_call_text = response.split("<tool_call>")[1].split("</tool_call>")[0]
        return Action(action={"type": "tool_call", "tool_call": tool_call_text.strip()})
    if "<answer>" in response and "</answer>" in response:
        answer = response.split("<answer>")[1].split("</answer>")[0].strip()
        return Action(action={"type": "final_answer", "answer": answer})
    return Action(action={"type": "reasoning", "content": response})


def _is_valid_format(content: str) -> bool:
    if not isinstance(content, str) or not content:
        return False
    pattern = (
        r"^<think>.*?</think>\s*(<tool_call>.*?</tool_call>|<answer>.*?</answer>)\s*$"
    )
    return re.match(pattern, content, re.DOTALL) is not None


def _format_reward_for_step(step: Step) -> float:
    if step.info.get("step_error"):
        return 0.0
    if _has_tool_error_observation(step.observation):
        return 0.0
    content = step.model_response if isinstance(step.model_response, str) else ""
    return 1.0 if _is_valid_format(content) else 0.0


def _has_tool_error_observation(observation: Any) -> bool:
    if not isinstance(observation, str):
        return False
    error_markers = (
        "[Json Parse Error]",
        "[Python Interpreter Error]",
        "Python execution error:",
        "PythonInterpreter tool not available",
        "PythonInterpreter tool is not callable",
    )
    return any(marker in observation for marker in error_markers)


def _get_next_observation(messages: list[dict], current_index: int) -> str:
    if current_index + 1 < len(messages):
        next_msg = messages[current_index + 1]
        if next_msg["role"] == "user" and "<tool_response>" in next_msg["content"]:
            return next_msg["content"]
    return ""


def _map_termination_reason(termination: str) -> TerminationReason:
    termination = str(termination or "")
    if termination.startswith("answer not found after "):
        termination = "turn_budget_no_answer"

    mapping = {
        "answer": TerminationReason.ENV_DONE,
        "turn_budget_no_answer": TerminationReason.ENV_DONE,
        "left_panel_violation": TerminationReason.ENV_DONE,
        "left_panel_no_answer": TerminationReason.ENV_DONE,
        "mixed_tool_and_answer": TerminationReason.ENV_DONE,
        "consecutive_empty_search": TerminationReason.ENV_DONE,
        "length_budget_no_answer": TerminationReason.UNKNOWN,
        "timeout": TerminationReason.UNKNOWN,
        "max_rounds_reached": TerminationReason.UNKNOWN,
        "token_limit_no_answer": TerminationReason.UNKNOWN,
        "answer_token_limit": TerminationReason.UNKNOWN,
        "exceed available llm calls": TerminationReason.UNKNOWN,
        "prompt_budget_reached": TerminationReason.UNKNOWN,
        "max_rounds_reached_no_answer": TerminationReason.UNKNOWN,
        "wait_loop_detected": TerminationReason.UNKNOWN,
        "repetition_detected": TerminationReason.UNKNOWN,
        "bad_format_no_answer": TerminationReason.UNKNOWN,
        "model_error_loop": TerminationReason.UNKNOWN,
        "tool_errors_too_many": TerminationReason.UNKNOWN,
        "consecutive_step_errors": TerminationReason.UNKNOWN,
        "error": TerminationReason.UNKNOWN,  # Will be masked
    }
    return mapping.get(termination, TerminationReason.UNKNOWN)


def _evaluate_answer(prediction: str, ground_truth: Any) -> bool:
    if not prediction or ground_truth is None:
        return False
    if not isinstance(ground_truth, str):
        return False
    return prediction.strip().lower() == ground_truth.strip().lower()


def _resolve_data_path(task: dict, value: Any) -> Any:
    """Resolve dataset-relative paths against task['data_root'] when present."""
    if not isinstance(value, str) or not value:
        return value
    if value.startswith("data:image/") or value.startswith("http://") or value.startswith("https://"):
        return value
    path = Path(value)
    if path.is_absolute() or os.path.exists(value):
        return value
    data_root = task.get("data_root") or os.getenv("MM_DATA_ROOT") or os.getenv("DATA_ROOT")
    if not data_root:
        return value
    return str(Path(str(data_root)) / value)


def _resolve_image_ref(task: dict, image_ref: Any) -> Any:
    if isinstance(image_ref, str):
        return _resolve_data_path(task, image_ref)
    if isinstance(image_ref, dict):
        resolved = dict(image_ref)
        for key in ("image", "path", "url", "data"):
            if key in resolved and isinstance(resolved[key], str):
                resolved[key] = _resolve_data_path(task, resolved[key])
        return resolved
    return image_ref


def _should_mask_episode(result: dict, episode: Episode) -> tuple[bool, str]:
    steps = episode.trajectories[0].steps if episode.trajectories else []
    termination = result.get("termination", "")

    if _map_termination_reason(termination) == TerminationReason.UNKNOWN:
        return True, termination or "no_final_answer"

    total_steps = len(steps)
    if total_steps > 0:
        infra_steps = sum(1 for step in steps if step.info.get("err_cls") == "infra_error")
        if infra_steps / total_steps > 0.5:
            print(
                f"VDR_TOOLERR_GATE_FIRED infra_steps={infra_steps}/{total_steps} "
                f"termination={termination}"
            )
            return True, "tool_errors_too_many"

    return False, ""


def _to_pil_image(img: Any) -> Optional[Image.Image]:
    """Best-effort conversion to PIL.Image for downstream multi-modal pipeline."""
    if isinstance(img, Image.Image):
        return img
    pil = as_pil_image(img)
    if pil is not None:
        return pil
    if isinstance(img, dict) and "bytes" in img:
        try:
            return Image.open(BytesIO(img["bytes"])).convert("RGB")
        except Exception:  # noqa: BLE001
            return None
    if isinstance(img, str):
        try:
            return Image.open(img).convert("RGB")
        except Exception:  # noqa: BLE001
            return None
    return None


class DeepResearchWorkflow(Workflow):
    def __init__(
        self,
        rollout_engine: RolloutEngine,
        executor,
        tools: dict | None = None,
        system_prompt: str | None = None,
        reward_function: RewardFunction | None = None,
        **kwargs,
    ):
        super().__init__(rollout_engine, executor, **kwargs)

        self.tools = tools or {}
        for tool in self.tools.values():
            if hasattr(tool, "set_executor"):
                tool.set_executor(self.executor)
        self.system_prompt = system_prompt
        self.reward_function = reward_function

        self.agent = DeepResearchAgent(
            rollout_engine=rollout_engine,
            tools=self.tools,
            system_prompt=self.system_prompt,
        )

    async def run(self, task: dict, uid: str, **kwargs) -> Episode:
        self.reset(task=task, uid=uid)

        question = task.get("question", task.get("query", "No question provided"))
        answer = task.get("answer", "")

        print(f"🚀 Starting DeepResearch workflow for task {uid}")
        print(f"   Question: {question}")

        try:
            tool_output_dir = kwargs.pop("tool_output_dir", None)
            raw_images = None
            if "images" in task:
                raw_images = task.get("images")
            elif task.get("image_path"):
                raw_images = [task.get("image_path")]

            resolved_images: list[Any] = []
            pil_images: List[Image.Image] = []
            if raw_images is not None:
                if not isinstance(raw_images, list):
                    raw_images = [raw_images]
                resolved_images = [_resolve_image_ref(task, img) for img in raw_images]
                for img in resolved_images:
                    pil = _to_pil_image(img)
                    if pil is not None:
                        pil_images.append(pil)

            if pil_images:
                result = await self.agent.run(
                    question=question,
                    answer=answer,
                    images=pil_images,
                    image_path=resolved_images[0] if resolved_images else None,
                    task=task,
                    uid=uid,
                    tool_output_dir=tool_output_dir,
                    **kwargs,
                )
            else:
                result = await self.agent.run(
                    question=question,
                    answer=answer,
                    task=task,
                    uid=uid,
                    tool_output_dir=tool_output_dir,
                    **kwargs,
                )

            episode = self._convert_result_to_episode(result, task, uid)

            prediction = result.get("prediction", "")
            scalar_reward: float | None = None
            if self.reward_function is not None and prediction:
                # Decode JSON-encoded GT so the reward sees numeric bbox lists.
                reward_task = _decode_gt_fields(task)
                try:
                    if asyncio.iscoroutinefunction(self.reward_function):
                        reward_out = await self.reward_function(reward_task, prediction)
                    else:
                        loop = asyncio.get_running_loop()
                        reward_out = await loop.run_in_executor(
                            self.executor,
                            lambda: self.reward_function(reward_task, prediction),
                        )
                except Exception as err:  # noqa: BLE001
                    print(f"Reward function failed: {err}")
                else:
                    if reward_out.is_correct is not None:
                        episode.is_correct = bool(reward_out.is_correct)
                    if isinstance(reward_out.metadata, dict):
                        reward_metadata = episode.info.setdefault("reward_metadata", {})
                        for key, value in reward_out.metadata.items():
                            if key not in reward_metadata:
                                reward_metadata[key] = value
                    if getattr(reward_out, "reward", None) is not None:
                        scalar_reward = float(reward_out.reward)
                        episode.info["reward_function_reward"] = scalar_reward

            # Check whether to mask the whole episode.
            should_mask_episode, mask_reason = _should_mask_episode(result, episode)

            if should_mask_episode:
                episode.termination_reason = TerminationReason.UNKNOWN
                episode.metrics = {
                    "reward/outcome": 0.0,
                    "masked": 1.0,
                }
                episode.info["mask_reason"] = mask_reason or result.get(
                    "termination", "unknown"
                )
            else:
                # No mask: train on scalar reward when a reward function provides it.
                # is_correct is retained only for logging / optional rejection sampling.
                outcome_reward = (
                    scalar_reward if scalar_reward is not None else 1.0 if episode.is_correct else 0.0
                )

                if result.get("had_length_truncation"):
                    outcome_reward = 0.0
                    episode.is_correct = False

                # Crop-shaping process reward (v14 exp1) — env-gated, DEFAULT OFF.
                # R = clamp(R_out + λ·φ_best·(1−R_out) − β·wander, 0, 1); φ = containment×
                # sat_zoom×context_ok on ORIGINAL-frame crops (see crop_containment_reward.py
                # + RL_EXPERIMENT_LOG.md §D/E/F). φ_best/wander/bonus/penalty logged separately.
                crop_info: dict = {}
                if os.getenv("MM_REWARD_CROP_SHAPING", "0") == "1":
                    shaping_task = _decode_gt_fields(task)
                    if "ground" in str(shaping_task.get("task", "")).lower():
                        terms = crop_shaping_terms(
                            result,
                            shaping_task.get("bbox_1000_xyxy"),
                            task=shaping_task.get("task"),
                        )
                        outcome_reward, crop_info = compose_crop_shaped_reward(
                            outcome_reward,
                            terms,
                            lam=float(os.getenv("MM_REWARD_CROP_LAMBDA", "0.3")),
                            beta=float(os.getenv("MM_REWARD_CROP_BETA", "0.2")),
                        )
                        if crop_info:
                            episode.info["crop_shaping"] = crop_info

                for trajectory in episode.trajectories:
                    if not trajectory.steps:
                        continue
                    trajectory.reward = outcome_reward

                    # last_step = trajectory.steps[-1]
                    # last_step.reward = trajectory.reward
                    # trajectory.steps = [last_step]

                episode.metrics = {
                    "reward/outcome": outcome_reward,
                    "reward/is_correct": 1.0 if episode.is_correct else 0.0,
                    "masked": 0.0,
                    "gen/length_truncated": 1.0 if result.get("had_length_truncation") else 0.0,
                }
                comp = (episode.info.get("reward_metadata") or {}).get("components") or {}
                if str(task.get("task") or "") == "segmentation":
                    episode.metrics["seg/sam_fallback"] = float(comp.get("sam_fallback", 0.0))
                for _cls in ("model_error", "infra_error", "empty_result", "budget"):
                    episode.metrics[f"tool_fail/{_cls}"] = float(
                        (result.get("tool_error_classes") or []).count(_cls))
                episode.metrics["tool_fail/total"] = float(
                    len(result.get("tool_error_classes") or []))
                if crop_info:
                    episode.metrics.update({f"crop/{k}": v for k, v in crop_info.items()})
                    # φ_best split by FINAL-answer correctness. Emitted conditionally so
                    # the trainer's per-key np.mean gives E[φ_best | correct] resp.
                    # E[φ_best | incorrect] (a missing key is simply not appended → no NaN).
                    # This discriminates the two φ_best-suppression hypotheses:
                    #   correct < incorrect  → mode-A wins (low-crop lucky answers) dragging
                    #                           φ_best down = R_out overpowers φ → tune λ/coupling;
                    #   correct >= incorrect → crop→answer working, any φ_best drop is
                    #                           β/exploration → tune β.
                    # Zero-crop applicable episodes (φ_best=0) are included on purpose: a
                    # correct zero-crop answer IS the mode-A signal we want to catch.
                    phi_best_val = float(crop_info.get("crop_phi_best", 0.0))
                    if episode.is_correct:
                        episode.metrics["crop/phi_best_correct"] = phi_best_val
                    else:
                        episode.metrics["crop/phi_best_incorrect"] = phi_best_val

                    # --- v15 PRIMARY behavioural metric ---------------------
                    # These are CONDITIONAL rates, and the trainer takes a plain
                    # per-key np.mean over whichever episodes emitted the key. So
                    # emit them only on the conditioning subset; the mean is then
                    # the conditional probability (same trick as phi_best_correct).
                    #   retry_given_bad_first = P(made a 2nd crop | 1st crop missed)
                    #     baseline 51.3% -- THE number v15 is meant to move.
                    #   phi_gain_given_retry = E[phi_best - phi_first | retried]
                    #     under v14 this was ~0 for 81.6% of retries (phi saturated);
                    #     if it stays ~0 under v15 the new phi still has no range.
                    if float(crop_info.get("crop_first_bad", 0.0)) >= 1.0:
                        episode.metrics["crop/retry_given_bad_first"] = float(
                            crop_info.get("crop_retried", 0.0)
                        )
                    if float(crop_info.get("crop_retried", 0.0)) >= 1.0:
                        episode.metrics["crop/phi_gain_given_retry"] = float(
                            crop_info.get("crop_phi_gain_from_retry", 0.0)
                        )
                    #   left_given_found = P(ended off-target | ever framed the target)
                    #     baseline 0.262. Abandoning costs +0.075 hit@0.5 WITHIN task, so
                    #     this should FALL. It is the counterpart to retry_given_bad_first:
                    #     together they say "keep looking until you find it, then commit".
                    if float(crop_info.get("crop_contained", 0.0)) >= 1.0:
                        episode.metrics["crop/left_given_found"] = float(
                            crop_info.get("crop_found_then_left", 0.0)
                        )

            print(f"✅ DeepResearch workflow completed for task {uid}")
            print(f"   Prediction: {result.get('prediction', 'No prediction')}")
            print(f"   True Answer: {answer}")
            print(f"   Metrics: {episode.metrics}")
            if episode.info.get("mask_reason"):
                print(f"   Mask Reason: {episode.info['mask_reason']}")
            return episode

        except Exception as exc:  # noqa: BLE001
            print(f"❌ DeepResearch workflow failed for task {uid}: {exc}")
            episode = Episode()
            episode.id = uid
            episode.task = task
            episode.termination_reason = TerminationReason.ERROR
            episode.is_correct = False
            episode.trajectories = []
            episode.metrics = {
                "reward/outcome": 0.0,
                "masked": 1.0,
            }
            episode.info = {"error": str(exc)}
            return episode

    def _convert_result_to_episode(self, result: dict, task: dict, uid: str) -> Episode:
        messages = result.get("messages", [])
        prediction = result.get("prediction", "")
        termination = result.get("termination", "unknown")
        rounds = result.get("rounds", 0)
        time_taken = result.get("time_taken", 0.0)

        trajectories: list[Trajectory] = []
        steps: list[Step] = []

        i = 0
        while i < len(messages):
            msg = messages[i]
            if msg["role"] == "assistant":
                context = messages[: i + 1]
                assistant_content = msg.get("content", "")
                action = _extract_action_from_response(assistant_content)
                observation = _get_next_observation(messages, i)
                step = Step(
                    chat_completions=context.copy(),
                    model_response=assistant_content,
                    action=action,
                    observation=observation,
                    reward=0.0,
                )
                step.model_output = None
                if msg.get("step_error"):
                    step.info["step_error"] = True
                if msg.get("err_cls"):
                    step.info["err_cls"] = msg["err_cls"]
                if _has_tool_error_observation(observation):
                    step.info["step_error"] = True
                steps.append(step)
            i += 1

        trajectory = Trajectory(
            name="deepresearch_agent",
            task=task,
            steps=steps,
            reward=0.0,
            info={},
        )
        trajectories.append(trajectory)

        answer_text = task.get("answer", "")
        is_correct = _evaluate_answer(prediction, answer_text) if answer_text else False

        episode = Episode()
        episode.id = uid
        episode.task = task
        episode.termination_reason = _map_termination_reason(termination)
        episode.is_correct = is_correct
        episode.trajectories = trajectories
        episode.metrics = {}
        episode.info = {
            "rounds": rounds,
            "time_taken": time_taken,
            "prediction": prediction,
            "answer": answer_text,
            "token_usage": result.get("token_usage", {}),
        }
        return episode

    def reset(self, task: dict | None = None, uid: str | None = None):
        # MultiTurnReactAgent handles per-run state; nothing to reset here.
        return

    def is_multithread_safe(self) -> bool:
        return True


__all__ = ["DeepResearchWorkflow"]
