import hydra
import json
import os

from vision_deepresearch_async_workflow.deepresearch_tools_async_executor import (
    get_all_tools,
    get_local_tools,
)
from vision_deepresearch_async_workflow.deepresearch_workflow import (
    DeepResearchWorkflow,
)

from rllm.data.dataset import DatasetRegistry
from rllm.rewards.reward_fn import mm_deepresearch_reward_fn_async
from rllm.trainer.agent_trainer import AgentTrainer

# Fields JSON-encoded at register time (see data_prepare/register_rl_dataset.py
# JSON_ENCODED_FIELDS) because they are type-heterogeneous across task families.
# Decode them back to Python objects so the reward (which reads task_info as-is)
# sees the original lists/scalars.
_JSON_ENCODED_FIELDS = ("bbox_1000_xyxy", "answer")

# Which registered dataset to train on. Default keeps the full unified RL data.
# For the no-search demo set RLLM_DATASET=Vision-DeepResearch-QA-nosearch.
_DATASET_NAME = os.environ.get("RLLM_DATASET", "Vision-DeepResearch-QA")
# Tool policy: "all" = all 9 tools (default); "local" = no-network tools only
# (crop/verify/verify_part/verify_mask/python) for the no-search demo.
_TOOLSET = os.environ.get("RLLM_TOOLSET", "all").strip().lower()


def _build_tools():
    if _TOOLSET == "local":
        return get_local_tools()
    return get_all_tools()


def _decode_json_fields(dataset):
    if dataset is None:
        return dataset
    for row in dataset.data:
        for k in _JSON_ENCODED_FIELDS:
            v = row.get(k)
            if isinstance(v, str):
                try:
                    row[k] = json.loads(v)
                except (ValueError, TypeError):
                    pass
    return dataset


@hydra.main(
    config_path="pkg://rllm.trainer.config",
    config_name="agent_ppo_trainer",
    version_base=None,
)
def main(config):
    train_dataset = _decode_json_fields(
        DatasetRegistry.load_dataset(_DATASET_NAME, "train")
    )
    if train_dataset is None or len(train_dataset) == 0:
        raise ValueError(f"{_DATASET_NAME}/train is missing or empty")

    test_dataset = _decode_json_fields(
        DatasetRegistry.load_dataset(_DATASET_NAME, "test")
    )
    if test_dataset is None or len(test_dataset) == 0:
        # The unified RL data intentionally has no internal benchmark split.
        # Disable validation completely instead of mirroring train samples.
        test_dataset = None
        if hasattr(config, "data"):
            config.data.val_files = []

    trainer = AgentTrainer(
        workflow_class=DeepResearchWorkflow,
        workflow_args={
            "reward_function": mm_deepresearch_reward_fn_async,
            "tools": _build_tools(),
        },
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
