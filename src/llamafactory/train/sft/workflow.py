# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/summarization/run_summarization.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

from ...data import SFTDataCollatorWith4DAttentionMask, get_dataset, get_template_and_fix_tokenizer
from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from ...extras.misc import calculate_tps
from ...extras.packages import is_transformers_version_greater_than
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ..trainer_utils import create_modelcard_and_push
from .metric import ComputeAccuracy, ComputeMultiLabelMetrics, ComputeSimilarity, eval_logit_processor
from .trainer import CustomSeq2SeqTrainer


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


logger = get_logger(__name__)


def _load_multi_label_assets() -> tuple[list[str], list[tuple[int, int]]]:
    assets_dir = Path(__file__).resolve().parents[4] / "assets"
    with open(assets_dir / "labels_file.json", "r", encoding="utf-8") as f:
        label_mapping = json.load(f)
    label_list = [None] * len(label_mapping)
    for name, idx in label_mapping.items():
        label_list[idx] = name

    with open(assets_dir / "parent_child_pairs.json", "r", encoding="utf-8") as f:
        pairs = json.load(f)
    parent_child_pairs = [(int(parent), int(child)) for parent, child in pairs]
    return label_list, parent_child_pairs


def _parse_multi_label_text(text: str, num_labels: int) -> np.ndarray:
    matches = re.findall(r"<yes>|<no>", text)
    vector = np.zeros(num_labels, dtype=np.float32)
    for idx, tag in enumerate(matches[:num_labels]):
        vector[idx] = 1.0 if tag == "<yes>" else 0.0
    return vector


def _expand_with_ancestors(indices: set[int], parent_child_pairs: list[tuple[int, int]]) -> set[int]:
    parent_map: dict[int, list[int]] = {}
    for parent, child in parent_child_pairs:
        parent_map.setdefault(child, []).append(parent)

    closure = set(indices)
    stack = list(indices)
    while stack:
        node = stack.pop()
        for parent in parent_map.get(node, []):
            if parent not in closure:
                closure.add(parent)
                stack.append(parent)
    return closure


def _evaluate_generated_multi_label(prediction_file: Path, label_list: list[str], parent_child_pairs: list[tuple[int, int]]) -> dict[str, float]:
    predictions: list[np.ndarray] = []
    references: list[np.ndarray] = []
    num_labels = len(label_list)

    if not prediction_file.exists():
        logger.warning_rank0_once(f"Prediction file not found at {prediction_file}, skipping statistics results.")
        return {}

    with open(prediction_file, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            predictions.append(_parse_multi_label_text(record.get("predict", ""), num_labels))
            references.append(_parse_multi_label_text(record.get("label", ""), num_labels))

    if len(predictions) == 0:
        return {}

    preds = np.stack(predictions)
    labels = np.stack(references)
    eps = 1e-8

    hamming_loss = float(np.mean(preds != labels))
    subset_accuracy = float(np.mean(np.all(preds == labels, axis=1)))

    tp = float(np.sum(preds * labels))
    fp = float(np.sum(preds * (1 - labels)))
    fn = float(np.sum((1 - preds) * labels))
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    hier_tp = 0.0
    hier_fp = 0.0
    hier_fn = 0.0
    violations = 0.0
    total_pairs = len(parent_child_pairs) * preds.shape[0]
    for pred_vec, label_vec in zip(preds, labels):
        pred_set = {i for i, v in enumerate(pred_vec) if v == 1}
        label_set = {i for i, v in enumerate(label_vec) if v == 1}
        pred_aug = _expand_with_ancestors(pred_set, parent_child_pairs)
        label_aug = _expand_with_ancestors(label_set, parent_child_pairs)

        hier_tp += len(pred_aug & label_aug)
        hier_fp += len(pred_aug - label_aug)
        hier_fn += len(label_aug - pred_aug)

        for parent, child in parent_child_pairs:
            if pred_vec[child] == 1 and pred_vec[parent] == 0:
                violations += 1

    hier_precision = hier_tp / (hier_tp + hier_fp + eps)
    hier_recall = hier_tp / (hier_tp + hier_fn + eps)
    hier_f1 = 0.0 if hier_precision + hier_recall == 0 else 2 * hier_precision * hier_recall / (hier_precision + hier_recall)
    violation_rate = violations / (total_pairs + eps)

    return {
        "hamming_loss": hamming_loss,
        "subset_accuracy": subset_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "hierarchical_f1": hier_f1,
        "hierarchical_violation_rate": violation_rate,
    }


def run_sft(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[list["TrainerCallback"]] = None,
):
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]

    if getattr(data_args, "task_type", None) == "multi_label_sft_logits":
        tokenizer.add_special_tokens({"additional_special_tokens": ["<yes>", "<no>"]})
        model_args.resize_vocab = True
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="sft", **tokenizer_module)
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)

    if getattr(model, "is_quantized", False) and not training_args.do_train:
        setattr(model, "_hf_peft_config_loaded", True)  # hack here: make model compatible with prediction

    data_collator = SFTDataCollatorWith4DAttentionMask(
        template=template,
        model=model if not training_args.predict_with_generate else None,
        pad_to_multiple_of=8 if training_args.do_train else None,  # for shift short attention
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        block_diag_attn=model_args.block_diag_attn,
        attn_implementation=getattr(model.config, "_attn_implementation", None),
        compute_dtype=model_args.compute_dtype,
        **tokenizer_module,
    )

    # Metric utils
    metric_module = {}
    if getattr(data_args, "task_type", None) == "multi_label_sft_logits":
        if finetuning_args.eval_with_generate:
            metric_module["compute_metrics"] = ComputeMultiLabelMetrics()
    elif training_args.predict_with_generate:
        metric_module["compute_metrics"] = ComputeSimilarity(tokenizer=tokenizer)
    elif finetuning_args.compute_accuracy:
        metric_module["compute_metrics"] = ComputeAccuracy()
        metric_module["preprocess_logits_for_metrics"] = eval_logit_processor

    # Keyword arguments for `model.generate`
    gen_kwargs = generating_args.to_dict(obey_generation_config=True)

    # Compatible with Transformers v4 and Transformers v5
    if is_transformers_version_greater_than("4.58.0"):
        extra_ids = getattr(tokenizer, "additional_special_tokens_ids", None)
        if not isinstance(extra_ids, list):
            extra_special_tokens = getattr(tokenizer, "_extra_special_tokens", [])
            string_tokens = [str(t) for t in extra_special_tokens]
            extra_ids = tokenizer.convert_tokens_to_ids(string_tokens)
        all_eos_ids = [tokenizer.eos_token_id] + [i for i in extra_ids if i != -1]
        unique_eos_ids = list(dict.fromkeys(all_eos_ids))
        gen_kwargs["eos_token_id"] = unique_eos_ids
    else:
        gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id

    # Initialize our Trainer
    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        callbacks=callbacks,
        gen_kwargs=gen_kwargs,
        data_args=data_args,
        **dataset_module,
        **tokenizer_module,
        **metric_module,
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        if finetuning_args.include_effective_tokens_per_second:
            train_result.metrics["effective_tokens_per_sec"] = calculate_tps(
                dataset_module["train_dataset"], train_result.metrics, stage="sft"
            )

        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            keys = ["loss"]
            if data_args.task_type == "multi_label_sft_logits":
                keys += ["loss_sft", "loss_cls", "loss_bce", "loss_dice", "loss_hier"]
            if isinstance(dataset_module.get("eval_dataset"), dict):
                keys += sum(
                    [[f"eval_{key}_loss", f"eval_{key}_accuracy"] for key in dataset_module["eval_dataset"].keys()], []
                )
            else:
                keys += ["eval_loss", "eval_accuracy"]

            plot_loss(training_args.output_dir, keys=keys)

    if training_args.predict_with_generate:
        tokenizer.padding_side = "left"  # use left-padding in generation

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        logger.warning_rank0_once("Batch generation can be very slow. Consider using `scripts/vllm_infer.py` instead.")
        predict_results = trainer.predict(dataset_module["eval_dataset"], metric_key_prefix="predict", **gen_kwargs)
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)
        trainer.save_predictions(dataset_module["eval_dataset"], predict_results, generating_args.skip_special_tokens)

        if (
            getattr(data_args, "task_type", None) == "multi_label_sft_logits"
            and training_args.predict_with_generate
            and finetuning_args.statistics_results
        ):
            label_list, parent_child_pairs = _load_multi_label_assets()
            prediction_file = Path(training_args.output_dir) / "generated_predictions.jsonl"
            stats = _evaluate_generated_multi_label(prediction_file, label_list, parent_child_pairs)
            if stats:
                stats_file = Path(training_args.output_dir) / "generated_statistics.json"
                with open(stats_file, "w", encoding="utf-8") as f:
                    json.dump(stats, f, ensure_ascii=False, indent=2)
                logger.info_rank0(f"Saved multi-label generation statistics to {stats_file}")

    # Create model card
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)
