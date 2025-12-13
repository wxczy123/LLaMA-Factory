# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
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
import os
import re
from types import MethodType
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import SaveProcessorCallback
from ..fp8_utils import configure_fp8_environment, verify_fp8_status
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler


if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import DataArguments, FinetuningArguments, ModelArguments


logger = logging.get_logger(__name__)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE."""

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        model_args: Optional["ModelArguments"] = None,
        data_args: Optional["DataArguments"] = None,
        gen_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        # Configure FP8 environment if enabled
        if model_args is not None and model_args.fp8:
            configure_fp8_environment(model_args)
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        else:
            self.processing_class: PreTrainedTokenizer = kwargs.get("tokenizer")

        super().__init__(**kwargs)
        if processor is not None:
            # avoid wrong loss under gradient accumulation
            # https://github.com/huggingface/transformers/pull/36044#issuecomment-2746657112
            self.model_accepts_loss_kwargs = False

        self.finetuning_args = finetuning_args
        if gen_kwargs is not None:
            # https://github.com/huggingface/transformers/blob/v4.45.0/src/transformers/trainer_seq2seq.py#L287
            self._gen_kwargs = gen_kwargs

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

        if finetuning_args.use_dft_loss:
            from ..trainer_utils import dft_loss_func

            self.compute_loss_func = dft_loss_func

        # Verify FP8 status after trainer initialization (accelerator should be available)
        if model_args is not None and model_args.fp8 and hasattr(self, "accelerator"):
            verify_fp8_status(self.accelerator, model_args)

        self.data_args = data_args
        self._is_multi_label_task = getattr(data_args, "task_type", None) == "multi_label_sft_logits"
        self._label_list: Optional[list[str]] = None
        self._parent_child_pairs: Optional[list[tuple[int, int]]] = None
        self._pos_weight: Optional[torch.Tensor] = None
        self._num_labels: Optional[int] = None
        if self._is_multi_label_task:
            self._init_multi_label_resources()

    def _init_multi_label_resources(self) -> None:
        assets_dir = Path(__file__).resolve().parents[5] / "assets"
        labels_path = assets_dir / "labels_file.json"
        parent_child_path = assets_dir / "parent_child_pairs.json"

        with open(labels_path, "r", encoding="utf-8") as f:
            label_mapping = json.load(f)
        self._label_list = [None] * len(label_mapping)
        for name, idx in label_mapping.items():
            self._label_list[idx] = name
        self._num_labels = len(self._label_list)

        with open(parent_child_path, "r", encoding="utf-8") as f:
            pairs = json.load(f)
        self._parent_child_pairs = [(int(parent), int(child)) for parent, child in pairs]

        self.yes_id = self.processing_class.convert_tokens_to_ids("<yes>")
        self.no_id = self.processing_class.convert_tokens_to_ids("<no>")
        if self.yes_id == self.processing_class.unk_token_id or self.no_id == self.processing_class.unk_token_id:
            raise ValueError("<yes>/<no> tokens must be added to the tokenizer before training.")

        self._maybe_init_pos_weight()

    def _maybe_init_pos_weight(self) -> None:
        if not self.finetuning_args.use_pos_weight:
            self._pos_weight = None
            return

        if self.finetuning_args.pos_weight_file is not None:
            with open(self.finetuning_args.pos_weight_file, "r", encoding="utf-8") as f:
                weights = json.load(f)
            self._pos_weight = torch.tensor(weights, dtype=torch.float32)
            if self._pos_weight.numel() != self._num_labels:
                raise ValueError(
                    f"Expected pos_weight length {self._num_labels}, got {self._pos_weight.numel()}."
                )
            return

        if getattr(self.data_args, "streaming", False) or self.train_dataset is None:
            self._pos_weight = None
            return

        pos_counts = np.zeros(self._num_labels, dtype=np.float64)
        neg_counts = np.zeros(self._num_labels, dtype=np.float64)
        for example in self.train_dataset:
            targets = example.get("binary_targets")
            if targets is None:
                continue
            arr = np.array(targets, dtype=np.float64)
            pos_counts += arr
            neg_counts += 1.0 - arr

        raw = neg_counts / (pos_counts + 1e-8)
        raw = np.clip(raw, 0.0, self.finetuning_args.pos_weight_max)
        self._pos_weight = torch.tensor(raw, dtype=torch.float32)

    def _classification_enabled(self) -> bool:
        return bool(
            self.finetuning_args.use_bce_loss
            or self.finetuning_args.use_dice_loss
            or self.finetuning_args.use_hier_loss
        )

    def _extract_binary_logits(self, logits: torch.Tensor, label_positions: torch.Tensor) -> torch.Tensor:
        logits_label_tokens = logits.gather(
            dim=1, index=label_positions.unsqueeze(-1).expand(-1, -1, logits.size(-1))
        )
        logits_yes = logits_label_tokens[..., self.yes_id]
        logits_no = logits_label_tokens[..., self.no_id]
        return logits_yes - logits_no

    def _compute_classification_losses(
        self, binary_logits: torch.Tensor, binary_targets: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        loss_cls = torch.zeros((), device=binary_logits.device, dtype=binary_logits.dtype)
        losses: dict[str, torch.Tensor] = {}
        probs = torch.sigmoid(binary_logits)

        if self.finetuning_args.use_bce_loss:
            pos_weight = self._pos_weight.to(binary_logits.device) if self._pos_weight is not None else None
            if pos_weight is not None:
                pos_weight = pos_weight.to(dtype=binary_logits.dtype)
            loss_bce = F.binary_cross_entropy_with_logits(binary_logits, binary_targets, pos_weight=pos_weight)
            losses["loss_bce"] = loss_bce
            loss_cls = loss_cls + self.finetuning_args.lambda_bce * loss_bce

        if self.finetuning_args.use_dice_loss:
            smooth = 1e-6
            intersection = (probs * binary_targets).sum(dim=-1)
            union = probs.sum(dim=-1) + binary_targets.sum(dim=-1)
            dice_score = (2 * intersection + smooth) / (union + smooth)
            loss_dice = 1.0 - dice_score.mean()
            losses["loss_dice"] = loss_dice
            loss_cls = loss_cls + self.finetuning_args.lambda_dice * loss_dice

        if self.finetuning_args.use_hier_loss and self._parent_child_pairs is not None:
            violations = []
            for parent_idx, child_idx in self._parent_child_pairs:
                p_parent = probs[:, parent_idx]
                p_child = probs[:, child_idx]
                violations.append(F.relu(p_child - p_parent))
            if violations:
                loss_hier = torch.stack(violations, dim=0).mean()
                losses["loss_hier"] = loss_hier
                loss_cls = loss_cls + self.finetuning_args.lambda_hier * loss_hier

        losses["loss_cls"] = loss_cls
        return loss_cls, losses

    def _log_multi_label_losses(self, logs: dict[str, torch.Tensor]) -> None:
        safe_logs = {}
        for key, value in logs.items():
            if value is None:
                continue
            if isinstance(value, torch.Tensor):
                value = value.detach().to(torch.float32).item()
            safe_logs[key] = value
        if safe_logs:
            self.log(safe_logs)

    def _generate_multi_label(self, prompts: list[str], model: "torch.nn.Module") -> list[str]:
        if len(prompts) == 0:
            return []
        inputs = self.processing_class(
            prompts, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False
        ).to(model.device)
        gen_kwargs = getattr(self, "_gen_kwargs", {})
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
        return self.processing_class.batch_decode(outputs, skip_special_tokens=True)

    def _parse_generated_predictions(self, texts: list[str]) -> torch.Tensor:
        preds = []
        for text in texts:
            matches = re.findall(r"<yes>|<no>", text)
            vector = np.zeros(self._num_labels, dtype=np.float32)
            for idx, tag in enumerate(matches[: self._num_labels]):
                vector[idx] = 1.0 if tag == "<yes>" else 0.0
            preds.append(vector)

        return torch.tensor(np.array(preds), dtype=torch.float32)


    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler(*args, **kwargs)

    @override
    def compute_loss(self, model, inputs, *args, **kwargs):
        if self._is_multi_label_task:
            return_outputs = kwargs.pop("return_outputs", False)
            model_inputs = {
                k: v
                for k, v in inputs.items()
                if k not in {"label_positions", "binary_targets", "full_text", "prompt_text"}
            }
            outputs = model(**model_inputs)
            sft_loss = outputs.loss
            logs: dict[str, torch.Tensor] = {}
            if sft_loss is not None:
                logs["loss_sft"] = sft_loss

            if self._classification_enabled() and "label_positions" in inputs and "binary_targets" in inputs:
                binary_logits = self._extract_binary_logits(outputs.logits, inputs["label_positions"])
                binary_targets = inputs["binary_targets"].to(dtype=binary_logits.dtype)
                loss_cls, loss_parts = self._compute_classification_losses(binary_logits, binary_targets)
                logs.update(loss_parts)
                loss = self.finetuning_args.lambda_sft * sft_loss + loss_cls if sft_loss is not None else loss_cls
            else:
                loss = sft_loss

            if loss is not None:
                logs["loss"] = loss
            self._log_multi_label_losses(logs)

            return (loss, outputs) if return_outputs else loss

        return super().compute_loss(model, inputs, *args, **kwargs)

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        if self._is_multi_label_task:
            inputs = self._prepare_inputs(inputs)
            label_positions = inputs.pop("label_positions", None)
            binary_targets = inputs.pop("binary_targets", None)
            prompt_text = inputs.pop("prompt_text", inputs.pop("full_text", None))

            if self.finetuning_args.use_teacher_forcing_logits:
                with torch.no_grad():
                    outputs = model(**inputs)
                    loss = outputs.loss
                    binary_logits = self._extract_binary_logits(outputs.logits, label_positions)
                    if self._classification_enabled() and binary_targets is not None:
                        binary_targets_cast = binary_targets.to(dtype=binary_logits.dtype)
                        loss_cls, _ = self._compute_classification_losses(binary_logits, binary_targets_cast)
                        loss = self.finetuning_args.lambda_sft * loss + loss_cls if loss is not None else loss_cls

                    probs = torch.sigmoid(binary_logits)
                    preds = (probs >= 0.5).float()
                if prediction_loss_only:
                    return loss, None, None
                return loss, preds, binary_targets

            prompts = prompt_text if isinstance(prompt_text, list) else []
            generated_texts = self._generate_multi_label(prompts, model)
            preds = self._parse_generated_predictions(generated_texts)
            if prediction_loss_only:
                return None, None, None
            return None, preds, binary_targets

        # default behavior
        r"""Remove the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        if self.args.predict_with_generate:  # do not pass labels to model when generate
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def save_predictions(
        self, dataset: "Dataset", predict_results: "PredictionOutput", skip_special_tokens: bool = True
    ) -> None:
        r"""Save model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.processing_class.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        decoded_inputs = self.processing_class.batch_decode(dataset["input_ids"], skip_special_tokens=False)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")
