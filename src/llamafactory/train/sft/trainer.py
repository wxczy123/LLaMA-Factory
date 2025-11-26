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
from types import MethodType
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
from .label_space import LABEL_COUNT, LEAF_LABEL_INDICES, PARENT_CHILD_EDGES


if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments, ModelArguments


logger = logging.get_logger(__name__)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE."""

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        model_args: Optional["ModelArguments"] = None,
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

        self._one_token_id: Optional[int] = None
        self._warned_label_count = False
        tokenizer = getattr(self, "processing_class", None)
        if tokenizer is not None:
            one_token_ids = tokenizer.encode("1", add_special_tokens=False)
            if len(one_token_ids) == 1:
                self._one_token_id = one_token_ids[0]
            else:
                logger.warning_rank0("Tokenizer encodes digit `1` into multiple tokens: %s", one_token_ids)

        # Verify FP8 status after trainer initialization (accelerator should be available)
        if model_args is not None and model_args.fp8 and hasattr(self, "accelerator"):
            verify_fp8_status(self.accelerator, model_args)

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
    def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs):
        bce_weight = getattr(self.finetuning_args, "classification_loss_weight", 0.0)
        dice_weight = getattr(self.finetuning_args, "classification_dice_weight", 0.0)
        hier_weight = getattr(self.finetuning_args, "classification_hier_weight", 0.0)
        need_outputs = return_outputs or any(weight > 0 for weight in (bce_weight, dice_weight, hier_weight))

        if not need_outputs:
            return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

        loss, outputs = super().compute_loss(model, inputs, return_outputs=True, **kwargs)

        if need_outputs:
            cls_loss, components = self._compute_classification_loss(
                outputs, inputs, bce_weight=bce_weight, dice_weight=dice_weight, hier_weight=hier_weight
            )
            if cls_loss is not None:
                loss = loss + cls_loss
                if isinstance(outputs, dict):
                    outputs = dict(outputs)
                    outputs.update({k: v.detach() for k, v in components.items()})
                    outputs["loss"] = loss.detach()

        if return_outputs:
            return loss, outputs
        return loss

    def _compute_classification_loss(
        self,
        outputs: Any,
        inputs: dict[str, torch.Tensor],
        *,
        bce_weight: float,
        dice_weight: float,
        hier_weight: float,
    ) -> tuple[Optional[torch.Tensor], dict[str, torch.Tensor]]:
        components: dict[str, torch.Tensor] = {}

        if "labels" not in inputs:
            return None, components

        logits = self._get_logits(outputs)
        tokenizer = getattr(self, "processing_class", None)
        if logits is None or tokenizer is None or self._one_token_id is None:
            return None, components

        samples = self._gather_classification_tensors(inputs["labels"], logits, tokenizer)
        if not samples:
            return None, components

        total_loss: torch.Tensor | None = None

        if bce_weight > 0:
            stacked_logits = torch.cat([item[0] for item in samples])
            stacked_targets = torch.cat([item[1] for item in samples])
            bce_loss = F.binary_cross_entropy_with_logits(stacked_logits, stacked_targets)
            components["classification_bce_loss"] = bce_loss
            total_loss = bce_weight * bce_loss

        if dice_weight > 0:
            dice_loss = self._compute_dice_loss(samples)
            if dice_loss is not None:
                components["classification_dice_loss"] = dice_loss
                total_loss = dice_loss * dice_weight if total_loss is None else total_loss + dice_loss * dice_weight

        if hier_weight > 0:
            hier_loss = self._compute_hierarchical_loss(samples)
            if hier_loss is not None:
                components["classification_hier_loss"] = hier_loss
                total_loss = hier_loss * hier_weight if total_loss is None else total_loss + hier_loss * hier_weight

        return total_loss, components

    def _extract_label_tokens(
        self, sample_labels: torch.Tensor, tokenizer: "PreTrainedTokenizer"
    ) -> tuple[list[int], list[int], list[int]]:
        mask = sample_labels.ne(IGNORE_INDEX)
        if not mask.any():
            return [], [], []

        valid_positions = mask.nonzero(as_tuple=False).view(-1)
        valid_ids = sample_labels[valid_positions].tolist()
        tokens = tokenizer.convert_ids_to_tokens(valid_ids)

        positions: list[int] = []
        targets: list[int] = []
        label_indices: list[int] = []

        for idx, token in enumerate(tokens):
            digit = self._digit_from_token(token)
            if digit is None:
                continue

            positions.append(valid_positions[idx].item())
            targets.append(digit)
            label_indices.append(len(targets) - 1)

            if len(targets) == LABEL_COUNT:
                break

        if len(targets) != LABEL_COUNT and not self._warned_label_count:
            self._warned_label_count = True
            logger.warning_rank0(
                "Expected %d label tokens but found %d. Classification loss may be misaligned.",
                LABEL_COUNT,
                len(targets),
            )

        return positions, targets, label_indices

    @staticmethod
    def _digit_from_token(token: str) -> Optional[int]:
        if "0" in token and "1" in token:
            return None
        if "1" in token:
            return 1
        if "0" in token:
            return 0
        return None

    def _gather_classification_tensors(
        self, labels: torch.Tensor, logits: torch.Tensor, tokenizer: "PreTrainedTokenizer"
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        samples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

        for batch_idx in range(labels.size(0)):
            positions, targets, label_indices = self._extract_label_tokens(labels[batch_idx], tokenizer)
            if not positions:
                continue

            sample_logits = logits[batch_idx, positions, self._one_token_id]
            target_tensor = torch.tensor(targets, device=logits.device, dtype=logits.dtype)
            index_tensor = torch.tensor(label_indices, device=logits.device, dtype=torch.long)
            samples.append((sample_logits, target_tensor, index_tensor))

        return samples

    def _compute_dice_loss(
        self, samples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    ) -> Optional[torch.Tensor]:
        if not samples:
            return None

        leaf_indices = torch.tensor(LEAF_LABEL_INDICES, device=samples[0][0].device)
        losses: list[torch.Tensor] = []

        for logits, targets, indices in samples:
            mask = (indices.unsqueeze(-1) == leaf_indices).any(dim=-1)
            if not mask.any():
                continue

            probs = torch.sigmoid(logits[mask])
            t = targets[mask]
            loss = 1 - (2 * (probs * t).sum() + 1) / (probs.sum() + t.sum() + 1)
            losses.append(loss)

        if not losses:
            return None

        return torch.stack(losses).mean()

    def _compute_hierarchical_loss(
        self, samples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    ) -> Optional[torch.Tensor]:
        if not samples:
            return None

        edges = PARENT_CHILD_EDGES
        all_edges: list[torch.Tensor] = []

        for logits, _, indices in samples:
            probs = torch.full((LABEL_COUNT,), float("nan"), device=logits.device, dtype=logits.dtype)
            probs[indices] = torch.sigmoid(logits)

            penalties: list[torch.Tensor] = []
            for parent, child in edges:
                parent_prob = probs[parent]
                child_prob = probs[child]
                if torch.isnan(parent_prob) or torch.isnan(child_prob):
                    continue
                penalties.append(torch.clamp_min(child_prob - parent_prob, 0.0))

            if penalties:
                all_edges.append(torch.stack(penalties).sum())

        if not all_edges:
            return None

        return torch.stack(all_edges).mean()

    @staticmethod
    def _get_logits(outputs: Any) -> Optional[torch.Tensor]:
        if outputs is None:
            return None
        if isinstance(outputs, dict):
            return outputs.get("logits")
        return getattr(outputs, "logits", None)

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
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
