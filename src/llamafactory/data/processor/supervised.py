# Copyright 2025 the LlamaFactory team.
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
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from .processor_utils import DatasetProcessor, greedy_knapsack, infer_seqlen


if TYPE_CHECKING:
    from ..mm_plugin import AudioInput, ImageInput, VideoInput


logger = logging.get_logger(__name__)


@dataclass
class SupervisedDatasetProcessor(DatasetProcessor):
    def __post_init__(self):
        self._label_list = None
        self._num_labels = None
        self._ml_stats = None

    def _load_label_space(self):
        if self._label_list is not None:
            return

        assets_dir = Path(__file__).resolve().parents[4] / "assets"
        labels_path = assets_dir / "labels_file.json"
        with open(labels_path, "r", encoding="utf-8") as f:
            label_mapping = json.load(f)

        label_list = [None] * len(label_mapping)
        for name, idx in label_mapping.items():
            label_list[idx] = name

        self._label_list = label_list
        self._num_labels = len(label_list)

    def _encode_data_example(
        self,
        prompt: list[dict[str, str]],
        response: list[dict[str, str]],
        system: Optional[str],
        tools: Optional[str],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
    ) -> tuple[list[int], list[int]]:
        messages = self.template.mm_plugin.process_messages(prompt + response, images, videos, audios, self.processor)
        input_ids, labels = self.template.mm_plugin.process_token_ids(
            [], [], images, videos, audios, self.tokenizer, self.processor
        )
        encoded_pairs = self.template.encode_multiturn(self.tokenizer, messages, system, tools)
        total_length = len(input_ids) + (1 if self.template.efficient_eos else 0)
        if self.data_args.mask_history:
            encoded_pairs = encoded_pairs[::-1]  # high priority for last turns

        for turn_idx, (source_ids, target_ids) in enumerate(encoded_pairs):
            if total_length >= self.data_args.cutoff_len:
                break

            source_len, target_len = infer_seqlen(
                len(source_ids), len(target_ids), self.data_args.cutoff_len - total_length
            )
            source_ids = source_ids[:source_len]
            target_ids = target_ids[:target_len]
            total_length += source_len + target_len

            if self.data_args.train_on_prompt:
                source_label = source_ids
            elif self.template.efficient_eos and turn_idx != 0:
                source_label = [self.tokenizer.eos_token_id] + [IGNORE_INDEX] * (source_len - 1)
            else:
                source_label = [IGNORE_INDEX] * source_len

            if self.data_args.mask_history and turn_idx != 0:  # train on the last turn only
                target_label = [IGNORE_INDEX] * target_len
            else:
                target_label = target_ids

            if self.data_args.mask_history:  # reversed sequences
                input_ids = source_ids + target_ids + input_ids
                labels = source_label + target_label + labels
            else:
                input_ids += source_ids + target_ids
                labels += source_label + target_label

        if self.template.efficient_eos:
            input_ids += [self.tokenizer.eos_token_id]
            labels += [self.tokenizer.eos_token_id]

        return input_ids, labels

    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        if getattr(self.data_args, "task_type", None) == "multi_label_sft_logits":
            return self._preprocess_multi_label_dataset(examples)

        # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
        # for multiturn examples, we only mask the prompt part in each prompt-response pair.
        model_inputs = defaultdict(list)
        for i in range(len(examples["_prompt"])):
            if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1:
                logger.warning_rank0(
                    "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
                )
                continue

            input_ids, labels = self._encode_data_example(
                prompt=examples["_prompt"][i],
                response=examples["_response"][i],
                system=examples["_system"][i],
                tools=examples["_tools"][i],
                images=examples["_images"][i] or [],
                videos=examples["_videos"][i] or [],
                audios=examples["_audios"][i] or [],
            )
            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(labels)
            model_inputs["images"].append(examples["_images"][i])
            model_inputs["videos"].append(examples["_videos"][i])
            model_inputs["audios"].append(examples["_audios"][i])

        return model_inputs

    def print_data_example(self, example: dict[str, list[int]]) -> None:
        valid_labels = list(filter(lambda x: x != IGNORE_INDEX, example["labels"]))
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(self.tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
        print("label_ids:\n{}".format(example["labels"]))
        print(f"labels:\n{self.tokenizer.decode(valid_labels, skip_special_tokens=False)}")

    def _preprocess_multi_label_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        self._load_label_space()

        if self._ml_stats is None:
            self._ml_stats = defaultdict(int)

        yes_id = self.tokenizer.convert_tokens_to_ids("<yes>")
        no_id = self.tokenizer.convert_tokens_to_ids("<no>")
        if yes_id == self.tokenizer.unk_token_id or no_id == self.tokenizer.unk_token_id:
            raise ValueError("<yes>/<no> tokens must be added to the tokenizer before preprocessing.")

        model_inputs = defaultdict(list)
        pattern = re.compile(r"\{([^{}]+)\}\s*(<yes>|<no>)")
        for i in range(len(examples.get("_prompt", []))):
            self._ml_stats["total"] += 1
            instruction = (examples.get("_ml_instruction", [""])[i] or "").strip()
            query = (examples.get("_ml_input", [""])[i] or "").strip()
            output = (examples.get("_ml_output", [""])[i] or "").strip()

            full_text = f"{instruction}\n\n{query}\n\n{output}"
            prompt_text = f"{instruction}\n\n{query}\n\n"

            parsed = pattern.findall(output)
            if len(parsed) != self._num_labels:
                logger.warning_rank0(
                    f"Skipped example {i} due to mismatched parsed label count: {len(parsed)} vs {self._num_labels}."
                )
                self._ml_stats["parsed_mismatch"] += 1
                continue

            parsed_labels, parsed_tags = zip(*parsed)
            if list(parsed_labels) != self._label_list:
                logger.warning_rank0(
                    f"Skipped example {i} due to label order mismatch with global label list."
                )
                self._ml_stats["label_order"] += 1
                continue

            binary_targets = [1.0 if tag == "<yes>" else 0.0 for tag in parsed_tags]

            tokenized = self.tokenizer(full_text, add_special_tokens=False)
            input_ids = tokenized["input_ids"]
            label_positions = [idx for idx, tid in enumerate(input_ids) if tid in (yes_id, no_id)]

            if len(label_positions) != self._num_labels:
                logger.warning_rank0(
                    f"Skipped example {i} due to mismatched label token count: {len(label_positions)} vs {self._num_labels}."
                )
                self._ml_stats["label_token_mismatch"] += 1
                continue

            if self.tokenizer.eos_token_id is not None:
                input_ids = input_ids + [self.tokenizer.eos_token_id]

            if len(input_ids) > self.data_args.cutoff_len:
                logger.warning_rank0(
                    f"Dropped lengthy example {i} with length {len(input_ids)} > {self.data_args.cutoff_len}."
                )
                self._ml_stats["too_long"] += 1
                continue

            labels = input_ids[:]
            attention_mask = [1] * len(input_ids)

            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append(attention_mask)
            model_inputs["labels"].append(labels)
            model_inputs["label_positions"].append(label_positions)
            model_inputs["binary_targets"].append(binary_targets)
            model_inputs["full_text"].append(full_text)
            model_inputs["prompt_text"].append(prompt_text)

            self._ml_stats["kept"] += 1

        return model_inputs


@dataclass
class PackedSupervisedDatasetProcessor(SupervisedDatasetProcessor):
    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        # TODO: use `position_ids` to achieve packing
        # build inputs with format `<bos> X1 Y1 <eos> <bos> X2 Y2 <eos>`
        # and labels with format `<ignore> ... <ignore> Y1 <eos> <ignore> ... <ignore> Y2 <eos>`
        valid_num = 0
        batch_input_ids, batch_labels, batch_images, batch_videos, batch_audios = [], [], [], [], []
        lengths = []
        length2indexes = defaultdict(list)
        for i in range(len(examples["_prompt"])):
            if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1:
                logger.warning_rank0(
                    "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
                )
                continue

            input_ids, labels = self._encode_data_example(
                prompt=examples["_prompt"][i],
                response=examples["_response"][i],
                system=examples["_system"][i],
                tools=examples["_tools"][i],
                images=examples["_images"][i] or [],
                videos=examples["_videos"][i] or [],
                audios=examples["_audios"][i] or [],
            )
            length = len(input_ids)
            if length > self.data_args.cutoff_len:
                logger.warning_rank0(f"Dropped lengthy example with length {length} > {self.data_args.cutoff_len}.")
            else:
                lengths.append(length)
                length2indexes[length].append(valid_num)
                batch_input_ids.append(input_ids)
                batch_labels.append(labels)
                batch_images.append(examples["_images"][i] or [])
                batch_videos.append(examples["_videos"][i] or [])
                batch_audios.append(examples["_audios"][i] or [])
                valid_num += 1

        model_inputs = defaultdict(list)
        knapsacks = greedy_knapsack(lengths, self.data_args.cutoff_len)
        for knapsack in knapsacks:
            packed_input_ids, packed_attention_masks, packed_position_ids, packed_labels = [], [], [], []
            packed_images, packed_videos, packed_audios = [], [], []
            for i, length in enumerate(knapsack):
                index = length2indexes[length].pop()
                packed_input_ids += batch_input_ids[index]
                packed_position_ids += list(range(len(batch_input_ids[index])))  # NOTE: pad_to_multiple_of ignore this
                packed_labels += batch_labels[index]
                packed_images += batch_images[index]
                packed_videos += batch_videos[index]
                packed_audios += batch_audios[index]
                if self.data_args.neat_packing:
                    packed_attention_masks += [i + 1] * len(batch_input_ids[index])  # start from 1
                else:
                    packed_attention_masks += [1] * len(batch_input_ids[index])

            if len(packed_input_ids) < self.data_args.cutoff_len + 1:  # avoid flash_attn drops attn mask
                pad_length = self.data_args.cutoff_len - len(packed_input_ids) + 1
                packed_input_ids += [self.tokenizer.pad_token_id] * pad_length
                packed_position_ids += [0] * pad_length
                packed_labels += [IGNORE_INDEX] * pad_length
                if self.data_args.neat_packing:
                    packed_attention_masks += [0] * pad_length
                else:
                    packed_attention_masks += [1] * pad_length  # more efficient flash_attn

            if len(packed_input_ids) != self.data_args.cutoff_len + 1:
                raise ValueError("The length of packed example should be identical to the cutoff length.")

            model_inputs["input_ids"].append(packed_input_ids)
            model_inputs["attention_mask"].append(packed_attention_masks)
            model_inputs["position_ids"].append(packed_position_ids)
            model_inputs["labels"].append(packed_labels)
            model_inputs["images"].append(packed_images or None)
            model_inputs["videos"].append(packed_videos or None)
            model_inputs["audios"].append(packed_audios or None)

        return model_inputs
