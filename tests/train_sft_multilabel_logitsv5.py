# -*- coding: utf-8 -*-
import os
import re
import json
import math
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F

from datasets import load_dataset
from sklearn.metrics import f1_score, hamming_loss

import matplotlib.pyplot as plt

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    TrainerCallback,
)

from peft import LoraConfig, get_peft_model


# ---------------------------
# Utils: load label order & parent-child
# ---------------------------

def load_labels_file(labels_file: str) -> Tuple[List[str], Dict[str, int]]:
    """
    labels_file.json: { "{LabelA}": 0, "{LabelB}": 1, ... }
    We need an ordered list of labels by index.
    """
    with open(labels_file, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    inv = {int(v): k for k, v in mapping.items()}
    num_labels = len(inv)
    labels = [inv[i] for i in range(num_labels)]
    return labels, mapping


def load_parent_child_pairs(pairs_file: Optional[str]) -> List[Tuple[int, int]]:
    if not pairs_file:
        return []
    with open(pairs_file, "r", encoding="utf-8") as f:
        pairs = json.load(f)
    out = []
    for x in pairs:
        if isinstance(x, (list, tuple)) and len(x) == 2:
            out.append((int(x[0]), int(x[1])))
        elif isinstance(x, dict):
            out.append((int(x["parent"]), int(x["child"])))
        else:
            raise ValueError(f"Bad parent_child_pairs item: {x}")
    return out


# ---------------------------
# Parse output "{Label}<yes>/<no>..."
# ---------------------------

YES_TOKEN = "<yes>"
NO_TOKEN = "<no>"

_label_tag_pat = re.compile(r"(\{[^{}]+\})(<yes>|<no>)")


def parse_output_to_tags(output_text: str, label_order: List[str]) -> List[str]:
    """
    Return list of "<yes>/<no>" in the same order as label_order.
    We enforce exact order by scanning matches and comparing labels.
    """
    matches = _label_tag_pat.findall(output_text)
    if len(matches) != len(label_order):
        raise ValueError(f"Output tags count mismatch: got {len(matches)} expected {len(label_order)}")

    tags = []
    for i, (lab, tag) in enumerate(matches):
        if lab != label_order[i]:
            raise ValueError(f"Label order mismatch at {i}: got {lab}, expected {label_order[i]}")
        tags.append(tag)
    return tags


def tags_to_multihot(tags: List[str]) -> np.ndarray:
    arr = np.zeros(len(tags), dtype=np.float32)
    for i, t in enumerate(tags):
        arr[i] = 1.0 if t == YES_TOKEN else 0.0
    return arr


# ---------------------------
# Losses
# ---------------------------

def bce_with_logits_loss(binary_logits: torch.Tensor,
                         targets: torch.Tensor,
                         pos_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    # binary_logits/targets: [B, C]
    if pos_weight is not None:
        return F.binary_cross_entropy_with_logits(binary_logits, targets, pos_weight=pos_weight)
    return F.binary_cross_entropy_with_logits(binary_logits, targets)


def dice_loss(binary_logits: torch.Tensor,
              targets: torch.Tensor,
              smooth: float = 1e-6) -> torch.Tensor:
    # multi-label dice on probs
    probs = torch.sigmoid(binary_logits)  # [B,C]
    inter = (probs * targets).sum(dim=-1)
    union = probs.sum(dim=-1) + targets.sum(dim=-1)
    dice = (2 * inter + smooth) / (union + smooth)
    return 1.0 - dice.mean()


def hierarchical_loss(binary_logits: torch.Tensor,
                      parent_child_pairs: List[Tuple[int, int]]) -> torch.Tensor:
    """
    Simple constraint: P(child) <= P(parent). penalty = ReLU(P_child - P_parent).
    """
    if not parent_child_pairs:
        return binary_logits.new_tensor(0.0)
    probs = torch.sigmoid(binary_logits)  # [B,C]
    penalties = []
    for p, c in parent_child_pairs:
        penalties.append(F.relu(probs[:, c] - probs[:, p]))
    return torch.stack(penalties, dim=0).mean()


def hierarchical_violation_rate(preds01: np.ndarray,
                                parent_child_pairs: List[Tuple[int, int]]) -> float:
    if preds01.size == 0 or not parent_child_pairs:
        return 0.0
    violations = 0
    total = preds01.shape[0] * len(parent_child_pairs)
    for p, c in parent_child_pairs:
        violations += int(((preds01[:, c] == 1) & (preds01[:, p] == 0)).sum())
    return float(violations / max(total, 1))


# ---------------------------
# pos_weight
# ---------------------------

def compute_pos_weight_from_targets(all_targets: np.ndarray, clip_max: float) -> torch.Tensor:
    """
    all_targets: [N,C] float32 0/1
    pos_weight = neg/pos, clipped
    """
    N, C = all_targets.shape
    pos = all_targets.sum(axis=0)  # [C]
    neg = N - pos
    raw = neg / (pos + 1e-8)
    raw = np.clip(raw, 0.0, clip_max)
    return torch.tensor(raw.astype(np.float32))


# ---------------------------
# Dataset preprocessing
# ---------------------------

def build_full_text(instruction: str, inp: str, out: str) -> str:
    # Important: must match the training-time tokenization used to compute label_positions.
    return f"{instruction}\n\n{inp}\n\n{out}"


def build_prompt_only(instruction: str, inp: str) -> str:
    return f"{instruction}\n\n{inp}\n\n"  # end with separator, no output


@dataclass
class MultiLabelFeature:
    input_ids: List[int]
    attention_mask: List[int]
    labels: List[int]  # LM labels, IGNORE_INDEX for prompt tokens
    label_positions: List[int]  # positions (absolute) in input_ids where <yes>/<no> appear (ONLY IN OUTPUT SEGMENT)
    binary_targets: List[float]  # [C]


IGNORE_INDEX = -100


class MultiLabelDataCollator:
    """
    Pad input_ids/attention_mask/labels; also keep label_positions length=C (num_labels),
    and binary_targets to tensor [B,C].
    """
    def __init__(self, tokenizer: AutoTokenizer, num_labels: int):
        self.tokenizer = tokenizer
        self.num_labels = num_labels

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)

        input_ids, attn, lm_labels = [], [], []
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.tokenizer.pad_token_id] * pad_len)
            attn.append(f["attention_mask"] + [0] * pad_len)
            lm_labels.append(f["labels"] + [IGNORE_INDEX] * pad_len)

        label_pos, bin_tgt = [], []
        for f in features:
            lp = f["label_positions"]
            if len(lp) != self.num_labels:
                raise ValueError(f"label_positions length {len(lp)} != num_labels {self.num_labels}")
            label_pos.append(lp)

            bt = f["binary_targets"]
            if len(bt) != self.num_labels:
                raise ValueError(f"binary_targets length {len(bt)} != num_labels {self.num_labels}")
            bin_tgt.append(bt)

        batch = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(lm_labels, dtype=torch.long),
            "label_positions": torch.tensor(label_pos, dtype=torch.long),
            "binary_targets": torch.tensor(bin_tgt, dtype=torch.float32),
        }
        return batch


# ---------------------------
# Metrics
# ---------------------------

def compute_metrics_from_preds(preds01: np.ndarray, targets01: np.ndarray) -> Dict[str, float]:
    micro = f1_score(targets01, preds01, average="micro", zero_division=0)
    macro = f1_score(targets01, preds01, average="macro", zero_division=0)
    exact = float((preds01 == targets01).all(axis=1).mean()) if preds01.size else 0.0
    hml = hamming_loss(targets01, preds01) if preds01.size else 0.0
    return {"micro_f1": float(micro), "macro_f1": float(macro), "exact_match": exact, "hamming_loss": float(hml)}


# ---------------------------
# Training curve callback (save png)
# ---------------------------

class TrainingCurveCallback(TrainerCallback):
    """
    Collect loss logs and save training_curves.png at the end of training.
    Curves: loss_sft / loss_bce / loss_dice / loss_hier / loss (total)
    """
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.steps = []
        self.series = {
            "loss_sft": [],
            "loss_bce": [],
            "loss_dice": [],
            "loss_hier": [],
            "loss": [],
        }

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        if "loss" not in logs and "loss_sft" not in logs:
            return

        step = int(state.global_step)
        self.steps.append(step)
        for k in self.series.keys():
            self.series[k].append(float(logs.get(k, np.nan)))

    def on_train_end(self, args, state, control, **kwargs):
        if not self.steps:
            return

        os.makedirs(self.output_dir, exist_ok=True)
        out_path = os.path.join(self.output_dir, "training_curves.png")

        plt.figure()
        for k, vals in self.series.items():
            if all(np.isnan(v) for v in vals):
                continue
            plt.plot(self.steps, vals, label=k)

        plt.xlabel("global_step")
        plt.ylabel("loss")
        plt.title("Training Curves")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"[INFO] Saved training curves to: {out_path}")


# ---------------------------
# Trainer
# ---------------------------

class MultiLabelSFTTrainer(Trainer):
    """
    - Train: always teacher forcing (standard SFT) + optional classification losses (BCE/Dice/Hier)
    - Eval/Predict: controlled by use_teacher_forcing_logits (ONLY affects eval/test)
    - Avoid returning huge [B,T,V] logits in eval loop.
    - Support automatic global threshold search using threshold_indicator.
    """
    def __init__(self,
                 num_labels: int,
                 yes_id: int,
                 no_id: int,
                 parent_child_pairs: List[Tuple[int, int]],
                 use_teacher_forcing_logits: bool,  # ONLY for eval/test
                 use_bce: bool,
                 use_dice: bool,
                 use_hier: bool,
                 lambda_sft: float,
                 lambda_bce: float,
                 lambda_dice: float,
                 lambda_hier: float,
                 pos_weight: Optional[torch.Tensor],
                 fixed_threshold: float,
                 threshold_indicator: Optional[str],
                 threshold_grid_step: float = 0.01,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_labels = num_labels
        self.yes_id = yes_id
        self.no_id = no_id
        self.parent_child_pairs = parent_child_pairs

        # IMPORTANT: only affects eval/test in prediction_step + compute_metrics
        self.use_teacher_forcing_logits = use_teacher_forcing_logits

        self.use_bce = use_bce
        self.use_dice = use_dice
        self.use_hier = use_hier

        self.lambda_sft = lambda_sft
        self.lambda_bce = lambda_bce
        self.lambda_dice = lambda_dice
        self.lambda_hier = lambda_hier

        self.pos_weight = pos_weight
        self.fixed_threshold = fixed_threshold

        # If None => fixed threshold; if set => search best threshold by this indicator
        self.threshold_indicator = threshold_indicator
        self.threshold_grid_step = threshold_grid_step

    def _extract_binary_logits_from_full_logits(self,
                                                logits: torch.Tensor,
                                                label_positions: torch.Tensor) -> torch.Tensor:
        """
        logits: [B,T,V]
        label_positions: [B,C] positions in T
        Return binary_logits: [B,C] = logit(yes)-logit(no)
        Note: gather directly to avoid [B,C,V].
        """
        B, T, _ = logits.shape
        b_idx = torch.arange(B, device=logits.device).unsqueeze(1).expand(B, self.num_labels)  # [B,C]
        pos = label_positions.clamp(0, T - 1)
        yes = logits[b_idx, pos, self.yes_id]  # [B,C]
        no = logits[b_idx, pos, self.no_id]    # [B,C]
        return yes - no

    def _compute_cls_losses(self, binary_logits: torch.Tensor, binary_targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute bce/dice/hier and combined loss_cls (already weighted by lambdas).
        """
        loss_bce = binary_logits.new_tensor(0.0)
        loss_dice = binary_logits.new_tensor(0.0)
        loss_hier = binary_logits.new_tensor(0.0)
        loss_cls = binary_logits.new_tensor(0.0)

        if self.use_bce:
            pw = self.pos_weight.to(binary_logits.device) if self.pos_weight is not None else None
            loss_bce = bce_with_logits_loss(binary_logits, binary_targets, pw)
            loss_cls = loss_cls + self.lambda_bce * loss_bce
        if self.use_dice:
            loss_dice = dice_loss(binary_logits, binary_targets)
            loss_cls = loss_cls + self.lambda_dice * loss_dice
        if self.use_hier:
            loss_hier = hierarchical_loss(binary_logits, self.parent_child_pairs)
            loss_cls = loss_cls + self.lambda_hier * loss_hier

        return {
            "loss_bce": loss_bce,
            "loss_dice": loss_dice,
            "loss_hier": loss_hier,
            "loss_cls": loss_cls,
        }

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        """
        TRAINING STAGE:
        - Always teacher forcing (standard SFT).
        - Classification losses do NOT depend on use_teacher_forcing_logits (that flag is eval/test only).
        """
        label_positions = inputs.pop("label_positions")
        binary_targets = inputs.pop("binary_targets")  # [B,C]

        outputs = model(**inputs)  # includes loss & logits
        loss_sft = outputs.loss

        loss_cls = loss_sft.new_tensor(0.0)
        loss_bce = loss_sft.new_tensor(0.0)
        loss_dice = loss_sft.new_tensor(0.0)
        loss_hier = loss_sft.new_tensor(0.0)

        if self.use_bce or self.use_dice or self.use_hier:
            binary_logits = self._extract_binary_logits_from_full_logits(outputs.logits, label_positions).float()
            binary_targets_t = binary_targets.to(binary_logits.device).float()
            cls = self._compute_cls_losses(binary_logits, binary_targets_t)
            loss_bce, loss_dice, loss_hier, loss_cls = cls["loss_bce"], cls["loss_dice"], cls["loss_hier"], cls["loss_cls"]

        loss = self.lambda_sft * loss_sft + loss_cls

        # log for curves (every logging_steps)
        log_steps = int(getattr(self.args, "logging_steps", 0) or 0)
        if log_steps > 0 and ((int(self.state.global_step) + 1) % log_steps == 0):
            self.log({
                "loss_sft": float(loss_sft.detach().cpu()),
                "loss_bce": float(loss_bce.detach().cpu()),
                "loss_dice": float(loss_dice.detach().cpu()),
                "loss_hier": float(loss_hier.detach().cpu()),
                "loss_cls": float(loss_cls.detach().cpu()),
                "loss": float(loss.detach().cpu()),
            })

        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        EVAL/TEST STAGE:
        - If use_teacher_forcing_logits=False => no classification metrics here (return loss only).
        - If True => return small [B,C] binary logits + targets to compute metrics/threshold search.
        """
        label_positions = inputs.get("label_positions", None)
        binary_targets = inputs.get("binary_targets", None)

        # If not teacher forcing logits in eval/test, we keep evaluation stable and fast.
        if not self.use_teacher_forcing_logits:
            with torch.no_grad():
                outputs = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    labels=inputs.get("labels", None),
                )
                loss = outputs.loss.detach() if outputs.loss is not None else None

            if prediction_loss_only:
                return (loss, None, None)
            return (loss, None, None)

        # Teacher-forcing eval/test: compute loss + binary logits
        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=inputs.get("labels", None),
            )
            loss = outputs.loss.detach() if outputs.loss is not None else None

            binary_logits = self._extract_binary_logits_from_full_logits(outputs.logits, label_positions)
            binary_logits = binary_logits.detach().float().clone().cpu()

            targets = binary_targets.detach().float().clone().cpu() if binary_targets is not None else None

        if prediction_loss_only:
            return (loss, None, None)
        return (loss, binary_logits, targets)

    def _threshold_to_logit_offset(self, t: float) -> float:
        eps = 1e-6
        t = float(np.clip(t, eps, 1.0 - eps))
        return float(math.log(t / (1.0 - t)))

    def _search_best_threshold(self, preds: np.ndarray, labels: np.ndarray) -> Tuple[float, Dict[str, float]]:
        indicator = self.threshold_indicator
        if indicator is None:
            t = self.fixed_threshold
            probs = 1.0 / (1.0 + np.exp(-preds))
            preds01 = (probs >= t).astype(np.int32)
            labels01 = (labels >= 0.5).astype(np.int32)
            out = compute_metrics_from_preds(preds01, labels01)
            out["hier_violation_rate"] = hierarchical_violation_rate(preds01, self.parent_child_pairs)
            out["best_threshold"] = float(t)
            out["threshold_indicator"] = "fixed"
            return float(t), out

        step = float(self.threshold_grid_step)
        grid = np.arange(step, 1.0, step, dtype=np.float32)

        preds_t = torch.tensor(preds, dtype=torch.float32)
        labels_t = torch.tensor(labels, dtype=torch.float32)
        pw = self.pos_weight.float() if self.pos_weight is not None else None

        best_t = None
        best_score = None
        best_metrics = None

        higher_better = indicator in ("micro_f1", "macro_f1", "exact_match")
        lower_better = indicator in ("ce_loss", "cls_loss", "hamming_loss")

        if not (higher_better or lower_better):
            raise ValueError(
                f"Unsupported threshold_indicator={indicator}. "
                f"Choose from: ce_loss, cls_loss, micro_f1, macro_f1, exact_match, hamming_loss"
            )

        for t in grid:
            offset = self._threshold_to_logit_offset(float(t))
            shifted = preds_t - offset

            probs = torch.sigmoid(preds_t)
            preds01 = (probs >= float(t)).to(torch.int32).cpu().numpy()
            labels01 = (labels_t >= 0.5).to(torch.int32).cpu().numpy()

            m = compute_metrics_from_preds(preds01, labels01)
            m["hier_violation_rate"] = hierarchical_violation_rate(preds01, self.parent_child_pairs)

            if indicator == "ce_loss":
                loss_bce = bce_with_logits_loss(shifted, labels_t, pw).item()
                score = float(loss_bce)
                m["ce_loss_at_threshold"] = float(loss_bce)
            elif indicator == "cls_loss":
                # same as training cls loss definition
                loss_cls = 0.0
                if self.use_bce:
                    loss_cls += float(self.lambda_bce) * float(bce_with_logits_loss(shifted, labels_t, pw).item())
                if self.use_dice:
                    loss_cls += float(self.lambda_dice) * float(dice_loss(shifted, labels_t).item())
                if self.use_hier:
                    loss_cls += float(self.lambda_hier) * float(hierarchical_loss(shifted, self.parent_child_pairs).item())
                score = float(loss_cls)
                m["cls_loss_at_threshold"] = float(score)
            elif indicator in ("micro_f1", "macro_f1", "exact_match"):
                score = float(m[indicator])
            elif indicator == "hamming_loss":
                score = float(m["hamming_loss"])
            else:
                raise RuntimeError("unreachable")

            if best_t is None:
                best_t = float(t)
                best_score = score
                best_metrics = m
            else:
                if higher_better:
                    if score > best_score:
                        best_t, best_score, best_metrics = float(t), score, m
                else:
                    if score < best_score:
                        best_t, best_score, best_metrics = float(t), score, m

        assert best_t is not None and best_metrics is not None
        best_metrics["best_threshold"] = float(best_t)
        best_metrics["threshold_indicator"] = str(indicator)
        best_metrics["threshold_best_score"] = float(best_score)
        return float(best_t), best_metrics

    def compute_metrics(self, eval_pred):
        preds = eval_pred.predictions
        labels = eval_pred.label_ids
        if preds is None or labels is None:
            return {}
        _, out = self._search_best_threshold(preds, labels)
        return out


# ---------------------------
# Generation-based prediction (offline)
# ---------------------------

def generate_and_parse(
    model,
    tokenizer,
    instruction: str,
    inp: str,
    label_order: List[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
) -> np.ndarray:
    """
    Generate from prompt-only, parse <yes>/<no> tags in order.
    Returns multi-hot [C].
    """
    prompt = build_prompt_only(instruction, inp)
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    enc = {k: v.to(model.device) for k, v in enc.items()}

    if do_sample:
        gen_do_sample = True
        gen_temperature = temperature if temperature > 0 else 1.0
        gen_top_p = top_p
    else:
        # greedy decoding
        gen_do_sample = False
        gen_temperature = 1.0
        gen_top_p = 1.0

    gen = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=gen_do_sample,
        temperature=gen_temperature,
        top_p=gen_top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # gen = model.generate(
    #     **enc,
    #     max_new_tokens=max_new_tokens,
    #     do_sample=(temperature > 0),
    #     temperature=temperature if temperature > 0 else 1.0,
    #     top_p=top_p,
    #     pad_token_id=tokenizer.pad_token_id,
    #     eos_token_id=tokenizer.eos_token_id,
    # )

    text = tokenizer.decode(gen[0], skip_special_tokens=False)

    matches = _label_tag_pat.findall(text)
    C = len(label_order)
    if len(matches) < C:
        return np.zeros(C, dtype=np.float32)

    tail = matches[-C:]
    tags = []
    for _, tag in tail:
        tags.append(tag)
    return tags_to_multihot(tags)


def generate_and_parse_with_logits(
    model,
    tokenizer,
    instruction: str,
    inp: str,
    label_order: List[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
    yes_id: int,
    no_id: int,
    num_labels: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    (1) Keep original text-parse behavior (same extraction approach as generate_and_parse)
    (2) Additionally extract binary logits from generation scores using the specified heuristic.
    Returns:
      - multi_hot_text: [C] float32
      - binary_logits: [C] float32
    """
    prompt = build_prompt_only(instruction, inp)
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    enc = {k: v.to(model.device) for k, v in enc.items()}

    if do_sample:
        gen_do_sample = True
        gen_temperature = temperature if temperature > 0 else 1.0
        gen_top_p = top_p
    else:
        # greedy decoding
        gen_do_sample = False
        gen_temperature = 1.0
        gen_top_p = 1.0

    outputs = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=gen_do_sample,
        temperature=gen_temperature,
        top_p=gen_top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        output_scores=True,
        return_dict_in_generate=True,
    )

    # outputs = model.generate(
    #     **enc,
    #     max_new_tokens=max_new_tokens,
    #     do_sample=(temperature > 0),
    #     temperature=temperature if temperature > 0 else 1.0,
    #     top_p=top_p,
    #     pad_token_id=tokenizer.pad_token_id,
    #     eos_token_id=tokenizer.eos_token_id,
    #     output_scores=True,
    #     return_dict_in_generate=True,
    # )

    # text-parse path (same behavior)
    text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=False)
    matches = _label_tag_pat.findall(text)
    C = len(label_order)
    if len(matches) < C:
        multi_hot_text = np.zeros(C, dtype=np.float32)
    else:
        tail = matches[-C:]
        tags = [tag for _, tag in tail]
        multi_hot_text = tags_to_multihot(tags)

    # logit-extract path
    binary_logits_list: List[float] = []
    for step_logits in outputs.scores:
        step_logits = step_logits[0]  # [V]
        top2 = torch.topk(step_logits, k=2).indices.tolist()
        if (yes_id in top2) or (no_id in top2):
            logit_yes = float(step_logits[yes_id].item())
            logit_no = float(step_logits[no_id].item())
            binary_logits_list.append(logit_yes - logit_no)

    if len(binary_logits_list) < num_labels:
        binary_logits_list.extend([0.0] * (num_labels - len(binary_logits_list)))
    elif len(binary_logits_list) > num_labels:
        binary_logits_list = binary_logits_list[-num_labels:]

    binary_logits = np.array(binary_logits_list, dtype=np.float32)
    return multi_hot_text, binary_logits


def _threshold_to_logit_offset(t: float) -> float:
    eps = 1e-6
    t = float(np.clip(t, eps, 1.0 - eps))
    return float(math.log(t / (1.0 - t)))


def search_best_threshold_for_logits(
    binary_logits: np.ndarray,        # [N,C]
    labels: np.ndarray,              # [N,C] in {0,1}
    parent_child_pairs: List[Tuple[int, int]],
    threshold_indicator: Optional[str],
    fixed_threshold: float,
    threshold_grid: np.ndarray,      # e.g. np.arange(0.1, 0.91, 0.05)
    pos_weight: Optional[torch.Tensor],
    use_bce: bool,
    use_dice: bool,
    use_hier: bool,
    lambda_bce: float,
    lambda_dice: float,
    lambda_hier: float,
) -> Tuple[float, Dict[str, float]]:
    """
    Returns best_threshold, metrics_at_best_threshold
    If threshold_indicator is None -> use fixed threshold (no search)
    """
    if threshold_indicator is None:
        t = float(fixed_threshold)
        probs = 1.0 / (1.0 + np.exp(-binary_logits))
        preds01 = (probs >= t).astype(np.int32)
        labels01 = labels.astype(np.int32)
        m = compute_metrics_from_preds(preds01, labels01)
        m["hier_violation_rate"] = hierarchical_violation_rate(preds01, parent_child_pairs)
        m["best_threshold"] = float(t)
        m["threshold_indicator"] = "fixed"
        return float(t), m

    indicator = str(threshold_indicator)

    higher_better = indicator in ("micro_f1", "macro_f1", "exact_match")
    lower_better = indicator in ("ce_loss", "cls_loss", "hamming_loss")
    if not (higher_better or lower_better):
        raise ValueError(
            f"Unsupported threshold_indicator={indicator}. "
            f"Choose from: ce_loss, cls_loss, micro_f1, macro_f1, exact_match, hamming_loss"
        )

    preds_t = torch.tensor(binary_logits, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.float32)
    pw = pos_weight.float() if pos_weight is not None else None

    best_t = None
    best_score = None
    best_metrics: Optional[Dict[str, float]] = None 

    # hard preds for metrics (threshold on probs of original logits)
    probs_np = 1.0 / (1.0 + np.exp(-binary_logits))
    labels01 = labels.astype(np.int32)

    for t in threshold_grid:
        t = float(t)

        preds01 = (probs_np >= t).astype(np.int32)

        m = compute_metrics_from_preds(preds01, labels01)
        m["hier_violation_rate"] = hierarchical_violation_rate(preds01, parent_child_pairs)

        # indicator scoring
        if indicator == "ce_loss":
            offset = _threshold_to_logit_offset(t)
            shifted = preds_t - offset
            ce = float(bce_with_logits_loss(shifted, labels_t, pw).item())
            score = ce
            m["ce_loss_at_threshold"] = ce
        elif indicator == "cls_loss":
            offset = _threshold_to_logit_offset(t)
            shifted = preds_t - offset
            cls = 0.0
            if use_bce:
                cls += float(lambda_bce) * float(bce_with_logits_loss(shifted, labels_t, pw).item())
            if use_dice:
                cls += float(lambda_dice) * float(dice_loss(shifted, labels_t).item())
            if use_hier:
                cls += float(lambda_hier) * float(hierarchical_loss(shifted, parent_child_pairs).item())
            score = float(cls)
            m["cls_loss_at_threshold"] = float(cls)
        elif indicator in ("micro_f1", "macro_f1", "exact_match"):
            score = float(m[indicator])
        elif indicator == "hamming_loss":
            score = float(m["hamming_loss"])
        else:
            raise RuntimeError("unreachable")

        if best_t is None:
            best_t = t
            best_score = score
            best_metrics = m
        else:
            if higher_better:
                if score > best_score:
                    best_t, best_score, best_metrics = t, score, m
            else:
                if score < best_score:
                    best_t, best_score, best_metrics = t, score, m

    assert best_t is not None and best_metrics is not None
    best_metrics["best_threshold"] = float(best_t)
    best_metrics["threshold_indicator"] = str(indicator)
    best_metrics["threshold_best_score"] = float(best_score)
    return float(best_t), best_metrics


# ---------------------------
# Main
# ---------------------------

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--train_file", type=str, required=True)
    p.add_argument("--validation_file", type=str, required=True)
    p.add_argument("--test_file", type=str, default=None)
    p.add_argument("--labels_file", type=str, required=True)
    p.add_argument("--parent_child_pairs", type=str, default=None)

    p.add_argument("--output_dir", type=str, default="./out_multilabel_sft")
    p.add_argument("--max_length", type=int, default=2048)

    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--per_device_eval_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--num_train_epochs", type=float, default=2.0)
    p.add_argument("--learning_rate", type=float, default=4e-4)

    p.add_argument("--logging_steps", type=int, default=20)
    p.add_argument("--eval_steps", type=int, default=100)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--save_total_limit", type=int, default=2)

    # precision
    p.add_argument("--bf16", action="store_true", help="Use bf16 (else fp32).")

    # LoRA
    p.add_argument("--use_lora", action="store_true", help="Enable LoRA.")
    p.add_argument("--lora_r", type=int, default=64)
    p.add_argument("--lora_alpha", type=int, default=128)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj")

    # Eval/Test teacher forcing logits switch (ONLY eval/test)
    p.add_argument("--use_teacher_forcing_logits", action="store_true",
                   help="If set: eval/test compute multi-label metrics from teacher-forced logits. Train is ALWAYS teacher forcing.")

    # losses toggles
    p.add_argument("--use_bce_loss", action="store_true")
    p.add_argument("--use_dice_loss", action="store_true")
    p.add_argument("--use_hier_loss", action="store_true")

    p.add_argument("--lambda_sft", type=float, default=1.0)
    p.add_argument("--lambda_bce", type=float, default=1.0)
    p.add_argument("--lambda_dice", type=float, default=1.0)
    p.add_argument("--lambda_hier", type=float, default=1.0)

    # pos_weight
    p.add_argument("--use_pos_weight", action="store_true")
    p.add_argument("--pos_weight_clip_max", type=float, default=50.0)

    # fixed threshold (used when threshold_indicator is not provided)
    p.add_argument("--threshold", type=float, default=0.5)

    # threshold search:
    # - If user does NOT use this arg => fixed threshold (0.5 by default)
    # - If user uses "--threshold_indicator" without value => default "ce_loss"
    # - If user uses "--threshold_indicator micro_f1" etc => search by that indicator
    p.add_argument(
        "--threshold_indicator",
        nargs="?",
        const="ce_loss",
        default=None,
        type=str,
        help="If set, search best global threshold on eval by indicator. "
             "Allowed: ce_loss, cls_loss, micro_f1, macro_f1, exact_match, hamming_loss. "
             "If provided without value, defaults to ce_loss."
    )
    p.add_argument("--threshold_grid_step", type=float, default=0.01,
                   help="Threshold search step size, e.g. 0.01 -> search 0.01..0.99")

    # early stopping
    p.add_argument("--early_stopping_patience", type=int, default=5)
    p.add_argument("--early_stopping_threshold", type=float, default=0.0)

    # offline generation-based test (separate from Trainer loop)
    p.add_argument("--run_generate_test", action="store_true",
                   help="After training, run generate-based prediction on test set and compute metrics.")
    p.add_argument("--gen_max_new_tokens", type=int, default=512)
    p.add_argument("--gen_temperature", type=float, default=0.0)
    p.add_argument("--gen_top_p", type=float, default=0.9)
    p.add_argument(
        "--do_sample",
        action="store_true",
        help="If set, enable sampling in generation. If not set, use greedy decoding."
    )


    # NEW: optional generate-logit evaluation for run_generate_test
    p.add_argument("--rgt_eval_logits", action="store_true",
                   help="If set (default False): in run_generate_test, additionally evaluate by extracting <yes>/<no> logits from generation scores.")

    return p.parse_args()


def load_split(file_path: str):
    ext = os.path.splitext(file_path)[1].lower()
    if ext in [".json", ".jsonl"]:
        return load_dataset("json", data_files={"data": file_path})["data"]
    elif ext == ".csv":
        return load_dataset("csv", data_files={"data": file_path})["data"]
    else:
        raise ValueError(f"Unsupported file: {file_path}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    label_order, _ = load_labels_file(args.labels_file)
    parent_child_pairs = load_parent_child_pairs(args.parent_child_pairs)
    num_labels = len(label_order)
    print(f"[INFO] num_labels={num_labels}, parent_child_pairs={len(parent_child_pairs)}")


    # tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True, trust_remote_code=True)

    # # Ensure <yes>/<no> are special tokens for train/eval/test (single shared tokenizer)
    # special_added = tokenizer.add_special_tokens({"additional_special_tokens": [YES_TOKEN, NO_TOKEN]})
    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.eos_token

    # yes_id = tokenizer.convert_tokens_to_ids(YES_TOKEN)
    # no_id = tokenizer.convert_tokens_to_ids(NO_TOKEN)
    # if yes_id is None or yes_id < 0 or no_id is None or no_id < 0:
    #     raise RuntimeError("Failed to register <yes>/<no> special tokens.")
    # print(f"[INFO] yes_id={yes_id}, no_id={no_id}, special_added={special_added}")

    # # model
    # _ = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    # dtype = torch.bfloat16 if args.bf16 else torch.float32

    # model = AutoModelForCausalLM.from_pretrained(
    #     args.model_name_or_path,
    #     torch_dtype=dtype if args.bf16 else None,
    #     device_map="auto",
    #     trust_remote_code=True,
    # )

    # # resize embeddings if new tokens added
    # model.resize_token_embeddings(len(tokenizer))


    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True, trust_remote_code=True)

    # Ensure <yes>/<no> are special tokens for train/eval/test (single shared tokenizer)
    special_added = tokenizer.add_special_tokens({"additional_special_tokens": [YES_TOKEN, NO_TOKEN]})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    yes_id = tokenizer.convert_tokens_to_ids(YES_TOKEN)
    no_id = tokenizer.convert_tokens_to_ids(NO_TOKEN)
    if yes_id is None or yes_id < 0 or no_id is None or no_id < 0:
        raise RuntimeError("Failed to register <yes>/<no> special tokens.")
    print(f"[INFO] yes_id={yes_id}, no_id={no_id}, special_added={special_added}")

    # model
    _ = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    dtype = torch.bfloat16 if args.bf16 else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype if args.bf16 else None,
        device_map="auto",
        trust_remote_code=True,
    )

    # resize embeddings if new tokens added
    model.resize_token_embeddings(len(tokenizer))

    # copy "yes"/"no" embedding & lm_head rows into <yes>/<no>, and unfreeze ONLY those new rows
    def _get_single_token_id(_tok: Any, _text: str) -> Optional[int]:
        _ids = _tok.encode(_text, add_special_tokens=False)
        return int(_ids[0]) if len(_ids) == 1 else None

    src_yes_id = _get_single_token_id(tokenizer, "yes") or _get_single_token_id(tokenizer, " yes")
    src_no_id = _get_single_token_id(tokenizer, "no") or _get_single_token_id(tokenizer, " no")
    if src_yes_id is None or src_no_id is None:
        raise RuntimeError('Failed to find single-token ids for "yes"/"no" (tried both with/without leading space).')

    with torch.no_grad():
        in_emb = model.get_input_embeddings()
        in_w = in_emb.weight
        in_w[yes_id].copy_(in_w[src_yes_id])
        in_w[no_id].copy_(in_w[src_no_id])

        out_emb = model.get_output_embeddings()
        if out_emb is not None:
            out_w = out_emb.weight
            out_w[yes_id].copy_(out_w[src_yes_id])
            out_w[no_id].copy_(out_w[src_no_id])
            if getattr(out_emb, "bias", None) is not None:
                out_emb.bias[yes_id].copy_(out_emb.bias[src_yes_id])
                out_emb.bias[no_id].copy_(out_emb.bias[src_no_id])

    keep_ids = torch.tensor([yes_id, no_id], dtype=torch.long)

    def _mask_rows_hook(_keep_ids: torch.Tensor):
        def _hook(_grad: torch.Tensor) -> torch.Tensor:
            if _grad is None:
                return None
            _k = _keep_ids.to(device=_grad.device)
            _g = _grad
            _out = torch.zeros_like(_g)
            _out[_k] = _g[_k]
            return _out
        return _hook

    # input embeddings: enable grads but mask to only <yes>/<no> rows
    in_emb = model.get_input_embeddings()
    in_emb.weight.requires_grad_(True)
    in_emb.weight.register_hook(_mask_rows_hook(keep_ids))

    # output embeddings (lm_head): enable grads but mask to only <yes>/<no> rows (avoid double hook if tied)
    out_emb = model.get_output_embeddings()
    if out_emb is not None:
        same_weight = (out_emb.weight.data_ptr() == in_emb.weight.data_ptr())
        if not same_weight:
            out_emb.weight.requires_grad_(True)
            out_emb.weight.register_hook(_mask_rows_hook(keep_ids))
        if getattr(out_emb, "bias", None) is not None:
            out_emb.bias.requires_grad_(True)
            out_emb.bias.register_hook(_mask_rows_hook(keep_ids))

 
    # LoRA
    if args.use_lora:
        target_modules = [x.strip() for x in args.lora_target_modules.split(",") if x.strip()]
        lora_cfg = LoraConfig(
            task_type="CAUSAL_LM",
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    # load datasets
    train_ds = load_split(args.train_file)
    val_ds = load_split(args.validation_file)
    test_ds = load_split(args.test_file) if args.test_file else None

    def preprocess_example(ex: Dict[str, Any]) -> Dict[str, Any]:
        instruction = ex["instruction"]
        inp = ex["input"]
        out = ex["output"]

        # parse output -> tags -> targets
        tags = parse_output_to_tags(out, label_order)
        multihot = tags_to_multihot(tags)  # [C]

        full_text = build_full_text(instruction, inp, out)
        prompt_text = build_prompt_only(instruction, inp)

        # tokenize WITHOUT adding special tokens (positions stable)
        ids = tokenizer(full_text, add_special_tokens=False, truncation=True, max_length=args.max_length)["input_ids"]
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False, truncation=True, max_length=args.max_length)["input_ids"]

        prompt_len = len(prompt_ids)
        if prompt_len > len(ids):
            raise ValueError(f"Prompt length > full length after truncation: {prompt_len} > {len(ids)}")

        # labels for SFT: mask prompt tokens
        labels = ids[:]  # token-level
        for i in range(min(prompt_len, len(labels))):
            labels[i] = IGNORE_INDEX

        attn = [1] * len(ids)

        # label_positions MUST be located ONLY within output segment
        output_ids = ids[prompt_len:]
        output_pos_rel = [j for j, tid in enumerate(output_ids) if tid == yes_id or tid == no_id]
        pos = [prompt_len + j for j in output_pos_rel]

        if len(pos) != num_labels:
            raise ValueError(
                f"label_positions mismatch (output-only scan): got {len(pos)} expected {num_labels}. "
                f"Maybe truncated output or output template mismatch."
            )

        return {
            "input_ids": ids,
            "attention_mask": attn,
            "labels": labels,
            "label_positions": pos,
            "binary_targets": multihot.tolist(),
        }

    # map without multiprocessing for stability
    train_tok = train_ds.map(preprocess_example, remove_columns=train_ds.column_names)
    val_tok = val_ds.map(preprocess_example, remove_columns=val_ds.column_names)
    test_tok = test_ds.map(preprocess_example, remove_columns=test_ds.column_names) if test_ds else None

    # pos_weight
    pos_weight = None
    if args.use_pos_weight:
        all_t = np.array(train_tok["binary_targets"], dtype=np.float32)  # [N,C]
        pos_weight = compute_pos_weight_from_targets(all_t, args.pos_weight_clip_max)
        print("[INFO] pos_weight (first 10):", pos_weight[:10].tolist())

    # training args
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,

        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,

        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,


        metric_for_best_model = "loss",
        greater_is_better = False,


        bf16=args.bf16,
        fp16=False,
        report_to="none",
        remove_unused_columns=False,
    )

    callbacks = [
        EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_threshold=args.early_stopping_threshold,
        ),
        TrainingCurveCallback(output_dir=args.output_dir),
    ]

    collator = MultiLabelDataCollator(tokenizer, num_labels)

    trainer = MultiLabelSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=collator,
        tokenizer=tokenizer,
        callbacks=callbacks,

        num_labels=num_labels,
        yes_id=yes_id,
        no_id=no_id,
        parent_child_pairs=parent_child_pairs,

        # ONLY affects eval/test
        use_teacher_forcing_logits=args.use_teacher_forcing_logits,

        use_bce=args.use_bce_loss,
        use_dice=args.use_dice_loss,
        use_hier=args.use_hier_loss,

        lambda_sft=args.lambda_sft,
        lambda_bce=args.lambda_bce,
        lambda_dice=args.lambda_dice,
        lambda_hier=args.lambda_hier,

        pos_weight=pos_weight,

        fixed_threshold=args.threshold,
        threshold_indicator=args.threshold_indicator,
        threshold_grid_step=args.threshold_grid_step,
    )

    # train (always teacher forcing)
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # eval (teacher forcing logits optional)
    eval_metrics = trainer.evaluate()
    print("[INFO] Eval metrics:", eval_metrics)

    # optional: run offline generate-based test (deployment-like)
    if args.run_generate_test and test_ds is not None:
        print("[INFO] Running generation-based prediction on test set...")
        model.eval()

        # --- original outputs must remain unchanged when rgt_eval_logits=False ---
        if not args.rgt_eval_logits:
            preds = []
            golds = []
            for ex in test_ds:
                mh = generate_and_parse(
                    model=model,
                    tokenizer=tokenizer,
                    instruction=ex["instruction"],
                    inp=ex["input"],
                    label_order=label_order,
                    max_new_tokens=args.gen_max_new_tokens,
                    temperature=args.gen_temperature,
                    top_p=args.gen_top_p,
                    do_sample=args.do_sample,
                )
                tags = parse_output_to_tags(ex["output"], label_order)
                gt = tags_to_multihot(tags)
                preds.append(mh)
                golds.append(gt)

            preds = np.array(preds, dtype=np.int32)
            golds = np.array(golds, dtype=np.int32)

            m = compute_metrics_from_preds(preds, golds)
            m["hier_violation_rate"] = hierarchical_violation_rate(preds, parent_child_pairs)
            print("[INFO] Generate-based test metrics:", m)

            out_path = os.path.join(args.output_dir, "generated_test_predictions.jsonl")
            with open(out_path, "w", encoding="utf-8") as f:
                for i in range(len(preds)):
                    f.write(json.dumps({"pred": preds[i].tolist(), "gold": golds[i].tolist()}, ensure_ascii=False) + "\n")
            print(f"[INFO] Saved generate-based predictions to: {out_path}")

        else:
            # --- rgt_eval_logits=True: keep text-parse evaluation + add logit-extract evaluation ---
            preds_text = []
            preds_logits = []
            golds = []

            for ex in test_ds:
                mh_text, bl = generate_and_parse_with_logits(
                    model=model,
                    tokenizer=tokenizer,
                    instruction=ex["instruction"],
                    inp=ex["input"],
                    label_order=label_order,
                    max_new_tokens=args.gen_max_new_tokens,
                    temperature=args.gen_temperature,
                    top_p=args.gen_top_p,
                    do_sample=args.do_sample,
                    yes_id=yes_id,
                    no_id=no_id,
                    num_labels=num_labels,
                )
                tags = parse_output_to_tags(ex["output"], label_order)
                gt = tags_to_multihot(tags)
                preds_text.append(mh_text)
                preds_logits.append(bl)
                golds.append(gt)

            preds_text = np.array(preds_text, dtype=np.int32)     # [N,C]
            preds_logits = np.array(preds_logits, dtype=np.float32)  # [N,C]
            golds = np.array(golds, dtype=np.int32)              # [N,C]

            # text-parse metrics (same values as original path)
            m_text = compute_metrics_from_preds(preds_text, golds)
            m_text["hier_violation_rate"] = hierarchical_violation_rate(preds_text, parent_child_pairs)
            print("Generate-based text-parse metrics:", m_text)

            # logit-based metrics with threshold search (or fixed if threshold_indicator is None)
            grid = np.arange(0.1, 0.91, 0.05, dtype=np.float32)
            best_t, m_logit = search_best_threshold_for_logits(
                binary_logits=preds_logits,
                labels=golds,
                parent_child_pairs=parent_child_pairs,
                threshold_indicator=args.threshold_indicator,
                fixed_threshold=args.threshold,
                threshold_grid=grid,
                pos_weight=pos_weight,
                use_bce=args.use_bce_loss,
                use_dice=args.use_dice_loss,
                use_hier=args.use_hier_loss,
                lambda_bce=args.lambda_bce,
                lambda_dice=args.lambda_dice,
                lambda_hier=args.lambda_hier,
            )
            # ensure required metrics exist in print dict
            # (compute_metrics_from_preds already provides micro/macro/exact/hamming; plus hier_violation_rate)
            print(f"Generate-based logit-extract metrics (best threshold {best_t:.2f}):", m_logit)

            # keep original artifact saving behavior (save text-parse preds only)
            out_path = os.path.join(args.output_dir, "generated_test_predictions.jsonl")
            with open(out_path, "w", encoding="utf-8") as f:
                for i in range(len(preds_text)):
                    f.write(json.dumps({"pred": preds_text[i].tolist(), "gold": golds[i].tolist()}, ensure_ascii=False) + "\n")
            print(f"[INFO] Saved generate-based predictions to: {out_path}")


if __name__ == "__main__":
    main()
