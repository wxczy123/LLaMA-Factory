import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from llamafactory.data.processor.supervised import SupervisedDatasetProcessor


class DummyTokenizer:
    def __init__(self):
        # Reserve 0 for unknowns, keep yes/no as dedicated ids
        self.vocab = {"<yes>": 1, "<no>": 2}
        self.unk_token_id = 0
        self.eos_token_id = 3
        self.pad_token_id = 4

    def convert_tokens_to_ids(self, token: str):
        return self.vocab.get(token, self.unk_token_id)

    def __call__(self, text: str, add_special_tokens: bool = False):
        # Simple tokenizer that treats <yes>/<no> as whole tokens and every
        # other character as an unknown token. This is enough to verify that
        # label positions are found for every label tag in the sequence.
        ids = []
        i = 0
        while i < len(text):
            if text.startswith("<yes>", i):
                ids.append(self.vocab["<yes>"])
                i += len("<yes>")
            elif text.startswith("<no>", i):
                ids.append(self.vocab["<no>"])
                i += len("<no>")
            else:
                ids.append(self.unk_token_id)
                i += 1
        return {"input_ids": ids}


@pytest.fixture()
def ml_examples():
    assets_dir = Path(__file__).resolve().parents[1] / "assets"
    data = json.loads((assets_dir / "abstract.json").read_text(encoding="utf-8"))
    examples = {
        "_prompt": [None] * len(data),
        "_ml_instruction": [row["instruction"] for row in data],
        "_ml_input": [row["input"] for row in data],
        "_ml_output": [row["output"] for row in data],
    }
    return examples


def test_multi_label_preprocess_keeps_abstract_samples(ml_examples):
    tokenizer = DummyTokenizer()
    data_args = SimpleNamespace(task_type="multi_label_sft_logits", cutoff_len=100000)
    processor = SupervisedDatasetProcessor(template=None, tokenizer=tokenizer, processor=None, data_args=data_args)

    model_inputs = processor.preprocess_dataset(ml_examples)

    # Expect both examples from assets/abstract.json to be kept
    assert len(model_inputs["input_ids"]) == 2
    assert len(model_inputs["binary_targets"]) == 2

    num_labels = processor._num_labels
    for label_positions, binary_targets in zip(
        model_inputs["label_positions"], model_inputs["binary_targets"]
    ):
        assert len(label_positions) == num_labels
        assert len(binary_targets) == num_labels
        assert all(tag in (0.0, 1.0) for tag in binary_targets)

    # Ground-truth binary targets derived from regex parsing of the output should match
    pattern = re.compile(r"(\{[^{}]+\})\s*(<yes>|<no>)")

    for output, stored_targets in zip(ml_examples["_ml_output"], model_inputs["binary_targets"]):
        parsed = pattern.findall(output)
        _, parsed_tags = zip(*parsed)
        expected = [1.0 if tag == "<yes>" else 0.0 for tag in parsed_tags]
        assert expected == stored_targets
