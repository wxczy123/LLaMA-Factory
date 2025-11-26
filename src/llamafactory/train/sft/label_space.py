"""Utilities for the hierarchical label space used by the classification prompt.

The label space mirrors the user-provided hierarchy in the training prompt.  We keep
it in code so that both the auxiliary classification loss and any bookkeeping
(e.g. reporting token positions) share a single source of truth.
"""

from __future__ import annotations

from typing import Iterable, Iterator, Sequence


# The canonical label space in the exact order given in the prompt template.
LABEL_SPACE: list[object] = [
    {"Reviews": ["Modern reviews", "Specialized reviews", "Datasets"]},
    {
        "Classification": [
            "Parameterized classifiers",
            {
                "Representations": [
                    "Jet images",
                    "Event images",
                    "Sequences",
                    "Trees",
                    "Graphs",
                    "Sets (point clouds)",
                    "Physics-inspired basis",
                ]
            },
            {
                "Targets": [
                    "$W/Z$ tagging",
                    "$H\\rightarrow b\\bar{b}$",
                    "quarks and gluons",
                    "top quark tagging",
                    "strange jets",
                    "$b$-tagging",
                    "Flavor physics",
                    "BSM particles and models",
                    "Particle identification",
                    "Neutrino Detectors",
                    "Direct Dark Matter Detectors",
                    "Cosmology, Astro Particle, and Cosmic Ray physics",
                    "Tracking",
                    "Heavy Ions / Nuclear Physics",
                ]
            },
            {
                "Learning strategies": [
                    "Hyperparameters",
                    "Weak/Semi supervision",
                    "Unsupervised",
                    "Reinforcement Learning",
                    "Quantum Machine Learning",
                    "Feature ranking",
                    "Attention",
                    "Regularization",
                    "Optimal Transport",
                ]
            },
            {
                "Fast inference / deployment": [
                    "Software",
                    "Hardware/firmware",
                    "Deployment",
                ]
            },
        ]
    },
    {
        "Regression": [
            "Pileup",
            "Calibration",
            "Recasting",
            "Matrix elements",
            "Parameter estimation",
            "Parton Distribution Functions (and related)",
            "Lattice Gauge Theory",
            "Function Approximation",
            "Symbolic Regression",
            "Monitoring",
        ]
    },
    "Equivariant networks.",
    "Physics-informed neural networks (PINNs) / Neural Operators.",
    "Decorrelation methods.",
    {
        "Generative models / density estimation": [
            "GANs",
            "(Variational) Autoencoders",
            "(Continuous) Normalizing flows",
            "Diffusion Models",
            "Transformer Models",
            "Physics-inspired",
            "Mixture Models",
            "Phase space generation",
            "Gaussian processes",
            "Evaluation of Generative Models",
            "Other/hybrid",
        ]
    },
    "Anomaly detection.",
    "Foundation Models, LLMs.",
    "Kolmogorov-Arnold Networks (KANs).",
    {
        "Simulation-based (likelihood-free) Inference": [
            "Parameter estimation",
            "Unfolding",
            "Domain adaptation",
            "BSM",
            "Differentiable Simulation",
        ]
    },
    {
        "Uncertainty Quantification": [
            "Interpretability",
            "Estimation",
            "Mitigation",
            "Uncertainty- and inference-aware learning",
        ]
    },
    {
        "Formal Theory and ML": [
            "Theory and physics for ML",
            "ML for theory",
        ]
    },
    {
        "Experimental results.": [
            "Performance studies",
            "Searches and measurements where ML reconstruction is a core component",
            "Final analysis discriminate for searches",
            "Measurements using deep learning directly (not through object reconstruction)",
        ]
    },
]


def _flatten(space: Sequence[object], prefix: tuple[str, ...] = ()) -> Iterator[tuple[str, ...]]:
    for item in space:
        if isinstance(item, str):
            yield prefix + (item,)
        elif isinstance(item, dict):
            for key, value in item.items():
                yield prefix + (key,)
                yield from _flatten(_to_sequence(value), prefix + (key,))
        else:
            yield from _flatten(_to_sequence(item), prefix)


def _label_metadata(space: Sequence[object]) -> tuple[list[tuple[str, ...]], list[int], list[tuple[int, int]]]:
    """Return flattened paths, leaf indices, and parent-child edges.

    The flattened paths match ``iter_label_paths``. Leaf indices correspond to
    nodes without children, and edges capture the direct parent-child
    relationships for hierarchical losses.
    """

    paths: list[tuple[str, ...]] = []
    leaf_indices: list[int] = []
    edges: list[tuple[int, int]] = []

    index = 0

    def walk(node: object, prefix: tuple[str, ...]) -> int:
        nonlocal index
        if isinstance(node, str):
            current = index
            index += 1
            paths.append(prefix + (node,))
            leaf_indices.append(current)
            return current

        if isinstance(node, dict):
            if len(node) != 1:
                raise ValueError("Dictionaries in the label space must have a single key")
            key, value = next(iter(node.items()))
            current = index
            index += 1
            paths.append(prefix + (key,))
            for child in _to_sequence(value):
                child_root = walk(child, prefix + (key,))
                edges.append((current, child_root))
            return current

        if isinstance(node, Sequence):
            last_child = -1
            for child in node:
                last_child = walk(child, prefix)
            if last_child == -1:
                raise ValueError("Empty sequences are not allowed in the label space")
            return last_child

        raise TypeError(f"Unsupported node type: {type(node)}")

    walk(space, ())
    return paths, leaf_indices, edges


def _to_sequence(value: object) -> Sequence[object]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return value
    return [value]


def iter_label_paths(space: Sequence[object] | None = None) -> Iterable[tuple[str, ...]]:
    """Yield every label path in a depth-first, left-to-right order."""

    yield from _flatten(space or LABEL_SPACE)


def flatten_label_space(space: Sequence[object] | None = None) -> list[str]:
    """Return label paths as readable strings joined by ``>``."""

    return [" > ".join(path) for path in iter_label_paths(space)]


def build_label_output(values: Iterable[int], space: Sequence[object] | None = None) -> str:
    """Render a numeric-only output string following the prompt format.

    Args:
        values: A sequence of integers (0/1) ordered exactly like ``flatten_label_space``.
        space: Optionally override the default label space (useful for testing).

    Returns:
        A compact string such as ``"[{1:[1,0]},{0:[0,1]}]"``.
    """

    iterator = iter(values)
    active_space = space or LABEL_SPACE

    def _render(node: object) -> str:
        if isinstance(node, str):
            return str(next(iterator))
        if isinstance(node, dict):
            if len(node) != 1:
                raise ValueError("Dictionaries in the label space must have a single key")
            key, value = next(iter(node.items()))
            return "{" + str(next(iterator)) + ":" + _render(_to_sequence(value)) + "}"
        if isinstance(node, Sequence):
            return "[" + ",".join(_render(child) for child in node) + "]"
        raise TypeError(f"Unsupported node type: {type(node)}")

    return _render(active_space)


_PATH_TUPLES, LEAF_LABEL_INDICES, PARENT_CHILD_EDGES = _label_metadata(LABEL_SPACE)
LABEL_PATHS: list[str] = [" > ".join(path) for path in _PATH_TUPLES]
LABEL_COUNT: int = len(LABEL_PATHS)

# 1-based positions of the numeric tokens inside the rendered output string. The
# positions are computed with a synthetic all-ones output since the location of
# each digit does not depend on the underlying label value.
_NUMERIC_RENDER = build_label_output([1] * LABEL_COUNT, LABEL_SPACE)
LABEL_DIGIT_POSITIONS: list[int] = [idx for idx, ch in enumerate(_NUMERIC_RENDER, 1) if ch.isdigit()]

if len(LABEL_DIGIT_POSITIONS) != LABEL_COUNT:
    raise ValueError(
        f"Expected {LABEL_COUNT} digits in the rendered label output, found {len(LABEL_DIGIT_POSITIONS)}"
    )

# 1-based position table for quick reference when reading model outputs.
LABEL_POSITION_TABLE: list[dict[str, object]] = [
    {"index": idx + 1, "path": path, "digit_position": LABEL_DIGIT_POSITIONS[idx]}
    for idx, path in enumerate(LABEL_PATHS)
]

