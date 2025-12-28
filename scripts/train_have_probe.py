import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

file_dir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(file_dir, "..", "src")))


@dataclass
class ProbeConfig:
    model_type: str
    hidden_dim: int
    epochs: int
    lr: float
    batch_size: int
    seed: int


class RepresentationDataset(Dataset):
    def __init__(self, features: List[List[float]], labels: List[int], groups: Optional[List[str]] = None):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.groups = groups or [""] * len(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


class LinearProbe(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.layer = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class MLPProbe(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _select_representation(representations: Dict, rep_key: str, layer_idx: Optional[int]) -> List[float]:
    if rep_key == "inputs_mean":
        return representations["inputs_mean"]
    if rep_key == "visual_mean":
        return representations["visual_mean"]
    if rep_key == "layers_mean":
        if layer_idx is None:
            raise ValueError("layer_idx must be provided when using layers_mean.")
        return representations["layers_mean"][layer_idx]
    raise ValueError(f"Unknown representation key: {rep_key}")


def _load_tracing_results(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_annotations(path: str) -> Dict[str, Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    lookup: Dict[str, Dict] = {}
    for item in data:
        key = str(item.get("question_id") or item.get("video_name") or item.get("video_path"))
        lookup[key] = item
    return lookup


def _build_have1_dataset(
    tracing: Dict,
    counterfactual_type: str,
    rep_key: str,
    layer_idx: Optional[int],
) -> RepresentationDataset:
    features: List[List[float]] = []
    labels: List[int] = []
    groups: List[str] = []
    for sample_id, sample in tracing.items():
        reps = sample.get("representations")
        if not reps:
            continue
        clean_vec = _select_representation(reps, rep_key, layer_idx)
        if counterfactual_type == "all":
            counterfactuals = sample.get("counterfactuals", {})
            for cf in counterfactuals.values():
                cf_rep = cf.get("representations")
                if not cf_rep:
                    continue
                cf_vec = _select_representation(cf_rep, rep_key, layer_idx)
                features.append(clean_vec + cf_vec)
                labels.append(1)
                groups.append(str(sample_id))
                features.append(cf_vec + clean_vec)
                labels.append(0)
                groups.append(str(sample_id))
        else:
            cf = sample.get("counterfactuals", {}).get(counterfactual_type)
            if not cf:
                continue
            cf_rep = cf.get("representations")
            if not cf_rep:
                continue
            cf_vec = _select_representation(cf_rep, rep_key, layer_idx)
            features.append(clean_vec + cf_vec)
            labels.append(1)
            groups.append(str(sample_id))
            features.append(cf_vec + clean_vec)
            labels.append(0)
            groups.append(str(sample_id))
    return RepresentationDataset(features, labels, groups)


def _build_have2_dataset(
    tracing: Dict,
    annotations: Dict[str, Dict],
    label_key: str,
    rep_key: str,
    layer_idx: Optional[int],
) -> Tuple[RepresentationDataset, Dict[int, str]]:
    features: List[List[float]] = []
    labels: List[int] = []
    groups: List[str] = []
    label_map: Dict[str, int] = {}
    for sample_id, sample in tracing.items():
        reps = sample.get("representations")
        if not reps:
            continue
        ann = annotations.get(sample_id)
        if not ann or label_key not in ann:
            continue
        raw_label = str(ann[label_key])
        if raw_label not in label_map:
            label_map[raw_label] = len(label_map)
        features.append(_select_representation(reps, rep_key, layer_idx))
        labels.append(label_map[raw_label])
        groups.append(str(sample_id))
    index_to_label = {idx: label for label, idx in label_map.items()}
    return RepresentationDataset(features, labels, groups), index_to_label


def _group_split(dataset: RepresentationDataset, seed: int, val_fraction: float = 0.2) -> Tuple[List[int], List[int]]:
    unique_groups = sorted(set(dataset.groups))
    if not unique_groups:
        return list(range(len(dataset))), []
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(unique_groups), generator=generator).tolist()
    val_size = max(1, int(val_fraction * len(unique_groups)))
    if val_size >= len(unique_groups):
        val_size = len(unique_groups) - 1
    val_groups = set(unique_groups[idx] for idx in perm[:val_size])
    train_indices = [i for i, group in enumerate(dataset.groups) if group not in val_groups]
    val_indices = [i for i, group in enumerate(dataset.groups) if group in val_groups]
    if not train_indices:
        train_indices = val_indices
    return train_indices, val_indices


def _binary_auroc(scores: List[float], labels: List[int]) -> float:
    paired = sorted(zip(scores, labels), key=lambda x: x[0])
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return 0.0
    rank_sum = 0.0
    for rank, (_, label) in enumerate(paired, start=1):
        if label == 1:
            rank_sum += rank
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


def _train_probe(
    dataset: RepresentationDataset,
    config: ProbeConfig,
    output_dim: int,
) -> Dict[str, float]:
    if len(dataset) == 0:
        raise ValueError("No samples available for training. Check representations and labels.")
    torch.manual_seed(config.seed)
    train_indices, val_indices = _group_split(dataset, config.seed)
    train_subset = torch.utils.data.Subset(dataset, train_indices)
    val_subset = torch.utils.data.Subset(dataset, val_indices)
    train_loader = DataLoader(train_subset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=config.batch_size)

    input_dim = dataset.features.shape[1]
    if config.model_type == "linear":
        model = LinearProbe(input_dim, output_dim)
    elif config.model_type == "mlp":
        model = MLPProbe(input_dim, config.hidden_dim, output_dim)
    else:
        raise ValueError(f"Unknown model_type: {config.model_type}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()

    for _ in range(config.epochs):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

    model.eval()
    correct = 0
    total = 0
    scores: List[float] = []
    labels: List[int] = []
    with torch.no_grad():
        for x, y in val_loader:
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.numel()
            if output_dim == 2:
                probs = torch.softmax(logits, dim=1)[:, 1]
                scores.extend(probs.tolist())
                labels.extend(y.tolist())
    accuracy = correct / total if total > 0 else 0.0
    metrics = {"val_accuracy": accuracy}
    if output_dim == 2 and labels:
        metrics["val_auroc"] = _binary_auroc(scores, labels)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Have-1/Have-2 probes on tracing outputs.")
    parser.add_argument("--tracing_results", required=True, help="Path to causal tracing results JSON.")
    parser.add_argument("--output", required=True, help="Path to save probe metrics JSON.")
    parser.add_argument(
        "--probe_type",
        choices=["have1", "have2"],
        required=True,
        help="Probe type to train: have1 or have2.",
    )
    parser.add_argument(
        "--representation",
        choices=["inputs_mean", "layers_mean", "visual_mean"],
        default="inputs_mean",
        help="Representation key to probe.",
    )
    parser.add_argument("--layer_idx", type=int, default=None, help="Layer index for layers_mean representations.")
    parser.add_argument(
        "--counterfactual_type",
        default="all",
        help="Counterfactual type for have1 (order_reverse/local_swap/motion_destroy/motion_only/all).",
    )
    parser.add_argument("--annotations", default=None, help="Annotations JSON (required for have2).")
    parser.add_argument(
        "--label_key",
        default="question_type",
        help="Annotation key to use as label for have2.",
    )
    parser.add_argument("--model_type", choices=["linear", "mlp"], default="linear")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    tracing = _load_tracing_results(args.tracing_results)
    config = ProbeConfig(
        model_type=args.model_type,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    if args.probe_type == "have1":
        dataset = _build_have1_dataset(tracing, args.counterfactual_type, args.representation, args.layer_idx)
        output_dim = 2
        metrics = _train_probe(dataset, config, output_dim)
        output = {
            "probe_type": "have1",
            "counterfactual_type": args.counterfactual_type,
            "representation": args.representation,
            "layer_idx": args.layer_idx,
            "metrics": metrics,
            "num_samples": len(dataset),
        }
    else:
        if not args.annotations:
            raise ValueError("--annotations is required for have2 probes.")
        annotations = _load_annotations(args.annotations)
        dataset, index_to_label = _build_have2_dataset(
            tracing,
            annotations,
            args.label_key,
            args.representation,
            args.layer_idx,
        )
        output_dim = len(index_to_label)
        metrics = _train_probe(dataset, config, output_dim)
        output = {
            "probe_type": "have2",
            "label_key": args.label_key,
            "representation": args.representation,
            "layer_idx": args.layer_idx,
            "metrics": metrics,
            "num_samples": len(dataset),
            "label_map": index_to_label,
        }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
