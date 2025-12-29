"""
训练Have-1和Have-2探针的脚本
Have-1: 用于区分原始表示和反事实表示的二元分类探针
Have-2: 用于预测样本标签的多分类探针
"""
import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

# 设置路径，将src目录添加到Python路径中
file_dir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(file_dir, "..", "src")))


@dataclass
class ProbeConfig:
    """探针训练配置类"""
    model_type: str  # 模型类型: "linear", "mlp" 或 "lstm"
    hidden_dim: int  # MLP/LSTM模型的隐藏层维度
    epochs: int  # 训练轮数
    lr: float  # 学习率
    batch_size: int  # 批次大小
    seed: int = 42  # 随机种子
    lstm_layers: int = 1  # LSTM层数


class RepresentationDataset(Dataset):
    """表示特征数据集类，用于存储特征向量和标签"""
    def __init__(self, features: List[List[float]], labels: List[int], groups: Optional[List[str]] = None):
        """
        初始化数据集
        Args:
            features: 特征向量列表
            labels: 标签列表
            groups: 可选的分组信息列表，用于数据分割时保持同一组的数据在同一集合中
        """
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.groups = groups or [""] * len(labels)

    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取指定索引的数据样本"""
        return self.features[idx], self.labels[idx]


class SequenceRepresentationDataset(Dataset):
    """序列表示数据集，用于加载形状为 (seq_len, feature_dim) 的张量"""
    def __init__(self, entries: List[Dict[str, object]], groups: Optional[List[str]] = None):
        """
        初始化数据集
        Args:
            entries: 样本条目，包含路径与标签
            groups: 可选的分组信息列表
        """
        self.entries = entries
        self.groups = groups or [""] * len(entries)

    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.entries)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取指定索引的数据样本"""
        entry = self.entries[idx]
        label = torch.tensor(entry["label"], dtype=torch.long)
        if entry["type"] == "single":
            tensor = torch.load(entry["path"], map_location="cpu")
        else:
            first = torch.load(entry["path_a"], map_location="cpu")
            second = torch.load(entry["path_b"], map_location="cpu")
            tensor = torch.cat([first, second], dim=0)
        return tensor.float(), label


class LinearProbe(nn.Module):
    """线性探针模型：单层线性变换"""
    def __init__(self, input_dim: int, output_dim: int):
        """
        初始化线性探针
        Args:
            input_dim: 输入特征维度
            output_dim: 输出维度（类别数）
        """
        super().__init__()
        self.layer = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.layer(x)


class MLPProbe(nn.Module):
    """多层感知机探针模型：包含一个隐藏层的全连接网络"""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        """
        初始化MLP探针
        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            output_dim: 输出维度（类别数）
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # 输入层到隐藏层
            nn.ReLU(),  # ReLU激活函数
            nn.Linear(hidden_dim, output_dim),  # 隐藏层到输出层
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.net(x)


class LSTMProbe(nn.Module):
    """LSTM探针模型：使用LSTM处理序列表示"""
    def __init__(
        self, 
        input_size: int, 
        hidden_dim: int, 
        output_dim: int,
        num_layers: int = 1,
    ):
        """
        初始化LSTM探针
        Args:
            input_size: 输入特征维度（每个时间步）
            hidden_dim: LSTM隐藏层维度
            output_dim: 输出维度（类别数）
            num_layers: LSTM层数
        """
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, input_size)
        Returns:
            输出logits，形状为 (batch_size, output_dim)
        """
        # lstm_out的形状：(batch_size, seq_len, hidden_dim)
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]  # (batch_size, hidden_dim)
        output = self.fc(last_hidden)
        return output


def _select_representation(
    representations: Dict,
    rep_key: str,
    layer_idx: Optional[int],
) -> Union[List[float], str]:
    """
    从表示字典中选择指定的表示向量
    Args:
        representations: 包含不同表示的字典
        rep_key: 表示类型键 ("inputs_mean", "visual_mean", "layers_mean")
        layer_idx: 层索引（仅在rep_key为"layers_mean"时需要）
    Returns:
        选中的表示向量
    """
    # 这里的 rep_key 可以为 "inputs_mean"（全输入平均表示），"visual_mean"（视觉帧tokens平均表示），
    # 或 "layers_mean"（每层的平均表示，需通过 layer_idx 指定层数）
    if rep_key == "inputs_mean":  # 输入（text+vision）token的平均表示
        return representations["inputs_mean"]
    if rep_key == "visual_mean":  # 仅视觉tokens的平均表示
        return representations["visual_mean"]
    if rep_key == "layers_mean":  # 所有层的平均表示，需 layer_idx 指定具体哪一层
        if layer_idx is None:
            raise ValueError("layer_idx must be provided when using layers_mean.")
        return representations["layers_mean"][layer_idx]
    if rep_key == "vprime_path":
        return representations["vprime_path"]
    raise ValueError(f"Unknown representation key: {rep_key}")


def _load_tracing_results(path: str) -> Dict:
    """
    加载因果追踪结果JSON文件
    Args:
        path: JSON文件路径
    Returns:
        包含追踪结果的字典
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_representation_results(path: str) -> Dict:
    """
    加载表示结果JSON文件
    Args:
        path: JSON文件路径
    Returns:
        包含表示结果的字典
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _merge_representations(tracing: Dict, representations: Dict) -> Dict:
    """
    将表示结果合并到追踪结果中
    Args:
        tracing: 因果追踪结果字典
        representations: 表示结果字典
    Returns:
        合并后的追踪结果字典
    """
    for sample_id, rep_entry in representations.items():
        sample = tracing.get(sample_id)
        if not sample:
            continue
        # 合并原始样本的表示
        if "representations" in rep_entry:
            sample["representations"] = rep_entry["representations"]
        # 合并反事实样本的表示
        rep_counterfactuals = rep_entry.get("counterfactuals", {})
        if not rep_counterfactuals:
            continue
        tracing_counterfactuals = sample.setdefault("counterfactuals", {})
        for cf_name, cf_rep_entry in rep_counterfactuals.items():
            cf_sample = tracing_counterfactuals.setdefault(cf_name, {})
            if "representations" in cf_rep_entry:
                cf_sample["representations"] = cf_rep_entry["representations"]
    return tracing


def _load_annotations(path: str) -> Dict[str, Dict]:
    """
    加载标注数据并构建查找字典
    Args:
        path: 标注JSON文件路径
    Returns:
        以样本ID为键的标注字典
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    lookup: Dict[str, Dict] = {}
    for item in data:
        # 尝试多种可能的ID字段
        key = str(item.get("question_id") or item.get("video_name") or item.get("video_path"))
        lookup[key] = item
    return lookup


def _build_have1_dataset(
    tracing: Dict,
    counterfactual_type: str,
    rep_key: str,
    layer_idx: Optional[int],
    have1_mode: str,
) -> Union[RepresentationDataset, SequenceRepresentationDataset]:
    """
    构建Have-1数据集：区分原始表示和反事实表示的二元分类任务
    每个样本对会生成两个训练样本：正样本(原始+反事实)和负样本(反事实+原始)
    Args:
        tracing: 因果追踪结果字典
        counterfactual_type: 反事实类型 ("all"表示使用所有类型)
        rep_key: 表示类型键
        layer_idx: 层索引
    Returns:
        Have-1数据集
    """
    if rep_key == "vprime_path":
        entries: List[Dict[str, object]] = []
        groups: List[str] = []
    else:
        features: List[List[float]] = []
        labels: List[int] = []
        groups = []
    for sample_id, sample in tracing.items():
        reps = sample.get("representations")
        if not reps:
            continue
        clean_vec = _select_representation(reps, rep_key, layer_idx)
        
        if counterfactual_type == "all":
            # 使用所有类型的反事实
            counterfactuals = sample.get("counterfactuals", {})
            for cf in counterfactuals.values():
                cf_rep = cf.get("representations")
                if not cf_rep:
                    continue
                cf_vec = _select_representation(cf_rep, rep_key, layer_idx)
                if rep_key == "vprime_path":
                    if have1_mode == "single":
                        entries.append({"type": "single", "path": clean_vec, "label": 1})
                        groups.append(str(sample_id))
                        entries.append({"type": "single", "path": cf_vec, "label": 0})
                        groups.append(str(sample_id))
                    else:
                        entries.append(
                            {"type": "pair", "path_a": clean_vec, "path_b": cf_vec, "label": 1}
                        )
                        groups.append(str(sample_id))
                        entries.append(
                            {"type": "pair", "path_a": cf_vec, "path_b": clean_vec, "label": 0}
                        )
                        groups.append(str(sample_id))
                else:
                    if have1_mode == "single":
                        features.append(clean_vec)
                        labels.append(1)
                        groups.append(str(sample_id))
                        features.append(cf_vec)
                        labels.append(0)
                        groups.append(str(sample_id))
                    else:
                        # 正样本：原始向量 + 反事实向量
                        features.append(clean_vec + cf_vec)
                        labels.append(1)
                        groups.append(str(sample_id))
                        # 负样本：反事实向量 + 原始向量
                        features.append(cf_vec + clean_vec)
                        labels.append(0)
                        groups.append(str(sample_id))
        else:
            # 使用指定类型的反事实
            cf = sample.get("counterfactuals", {}).get(counterfactual_type)
            if not cf:
                continue
            cf_rep = cf.get("representations")
            if not cf_rep:
                continue
            cf_vec = _select_representation(cf_rep, rep_key, layer_idx)
            if rep_key == "vprime_path":
                if have1_mode == "single":
                    entries.append({"type": "single", "path": clean_vec, "label": 1})
                    groups.append(str(sample_id))
                    entries.append({"type": "single", "path": cf_vec, "label": 0})
                    groups.append(str(sample_id))
                else:
                    entries.append({"type": "pair", "path_a": clean_vec, "path_b": cf_vec, "label": 1})
                    groups.append(str(sample_id))
                    entries.append({"type": "pair", "path_a": cf_vec, "path_b": clean_vec, "label": 0})
                    groups.append(str(sample_id))
            else:
                if have1_mode == "single":
                    features.append(clean_vec)
                    labels.append(1)
                    groups.append(str(sample_id))
                    features.append(cf_vec)
                    labels.append(0)
                    groups.append(str(sample_id))
                else:
                    # 正样本：原始向量 + 反事实向量
                    features.append(clean_vec + cf_vec)
                    labels.append(1)
                    groups.append(str(sample_id))
                    # 负样本：反事实向量 + 原始向量
                    features.append(cf_vec + clean_vec)
                    labels.append(0)
                    groups.append(str(sample_id))
    if rep_key == "vprime_path":
        return SequenceRepresentationDataset(entries, groups)
    return RepresentationDataset(features, labels, groups)


def _build_have2_dataset(
    tracing: Dict,
    annotations: Dict[str, Dict],
    label_key: str,
    rep_key: str,
    layer_idx: Optional[int],
) -> Tuple[Union[RepresentationDataset, SequenceRepresentationDataset], Dict[int, str]]:
    """
    构建Have-2数据集：基于表示预测样本标签的多分类任务
    Args:
        tracing: 因果追踪结果字典
        annotations: 标注字典
        label_key: 标注中用作标签的键名
        rep_key: 表示类型键
        layer_idx: 层索引
    Returns:
        数据集和标签索引到标签名的映射字典
    """
    if rep_key == "vprime_path":
        entries: List[Dict[str, object]] = []
        groups: List[str] = []
    else:
        features: List[List[float]] = []
        labels: List[int] = []
        groups = []
    label_map: Dict[str, int] = {}  # 标签名到索引的映射
    
    for sample_id, sample in tracing.items():
        reps = sample.get("representations")
        if not reps:
            continue
        ann = annotations.get(sample_id)
        if not ann or label_key not in ann:
            continue
        # 获取原始标签并转换为索引
        raw_label = str(ann[label_key])
        if raw_label not in label_map:
            label_map[raw_label] = len(label_map)
        # 添加特征和标签
        if rep_key == "vprime_path":
            entries.append(
                {
                    "type": "single",
                    "path": _select_representation(reps, rep_key, layer_idx),
                    "label": label_map[raw_label],
                }
            )
            groups.append(str(sample_id))
        else:
            features.append(_select_representation(reps, rep_key, layer_idx))
            labels.append(label_map[raw_label])
            groups.append(str(sample_id))
    
    # 构建索引到标签名的反向映射
    index_to_label = {idx: label for label, idx in label_map.items()}
    if rep_key == "vprime_path":
        return SequenceRepresentationDataset(entries, groups), index_to_label
    return RepresentationDataset(features, labels, groups), index_to_label


def _group_split(
    dataset: Union[RepresentationDataset, SequenceRepresentationDataset],
    seed: int,
    val_fraction: float = 0.2,
) -> Tuple[List[int], List[int]]:
    """
    按组划分数据集，确保同一组的数据不会同时出现在训练集和验证集中
    Args:
        dataset: 数据集
        seed: 随机种子
        val_fraction: 验证集比例
    Returns:
        训练集索引列表和验证集索引列表
    """
    unique_groups = sorted(set(dataset.groups))
    if not unique_groups:
        # 如果没有分组信息，返回所有数据作为训练集
        return list(range(len(dataset))), []
    
    # 随机打乱组顺序
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(unique_groups), generator=generator).tolist()
    
    # 计算验证集大小
    val_size = max(1, int(val_fraction * len(unique_groups)))
    if val_size >= len(unique_groups):
        val_size = len(unique_groups) - 1
    
    # 选择验证组
    val_groups = set(unique_groups[idx] for idx in perm[:val_size])
    
    # 根据组划分样本索引
    train_indices = [i for i, group in enumerate(dataset.groups) if group not in val_groups]
    val_indices = [i for i, group in enumerate(dataset.groups) if group in val_groups]
    
    # 确保训练集不为空
    if not train_indices:
        train_indices = val_indices
    return train_indices, val_indices


def _binary_auroc(scores: List[float], labels: List[int]) -> float:
    """
    计算二元分类的AUROC（Area Under ROC Curve）值
    使用Wilcoxon秩和统计量的计算方法
    Args:
        scores: 预测分数列表
        labels: 真实标签列表（0或1）
    Returns:
        AUROC值（0到1之间）
    """
    # 按分数排序
    paired = sorted(zip(scores, labels), key=lambda x: x[0])
    pos = sum(labels)  # 正样本数量
    neg = len(labels) - pos  # 负样本数量
    
    if pos == 0 or neg == 0:
        return 0.0
    
    # 计算正样本的秩和
    rank_sum = 0.0
    for rank, (_, label) in enumerate(paired, start=1):
        if label == 1:
            rank_sum += rank
    
    # 使用Wilcoxon秩和统计量计算AUROC
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


def _train_probe(
    dataset: Union[RepresentationDataset, SequenceRepresentationDataset],
    config: ProbeConfig,
    output_dim: int,
) -> Dict[str, float]:
    """
    训练探针模型并评估性能
    Args:
        dataset: 训练数据集
        config: 探针配置
        output_dim: 输出维度（类别数）
    Returns:
        包含验证集指标的字典
    """
    if len(dataset) == 0:
        raise ValueError("No samples available for training. Check representations and labels.")
    
    # 检测并设置设备（优先使用CUDA，如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"使用CUDA设备进行训练: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA不可用，使用CPU进行训练")
    
    # 设置随机种子
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    
    # 按组划分训练集和验证集
    train_indices, val_indices = _group_split(dataset, config.seed)
    train_subset = torch.utils.data.Subset(dataset, train_indices)
    val_subset = torch.utils.data.Subset(dataset, val_indices)
    train_loader = DataLoader(train_subset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=config.batch_size)

    # 创建模型
    sample_x, _ = dataset[0]
    if sample_x.dim() == 1:
        input_dim = sample_x.shape[0]
    else:
        input_dim = sample_x.shape[1]
    if config.model_type == "linear":
        if sample_x.dim() != 1:
            raise ValueError("linear probe expects 1D input tensors.")
        model = LinearProbe(input_dim, output_dim)
    elif config.model_type == "mlp":
        if sample_x.dim() != 1:
            raise ValueError("mlp probe expects 1D input tensors.")
        model = MLPProbe(input_dim, config.hidden_dim, output_dim)
    elif config.model_type == "lstm":
        if sample_x.dim() != 2:
            raise ValueError("lstm probe expects 2D input tensors.")
        model = LSTMProbe(
            input_size=input_dim,
            hidden_dim=config.hidden_dim,
            output_dim=output_dim,
            num_layers=config.lstm_layers,
        )
    else:
        raise ValueError(f"Unknown model_type: {config.model_type}")
    
    # 将模型移动到指定设备
    model = model.to(device)

    # 设置优化器和损失函数
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()

    # 训练循环
    for _ in range(config.epochs):
        model.train()
        for x, y in train_loader:
            # 将输入数据移动到指定设备
            x = x.to(device)
            y = y.to(device)
            
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

    # 评估模型
    model.eval()
    correct = 0
    total = 0
    scores: List[float] = []  # 用于计算AUROC的分数
    labels: List[int] = []  # 真实标签
    with torch.no_grad():
        for x, y in val_loader:
            # 将输入数据移动到指定设备
            x = x.to(device)
            y = y.to(device)
            
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.numel()
            # 对于二元分类，计算AUROC
            if output_dim == 2:
                probs = torch.softmax(logits, dim=1)[:, 1]  # 正类概率
                scores.extend(probs.cpu().tolist())  # 转移到CPU再转为列表
                labels.extend(y.cpu().tolist())  # 转移到CPU再转为列表
    
    # 计算准确率
    accuracy = correct / total if total > 0 else 0.0
    metrics = {"val_accuracy": accuracy}
    
    # 对于二元分类，计算AUROC
    if output_dim == 2 and labels:
        metrics["val_auroc"] = _binary_auroc(scores, labels)
    return metrics


def main() -> None:
    """
    主函数：解析参数，加载数据，训练探针，保存结果
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Train Have-1/Have-2 probes on tracing outputs.")
    parser.add_argument("--tracing_results", required=True, help="Path to causal tracing results JSON.")
    parser.add_argument(
        "--representations",
        default=None,
        help="Optional path to representations JSON saved separately from tracing results.",
    )
    parser.add_argument("--output", required=True, help="Path to save probe metrics JSON.")
    parser.add_argument(
        "--probe_type",
        choices=["have1", "have2"],
        required=True,
        help="Probe type to train: have1 or have2.",
    )
    parser.add_argument(
        "--representation",
        choices=["inputs_mean", "layers_mean", "visual_mean", "vprime_path"],
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
    parser.add_argument("--model_type", choices=["linear", "mlp", "lstm"], default="linear")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lstm_layers", type=int, default=1, help="Number of LSTM layers (for LSTM model)")
    parser.add_argument(
        "--have1_mode",
        choices=["single", "pair"],
        default="pair",
        help="Have-1 mode: single (clean vs cf) or pair (concat clean/cf).",
    )

    args = parser.parse_args()
    
    # 加载追踪结果
    tracing = _load_tracing_results(args.tracing_results)
    
    # 如果提供了单独的表示文件，则合并
    if args.representations:
        representations = _load_representation_results(args.representations)
        tracing = _merge_representations(tracing, representations)
    
    # 创建探针配置
    config = ProbeConfig(
        model_type=args.model_type,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
        lstm_layers=args.lstm_layers,
    )

    # 根据探针类型构建数据集并训练
    if args.probe_type == "have1":
        # Have-1: 二元分类任务（区分原始和反事实）
        dataset = _build_have1_dataset(
            tracing,
            args.counterfactual_type,
            args.representation,
            args.layer_idx,
            args.have1_mode,
        )
        output_dim = 2
        metrics = _train_probe(dataset, config, output_dim)
        output = {
            "probe_type": "have1",
            "counterfactual_type": args.counterfactual_type,
            "representation": args.representation,
            "have1_mode": args.have1_mode,
            "layer_idx": args.layer_idx,
            "metrics": metrics,
            "num_samples": len(dataset),
        }
    else:
        # Have-2: 多分类任务（预测标签）
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

    # 保存结果到JSON文件
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
