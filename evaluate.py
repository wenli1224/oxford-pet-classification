"""在 Oxford-IIIT Pet 测试集上评估训练好的 ResNet-18。

默认运行：python evaluate.py
指定权重：python evaluate.py --model_path ./outputs/best_model.pth
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

# 只保存图像，使用无需图形界面的后端；必须在导入 pyplot/绘图工具之前设置。
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader

from data.dataset import get_dataloaders
from models.model import create_model
from utils.metrics import (
    calculate_metrics,
    get_top_confusions,
    plot_confusion_matrix,
)


NUM_CLASSES: int = 37


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """解析模型路径、数据目录、批大小与评估结果目录。"""
    parser = argparse.ArgumentParser(description="评估 Oxford-IIIT Pet ResNet-18")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./outputs/best_model.pth",
        help="train.py 保存的模型 state_dict 路径",
    )
    parser.add_argument(
        "--data_root", type=str, default="./data", help="数据下载根目录"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="每批样本数量，默认 32"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/evaluation",
        help="评估结果和混淆矩阵保存目录",
    )
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch_size 必须大于 0")
    return args


def load_model(model_path: Path, device: torch.device) -> nn.Module:
    """创建 37 分类模型，并严格加载 train.py 保存的完整 state_dict。"""
    # 随后会载入全部已训练权重，因此不再下载 ImageNet 预训练权重。
    model = create_model(num_classes=NUM_CLASSES, pretrained=False).to(device)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    return model


def predict_test_set(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> Tuple[List[int], List[int]]:
    """仅在测试集上推理，返回全部真实标签与 Top-1 预测标签。"""
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            # 直接取最大 logit 的类别编号，不需要添加 Softmax。
            predictions = logits.argmax(dim=1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(predictions.cpu().tolist())

    if not y_true:
        raise ValueError("测试 DataLoader 中没有样本，无法计算评估指标。")
    return y_true, y_pred


def main() -> None:
    """加载最佳模型、评估测试集，并导出图像及 JSON 结果。"""
    args = parse_args()
    model_path = Path(args.model_path).expanduser()
    if not model_path.is_file():
        raise FileNotFoundError(
            f"未找到模型权重：{model_path.resolve()}。"
            "请先运行 train.py，或通过 --model_path 指定已有权重。"
        )

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"评估设备: {device}")
    model = load_model(model_path, device)

    # 与训练脚本调用相同接口，复用 seed=42 的划分；只迭代测试 loader。
    _, _, test_loader, class_names = get_dataloaders(
        root=args.data_root, batch_size=args.batch_size
    )
    if len(class_names) != NUM_CLASSES:
        raise ValueError(f"应有 {NUM_CLASSES} 个类别，实际为 {len(class_names)} 个。")
    y_true, y_pred = predict_test_set(model, test_loader, device)
    test_accuracy, macro_f1 = calculate_metrics(y_true, y_pred)
    num_test_samples: int = len(y_true)
    top_confusions = get_top_confusions(y_true, y_pred, class_names, top_k=10)

    print(f"Test Top-1 Accuracy: {test_accuracy:.2%}")
    print(f"Test Macro-F1: {macro_f1:.4f}")
    print(f"测试样本数量: {num_test_samples}")
    print("最容易混淆的前 10 组类别（真实类别 -> 预测类别）:")
    if top_confusions:
        for rank, pair in enumerate(top_confusions, start=1):
            print(
                f"{rank}. {pair['true_class']} -> {pair['predicted_class']}: "
                f"{pair['count']} 次"
            )
    else:
        print("没有误分类。")

    # 工具函数显式保留全部 37 类，并以 300 dpi 保存混淆矩阵。
    confusion_matrix_path = output_dir / "confusion_matrix.png"
    figure, _ = plot_confusion_matrix(
        y_true, y_pred, class_names, save_path=confusion_matrix_path
    )
    plt.close(figure)

    # 准确率和 Macro-F1 均按 0～1 的数值保存，样本数和误分类次数为整数。
    results: Dict[str, object] = {
        "test_accuracy": test_accuracy,
        "macro_f1": macro_f1,
        "num_test_samples": num_test_samples,
        "top_confusions": top_confusions,
    }
    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(results, metrics_file, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"混淆矩阵路径: {confusion_matrix_path.resolve()}")
    print(f"评估结果路径: {metrics_path.resolve()}")


if __name__ == "__main__":
    # Windows 多进程 DataLoader 需要此入口保护。
    main()
