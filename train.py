"""Oxford-IIIT Pet 的 ResNet-18 训练脚本。

默认运行：python train.py
标签平滑：python train.py --label_smoothing 0.1
冻结骨干：python train.py --freeze_backbone
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data.dataset import get_dataloaders
from models.model import create_model


SEED: int = 42
NUM_CLASSES: int = 37
WEIGHT_DECAY: float = 1e-4


def set_seed(seed: int = SEED) -> None:
    """固定 Python、NumPy、PyTorch 及 CUDA 的随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # 关闭 cuDNN 自动算法搜索，使用确定性的卷积算法。
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """解析训练参数，并检查取值范围。"""
    parser = argparse.ArgumentParser(description="训练 Oxford-IIIT Pet ResNet-18")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数，默认 10")
    parser.add_argument(
        "--batch_size", type=int, default=32, help="每批样本数量，默认 32"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率，默认 1e-4")
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.0,
        help="标签平滑系数，范围 [0, 1]，默认 0.0",
    )
    parser.add_argument(
        "--freeze_backbone",
        action="store_true",
        help="冻结骨干参数，仅训练 fc 层",
    )
    parser.add_argument(
        "--data_root", type=str, default="./data", help="数据下载根目录"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./outputs", help="模型和日志保存目录"
    )
    args = parser.parse_args(argv)
    if args.epochs <= 0:
        parser.error("--epochs 必须大于 0")
    if args.batch_size <= 0:
        parser.error("--batch_size 必须大于 0")
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("--lr 必须为有限的正数")
    if not 0.0 <= args.label_smoothing <= 1.0:
        parser.error("--label_smoothing 必须在 [0, 1] 范围内")
    return args


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """训练一轮，返回按样本平均的损失和准确率（范围 0～1）。"""
    model.train()
    total_loss: float = 0.0
    total_correct: int = 0
    total_samples: int = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        # 每批均依次清空梯度、前向传播、反向传播，再更新参数。
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        # 按实际批大小累计，避免最后一个不足整批的数据影响平均值。
        batch_samples: int = labels.size(0)
        total_loss += float(loss.item()) * batch_samples
        predictions = logits.argmax(dim=1)
        total_correct += int((predictions == labels).sum().item())
        total_samples += batch_samples

    if total_samples == 0:
        raise ValueError("训练 DataLoader 中没有样本。")
    return total_loss / total_samples, total_correct / total_samples


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """验证一轮，不计算梯度、不更新模型参数。"""
    model.eval()
    total_loss: float = 0.0
    total_correct: int = 0
    total_samples: int = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            batch_samples: int = labels.size(0)
            total_loss += float(loss.item()) * batch_samples
            predictions = logits.argmax(dim=1)
            total_correct += int((predictions == labels).sum().item())
            total_samples += batch_samples

    if total_samples == 0:
        raise ValueError("验证 DataLoader 中没有样本。")
    return total_loss / total_samples, total_correct / total_samples


def main() -> None:
    """配置实验、训练模型并保存最佳权重与训练历史。"""
    args = parse_args()
    set_seed(SEED)
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = output_dir / "best_model.pth"
    history_path = output_dir / "history.json"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模型选择只使用验证集，测试集留给独立的最终评估。
    train_loader, val_loader, _, class_names = get_dataloaders(
        root=args.data_root, batch_size=args.batch_size
    )
    if len(class_names) != NUM_CLASSES:
        raise ValueError(f"应有 {NUM_CLASSES} 个类别，实际为 {len(class_names)} 个。")
    model = create_model(
        num_classes=NUM_CLASSES,
        pretrained=True,
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    # label_smoothing=0.0 即标准交叉熵；命令行可直接指定 0.1 等系数。
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.lr,
        weight_decay=WEIGHT_DECAY,
    )
    writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))

    # history.json 为逐轮记录列表，准确率以 0～1 的数值保存。
    history: List[Dict[str, Union[int, float]]] = []
    # 从负无穷开始，确保首轮准确率即使为 0 也能保存模型。
    best_val_accuracy: float = float("-inf")
    best_epoch: int = 0
    print(f"训练设备: {device}")
    print(
        f"训练样本: {len(train_loader.dataset)} | "
        f"验证样本: {len(val_loader.dataset)} | 类别数: {NUM_CLASSES}"
    )
    start_time = time.perf_counter()

    try:
        for epoch in range(1, args.epochs + 1):
            train_loss, train_accuracy = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_accuracy = validate(model, val_loader, criterion, device)
            print(
                f"Epoch {epoch}/{args.epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Accuracy: {train_accuracy:.2%} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Accuracy: {val_accuracy:.2%}"
            )

            writer.add_scalar("Loss/Train", train_loss, epoch)
            writer.add_scalar("Loss/Validation", val_loss, epoch)
            writer.add_scalar("Accuracy/Train", train_accuracy, epoch)
            writer.add_scalar("Accuracy/Validation", val_accuracy, epoch)
            writer.flush()

            # 只在验证准确率严格提高时覆盖最佳权重，不使用测试集选模型。
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_epoch = epoch
                # 保存纯 state_dict，便于后续 create_model() 后直接加载。
                torch.save(model.state_dict(), best_model_path)

            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                }
            )
            # 每轮写入历史，保留已完成轮次的记录。
            with history_path.open("w", encoding="utf-8") as history_file:
                json.dump(history, history_file, ensure_ascii=False, indent=2)
    finally:
        # 正常结束或发生异常时均关闭 writer，释放日志文件句柄。
        writer.close()

    elapsed_seconds = time.perf_counter() - start_time
    print(f"最佳验证准确率: {best_val_accuracy:.2%}（Epoch {best_epoch}）")
    print(f"总训练时间: {elapsed_seconds:.2f} 秒（{elapsed_seconds / 60:.2f} 分钟）")
    print(f"最佳模型路径: {best_model_path.resolve()}")


if __name__ == "__main__":
    # Windows 多进程 DataLoader 需要此入口保护。
    main()
