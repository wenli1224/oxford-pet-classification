"""Oxford-IIIT Pet 数据下载、分层划分与 DataLoader 构建。

依赖：torch、torchvision、numpy、scikit-learn、Pillow。
调用 get_dataloaders() 时自动下载；导入本模块不会下载数据。
Windows 下使用多进程加载时，请在调用脚本的
``if __name__ == "__main__":`` 保护块中调用 get_dataloaders()。
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, List, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet


SEED: int = 42
EXPECTED_NUM_IMAGES: int = 7349
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)


class OxfordPetDataset(Dataset):
    """通过图片路径和零起始类别标签加载一个划分，独立应用 Transform。"""

    def __init__(
        self,
        image_paths: Sequence[Path],
        labels: Sequence[int],
        transform: Callable[[Image.Image], torch.Tensor],
    ) -> None:
        if len(image_paths) != len(labels):
            raise ValueError("图片路径数量必须与标签数量一致。")

        self.image_paths: List[Path] = [Path(path) for path in image_paths]
        self.labels: List[int] = [int(label) for label in labels]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        # 统一为 RGB，并及时关闭文件，避免多进程加载时累积文件句柄。
        with Image.open(self.image_paths[index]) as image:
            rgb_image = image.convert("RGB")
        return self.transform(rgb_image), self.labels[index]


def _read_samples(root: Path, split: str) -> Tuple[List[Path], List[int]]:
    """读取官方标注列表，无需为分层划分解码全部图片。"""
    dataset_dir = root / "oxford-iiit-pet"
    annotation_path = dataset_dir / "annotations" / f"{split}.txt"
    image_paths: List[Path] = []
    labels: List[int] = []

    with annotation_path.open("r", encoding="utf-8") as annotation_file:
        for line in annotation_file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            image_id, class_id, _, _ = line.split()
            image_paths.append(dataset_dir / "images" / f"{image_id}.jpg")
            # 官方类别编号为 1～37；PyTorch 分类标签需要 0～36。
            labels.append(int(class_id) - 1)

    return image_paths, labels


def _seed_worker(worker_id: int) -> None:
    """在各 worker 内同步 Python 和 NumPy 的随机种子。

    PyTorch 已根据 DataLoader 的 generator 和 worker_id 设置 worker 种子。
    此函数位于模块顶层，因此可以在 Windows 的 spawn 模式下使用。
    """
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def get_dataloaders(
    root: Union[str, Path] = "data",
    batch_size: int = 32,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """下载并合并官方 trainval/test，然后按 70%/15%/15% 分层划分。

    Args:
        root: 数据下载根目录，数据保存在 root/oxford-iiit-pet 中。
        batch_size: 每批样本数量，默认 32。
        num_workers: 数据加载进程数量，默认 2；设为 0 可禁用多进程。

    Returns:
        train_loader, val_loader, test_loader, class_names。
        class_names 按标签编号排序，class_names[label] 即对应的品种名。
        7349 张图片取整后分为训练 5144 张、验证 1102 张、测试 1103 张。

    Note:
        三个划分互不重叠，均保留各品种的大致比例。
        这里重新划分了官方测试集，评估结果不能直接与官方划分的结果比较。
    """
    if batch_size <= 0:
        raise ValueError("batch_size 必须大于 0。")
    if num_workers < 0:
        raise ValueError("num_workers 不能为负数。")

    # 固定划分、训练 shuffle、随机增强以及 worker 使用的随机种子。
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    root = Path(root).expanduser()

    official_trainval = OxfordIIITPet(
        root=root, split="trainval", target_types="category", download=True
    )
    official_test = OxfordIIITPet(
        root=root, split="test", target_types="category", download=True
    )
    class_names: List[str] = list(official_trainval.classes)
    if class_names != list(official_test.classes):
        raise RuntimeError("官方 trainval 和 test 的类别映射不一致。")

    # 从下载后的公开标注文件提取路径与标签，不依赖 torchvision 私有属性。
    trainval_paths, trainval_labels = _read_samples(root, "trainval")
    test_paths, test_labels = _read_samples(root, "test")
    image_paths = trainval_paths + test_paths
    labels = np.asarray(trainval_labels + test_labels, dtype=np.int64)
    if len(image_paths) != EXPECTED_NUM_IMAGES:
        raise RuntimeError(
            f"合并数据集应包含 {EXPECTED_NUM_IMAGES} 张图片，"
            f"实际为 {len(image_paths)} 张，请检查官方标注文件。"
        )

    # 第一次分层划分：70% 训练，30% 暂存；第二次将暂存部分平分。
    indices = np.arange(len(image_paths))
    train_indices, remaining_indices = train_test_split(
        indices, test_size=0.30, random_state=SEED, stratify=labels
    )
    val_indices, test_indices = train_test_split(
        remaining_indices,
        test_size=0.50,
        random_state=SEED,
        stratify=labels[remaining_indices],
    )

    train_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    train_dataset = OxfordPetDataset(
        [image_paths[int(i)] for i in train_indices],
        labels[train_indices].tolist(),
        train_transform,
    )
    val_dataset = OxfordPetDataset(
        [image_paths[int(i)] for i in val_indices],
        labels[val_indices].tolist(),
        eval_transform,
    )
    test_dataset = OxfordPetDataset(
        [image_paths[int(i)] for i in test_indices],
        labels[test_indices].tolist(),
        eval_transform,
    )

    # 每个 loader 使用独立 generator，验证/测试迭代不会消耗训练的随机状态。
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=_seed_worker,
        generator=torch.Generator().manual_seed(SEED),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=_seed_worker,
        generator=torch.Generator().manual_seed(SEED),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=_seed_worker,
        generator=torch.Generator().manual_seed(SEED),
    )
    return train_loader, val_loader, test_loader, class_names
