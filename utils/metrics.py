"""分类评估指标、完整类别混淆矩阵与常见误分类统计。"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple, TypedDict, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


LabelArray = Union[Sequence[int], np.ndarray]


class ConfusionPair(TypedDict):
    """一个有方向的误分类类别对及其出现次数。"""

    true_class: str
    predicted_class: str
    count: int


def _validate_predictions(
    y_true: LabelArray, y_pred: LabelArray
) -> Tuple[np.ndarray, np.ndarray]:
    """将标签转为一维 NumPy 数组，并检查输入是否有效。"""
    true_labels = np.asarray(y_true)
    predicted_labels = np.asarray(y_pred)
    if true_labels.ndim != 1 or predicted_labels.ndim != 1:
        raise ValueError("y_true 和 y_pred 必须是一维标签序列。")
    if true_labels.shape != predicted_labels.shape:
        raise ValueError("y_true 和 y_pred 的样本数量必须相同。")
    if true_labels.size == 0:
        raise ValueError("标签序列不能为空。")
    if not np.issubdtype(true_labels.dtype, np.integer) or not np.issubdtype(
        predicted_labels.dtype, np.integer
    ):
        raise TypeError("标签必须为整数类别编号。")
    return true_labels, predicted_labels


def calculate_metrics(y_true: LabelArray, y_pred: LabelArray) -> Tuple[float, float]:
    """返回 (accuracy, macro_f1)，两项均为 0～1 的 Python 浮点数。

    Macro-F1 对 y_true 或 y_pred 中出现的类别等权平均；无法定义的单类
    F1 记为 0。项目测试集包含全部 37 类时，即为 37 类 Macro-F1。
    """
    true_labels, predicted_labels = _validate_predictions(y_true, y_pred)
    accuracy = float(accuracy_score(true_labels, predicted_labels))
    macro_f1 = float(
        f1_score(true_labels, predicted_labels, average="macro", zero_division=0)
    )
    return accuracy, macro_f1


def _build_confusion_matrix(
    y_true: LabelArray,
    y_pred: LabelArray,
    class_names: Sequence[str],
) -> Tuple[np.ndarray, List[str]]:
    """构建包含全部指定类别的矩阵，行是真实类别，列是预测类别。"""
    true_labels, predicted_labels = _validate_predictions(y_true, y_pred)
    if isinstance(class_names, (str, bytes)):
        raise TypeError("class_names 必须是类别名称序列，不能是单个字符串。")
    names = list(class_names)
    if not names or any(not isinstance(name, str) for name in names):
        raise ValueError("class_names 必须包含有效的字符串类别名称。")
    num_classes = len(names)
    if (
        np.any(true_labels < 0)
        or np.any(true_labels >= num_classes)
        or np.any(predicted_labels < 0)
        or np.any(predicted_labels >= num_classes)
    ):
        raise ValueError(f"类别标签必须在 0～{num_classes - 1} 范围内。")

    # 显式指定全部类别：传入 37 个名称时，即使部分标签未出现，仍是 37×37。
    matrix = confusion_matrix(
        true_labels, predicted_labels, labels=np.arange(num_classes)
    )
    return matrix, names


def plot_confusion_matrix(
    y_true: LabelArray,
    y_pred: LabelArray,
    class_names: Sequence[str],
    save_path: Optional[Union[str, Path]] = None,
) -> Tuple[Figure, Axes]:
    """绘制完整类别的计数混淆矩阵，返回 (figure, axes)。

    class_names[i] 对应标签 i；项目中传入全部 37 个类别名称。
    指定 save_path 时以 300 dpi 保存，并创建所需的父目录。
    此函数不自动弹出窗口；调用者可继续编辑图像，使用后 plt.close(figure)。
    """
    matrix, names = _build_confusion_matrix(y_true, y_pred, class_names)
    # 大画布配合完整刻度与紧凑布局，给较长的品种名称留出空间。
    figure, axes = plt.subplots(figsize=(22, 20))
    image = axes.imshow(matrix, interpolation="nearest", cmap="Blues")
    colorbar = figure.colorbar(image, ax=axes, fraction=0.046, pad=0.04)
    colorbar.set_label("Count")

    ticks = np.arange(len(names))
    axes.set_xticks(ticks)
    axes.set_yticks(ticks)
    axes.set_xticklabels(names, rotation=90, ha="center", fontsize=9)
    axes.set_yticklabels(names, fontsize=9)
    axes.set_xlabel("Predicted Label", fontsize=13)
    axes.set_ylabel("True Label", fontsize=13)
    axes.set_title("Confusion Matrix", fontsize=16)

    # 仅标注非零计数，避免 37×37 矩阵被大量的零遮挡。
    threshold = float(matrix.max()) / 2.0
    for row, column in np.argwhere(matrix > 0):
        count = int(matrix[row, column])
        axes.text(
            int(column),
            int(row),
            str(count),
            ha="center",
            va="center",
            color="white" if count > threshold else "black",
            fontsize=7,
        )
    figure.tight_layout()

    if save_path is not None:
        destination = Path(save_path).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        # bbox_inches 防止导出时裁掉较长的类别标签。
        figure.savefig(destination, dpi=300, bbox_inches="tight")
    return figure, axes


def get_top_confusions(
    y_true: LabelArray,
    y_pred: LabelArray,
    class_names: Sequence[str],
    top_k: int = 10,
) -> List[ConfusionPair]:
    """返回误分类次数最多的前 top_k 个有方向类别对。

    每项包含 true_class、predicted_class 和 count，按次数降序排列。
    忽略对角线和零计数；次数相同时按真实、预测类别编号顺序排列。
    若实际误分类类别对不足 top_k，只返回存在的类别对。
    """
    if not isinstance(top_k, (int, np.integer)):
        raise TypeError("top_k 必须为整数。")
    if top_k < 0:
        raise ValueError("top_k 不能为负数。")
    matrix, names = _build_confusion_matrix(y_true, y_pred, class_names)
    if top_k == 0:
        return []

    # 单独复制矩阵并清零对角线，正确分类不会进入误分类统计。
    errors = matrix.copy()
    np.fill_diagonal(errors, 0)
    pairs = np.argwhere(errors > 0)
    if len(pairs) == 0:
        return []
    counts = errors[pairs[:, 0], pairs[:, 1]]
    # 稳定排序使同次数结果也具有确定的返回顺序。
    order = np.argsort(-counts, kind="stable")[:top_k]
    result: List[ConfusionPair] = []
    for index in order:
        row, column = pairs[index]
        result.append(
            {
                "true_class": names[int(row)],
                "predicted_class": names[int(column)],
                "count": int(errors[row, column]),
            }
        )
    return result
