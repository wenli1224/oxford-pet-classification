"""用于 Oxford-IIIT Pet 分类的 ResNet-18 模型定义。"""

from __future__ import annotations

from torch import nn
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.models.resnet import ResNet


def create_model(
    num_classes: int = 37,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> ResNet:
    """创建 ResNet-18，并替换最终分类层。

    Args:
        num_classes: 输出类别数量，默认 37。
        pretrained: 是否加载 ImageNet 预训练权重，默认 True。
        freeze_backbone: 是否冻结除 fc 层外的参数，默认 False。

    Returns:
        输出形状为 [batch_size, num_classes] 的 ResNet-18 模型。
        输出为未经 Softmax 的 logits，可直接用于 CrossEntropyLoss。
    """
    if num_classes <= 0:
        raise ValueError("num_classes 必须大于 0。")

    # 使用 torchvision 的权重枚举；关闭预训练时不加载或下载权重。
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)

    # 先加载完整的预训练模型，再将 ImageNet 分类层替换为目标类别数。
    in_features: int = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    # 冻结模式下先关闭全部参数的梯度；否则显式启用全部参数的梯度。
    for parameter in model.parameters():
        parameter.requires_grad_(not freeze_backbone)

    # 新的 fc 层在两种模式下均参与训练。
    for parameter in model.fc.parameters():
        parameter.requires_grad_(True)

    return model
