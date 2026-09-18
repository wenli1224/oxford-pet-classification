"""使用原生 PyTorch hooks 生成和保存 ResNet-18 的 Grad-CAM。

用法：
    with GradCAM(model) as cam:
        heatmap = cam.generate(image_tensor)
    save_gradcam(image_tensor, heatmap, true_class, pred_class, "gradcam.png")
"""

from __future__ import annotations

from pathlib import Path
from types import TracebackType
from typing import Optional, Tuple, Type, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


HEATMAP_SIZE: Tuple[int, int] = (224, 224)
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)


def _as_single_image(image_tensor: torch.Tensor) -> torch.Tensor:
    """接受 [3, H, W] 或 [1, 3, H, W]，统一为单张图片的批格式。"""
    if not isinstance(image_tensor, torch.Tensor):
        raise TypeError("image_tensor 必须为 PyTorch Tensor。")
    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)
    if (
        image_tensor.ndim != 4
        or image_tensor.shape[0] != 1
        or image_tensor.shape[1] != 3
        or image_tensor.shape[2] == 0
        or image_tensor.shape[3] == 0
    ):
        raise ValueError("image_tensor 的形状必须为 [3, H, W] 或 [1, 3, H, W]。")
    if not image_tensor.is_floating_point():
        raise TypeError("image_tensor 必须为经过 ImageNet Normalize 的浮点张量。")
    return image_tensor


class GradCAM:
    """基于目标类别 logit 的梯度生成 Grad-CAM，不依赖第三方 CAM 包。"""

    def __init__(
        self, model: nn.Module, target_layer: Optional[nn.Module] = None
    ) -> None:
        self.model = model
        if target_layer is None:
            if not hasattr(model, "layer4") or len(model.layer4) == 0:
                raise ValueError("默认目标层要求模型具有非空的 ResNet layer4。")
            target_layer = model.layer4[-1]
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._closed: bool = False

        # forward hook 保存特征图；full backward hook 保存输出特征图的梯度。
        self._forward_handle = self.target_layer.register_forward_hook(
            self._save_activation
        )
        try:
            self._backward_handle = self.target_layer.register_full_backward_hook(
                self._save_gradient
            )
        except Exception:
            self._forward_handle.remove()
            raise

    def _save_activation(
        self,
        module: nn.Module,
        inputs: Tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        self.activations = output.detach()

    def _save_gradient(
        self,
        module: nn.Module,
        grad_input: Tuple[Optional[torch.Tensor], ...],
        grad_output: Tuple[Optional[torch.Tensor], ...],
    ) -> None:
        if grad_output and grad_output[0] is not None:
            self.gradients = grad_output[0].detach()

    def generate(
        self, image_tensor: torch.Tensor, class_idx: Optional[int] = None
    ) -> np.ndarray:
        """返回 float32 的 224×224 热力图，取值范围为 0～1。

        class_idx=None 时解释模型预测的 Top-1 类别，否则解释指定类别。
        输入为单张经过 ImageNet Normalize 的图片，自动匹配模型设备和精度。
        """
        if self._closed:
            raise RuntimeError("GradCAM hooks 已移除，请创建新的 GradCAM 实例。")
        image = _as_single_image(image_tensor)
        if class_idx is not None and not isinstance(class_idx, (int, np.integer)):
            raise TypeError("class_idx 必须为整数类别编号或 None。")

        # 保留每个子模块的状态，避免破坏原模型可能存在的混合 train/eval 设置。
        training_states = [(module, module.training) for module in self.model.modules()]
        self.model.eval()
        self.activations = None
        self.gradients = None

        try:
            # 即使外层处于 no_grad/inference_mode，也为当前计算建立正常梯度图。
            with torch.inference_mode(False), torch.enable_grad():
                parameter = next(self.model.parameters(), None)
                device = parameter.device if parameter is not None else image.device
                dtype = parameter.dtype if parameter is not None else image.dtype
                # 冻结骨干时仍需输入梯度；复制后不会修改调用者图片或其梯度。
                input_tensor = image.detach().to(device=device, dtype=dtype).clone()
                input_tensor.requires_grad_(True)
                logits = self.model(input_tensor)
                if logits.ndim != 2 or logits.shape[0] != 1:
                    raise ValueError("模型必须返回形状为 [1, num_classes] 的 logits。")
                target_idx = (
                    int(logits.argmax(dim=1).item())
                    if class_idx is None
                    else int(class_idx)
                )
                if not 0 <= target_idx < logits.shape[1]:
                    raise ValueError(f"class_idx 必须在 0～{logits.shape[1] - 1} 范围内。")

                # 反向传播只累积到复制的输入，既触发 hook，也不改变参数已有梯度。
                logits[0, target_idx].backward(inputs=[input_tensor])
                if self.activations is None or self.gradients is None:
                    raise RuntimeError("目标层未捕获 activation/gradient，请检查目标层。")
                if self.activations.ndim != 4 or self.gradients.shape != self.activations.shape:
                    raise RuntimeError("目标层必须提供形状相同的四维特征图和梯度。")

                # Grad-CAM：空间平均梯度作为通道权重，加权特征图求和后取 ReLU。
                activation = self.activations.float()
                gradient = self.gradients.float()
                weights = gradient.mean(dim=(2, 3), keepdim=True)
                cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
                cam = F.interpolate(
                    cam, size=HEATMAP_SIZE, mode="bilinear", align_corners=False
                )[0, 0]
                if not bool(torch.isfinite(cam).all().item()):
                    raise RuntimeError("Grad-CAM 出现非有限数值，请检查模型和输入。")
                minimum, maximum = cam.min(), cam.max()
                value_range = maximum - minimum
                # 常量或全零热力图返回全零，避免归一化时除以零。
                if float(value_range.item()) > 0.0:
                    cam = (cam - minimum) / value_range
                else:
                    cam = torch.zeros_like(cam)
                return cam.detach().cpu().numpy()
        finally:
            for module, training in training_states:
                module.training = training

    def close(self) -> None:
        """移除两个 hooks 并释放缓存；可重复调用。"""
        if not self._closed:
            self._forward_handle.remove()
            self._backward_handle.remove()
            self._closed = True
        self.activations = None
        self.gradients = None

    def __enter__(self) -> GradCAM:
        if self._closed:
            raise RuntimeError("GradCAM hooks 已移除。")
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        self.close()


def save_gradcam(
    image_tensor: torch.Tensor,
    heatmap: Union[np.ndarray, torch.Tensor],
    true_class: str,
    pred_class: str,
    save_path: Union[str, Path],
) -> None:
    """反标准化图片，展示原图与 jet 热力图叠加结果，并以 300 dpi 保存。"""
    image = _as_single_image(image_tensor).detach().cpu().float()
    if tuple(image.shape[-2:]) != HEATMAP_SIZE:
        image = F.interpolate(
            image, size=HEATMAP_SIZE, mode="bilinear", align_corners=False
        )
    image = image[0]
    # Normalize 的逆操作：原始像素 = 标准化像素 × std + mean。
    mean = image.new_tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = image.new_tensor(IMAGENET_STD).view(3, 1, 1)
    original = (image * std + mean).clamp(0.0, 1.0)
    rgb_image = original.permute(1, 2, 0).numpy()

    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.detach().cpu().float().numpy()
    heatmap_array = np.asarray(heatmap, dtype=np.float32)
    if heatmap_array.shape != HEATMAP_SIZE:
        raise ValueError("heatmap 的形状必须为 224×224。")
    if not np.isfinite(heatmap_array).all():
        raise ValueError("heatmap 不能包含 NaN 或无穷大。")
    heatmap_array = np.clip(heatmap_array, 0.0, 1.0)

    destination = Path(save_path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(12, 6))
    try:
        axes[0].imshow(rgb_image)
        axes[0].set_title("Original Image")
        axes[0].axis("off")
        axes[1].imshow(rgb_image)
        axes[1].imshow(heatmap_array, cmap="jet", alpha=0.45, vmin=0.0, vmax=1.0)
        axes[1].set_title(f"True: {true_class}\nPred: {pred_class}")
        axes[1].axis("off")
        figure.tight_layout()
        figure.savefig(destination, dpi=300, bbox_inches="tight")
    finally:
        plt.close(figure)
