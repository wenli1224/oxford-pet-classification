# Oxford-IIIT Pet 细粒度分类项目

## 1. 项目简介

本项目使用 **Oxford-IIIT Pet** 数据集进行宠物品种的细粒度图像分类。合并官方 `trainval` 和 `test` 标注中的 **7349 张图片**，共有 **37 个类别**，通过 **ImageNet 预训练 ResNet-18 的迁移学习**完成分类。

细粒度分类需要区分外观相似的品种。本项目比较全参数微调的 Baseline、Label Smoothing 和 Freeze Backbone 三种设置，并使用混淆矩阵与 Grad-CAM 分析模型的预测表现。

## 2. 快速复现

使用 Python 3.10 或 3.11，在 `pet_project` 项目根目录依次执行：

```bash
pip install -r requirements.txt
python train.py
python evaluate.py --model_path ./outputs/best_model.pth
```

首次训练会自动下载 Oxford-IIIT Pet 数据集和 ImageNet 预训练权重，需要联网。已下载的数据会复用。脚本自动选择 CUDA；CUDA 不可用时使用 CPU，并自动创建输出目录。

训练完成后，模型与历史记录保存在 `./outputs`；测试指标与混淆矩阵保存在 `./outputs/evaluation`。查看运行参数可使用：

```bash
python train.py --help
python evaluate.py --help
```

## 3. 实验设置

### 数据划分与预处理

合并两个官方划分后，使用 `sklearn.model_selection.train_test_split` 按品种标签进行 **Stratified Split（分层划分）**。两次划分均固定 `random_state=42`，训练集、验证集和测试集互不重叠。

| 划分 | 比例 | 图片数量 | 预处理 |
| --- | --- | --- | --- |
| 训练集 | 70% | 5144 | `Resize(256)` → `RandomCrop(224)` → `RandomHorizontalFlip()` → `ToTensor()` → `Normalize` |
| 验证集 | 15% | 1102 | `Resize(256)` → `CenterCrop(224)` → `ToTensor()` → `Normalize` |
| 测试集 | 15% | 1103 | `Resize(256)` → `CenterCrop(224)` → `ToTensor()` → `Normalize` |

样本数按划分算法取整，合计 7349 张。分层划分保留各类别的大致比例。`Resize(256)` 将较短边缩放至 256，随后裁剪得到 224×224 的输入。

标准化使用 ImageNet 参数：`mean=(0.485, 0.456, 0.406)`、`std=(0.229, 0.224, 0.225)`。类别标签为 `0～36`，`class_names[label]` 对应相应品种。训练集启用 shuffle，验证集和测试集保持固定顺序；DataLoader 默认使用 2 个 worker。

验证集用于选择最佳模型，测试集仅用于最终评估。本项目采用重新划分后的测试集，结果应在这一划分设置下比较。

### Baseline 配置

| 配置项 | 设置 |
| --- | --- |
| 模型 | ResNet-18 |
| 初始权重 | ImageNet 预训练权重 `ResNet18_Weights.DEFAULT` |
| 分类层 | 将 `fc` 替换为 37 类输出 |
| 微调方式 | 全部参数参与训练 |
| 优化器 | AdamW |
| 学习率 | `lr=1e-4` |
| 权重衰减 | `weight_decay=1e-4` |
| 损失函数 | `CrossEntropyLoss`，默认 `label_smoothing=0.0` |
| 批大小 | `batch_size=32` |
| 训练轮数 | `epochs=10` |
| 随机种子 | `seed=42` |
| 模型选择 | 保存验证准确率最高的模型 |

模型输出未经 Softmax 的 logits，直接用于交叉熵损失；测试时取最大 logit 对应的类别作为 Top-1 预测。验证和测试均使用 `model.eval()` 与 `torch.no_grad()`。

## 4. 项目结构

```text
pet_project/
├── data/dataset.py
├── models/model.py
├── utils/metrics.py
├── utils/gradcam.py
├── train.py
├── evaluate.py
├── requirements.txt
└── README.md
```

| 文件 | 功能 |
| --- | --- |
| `data/dataset.py` | 自动下载数据、合并官方划分、分层划分及创建 DataLoader |
| `models/model.py` | 创建 ResNet-18，替换分类层并支持冻结骨干参数 |
| `utils/metrics.py` | 计算 Accuracy、Macro-F1，绘制完整混淆矩阵并统计主要误分类类别对 |
| `utils/gradcam.py` | 使用原生 PyTorch hooks 生成 Grad-CAM 热力图并保存叠加图 |
| `train.py` | 训练、验证、选择最佳模型及记录历史和 TensorBoard 日志 |
| `evaluate.py` | 加载训练权重，在测试集上评估并导出结果 |
| `requirements.txt` | 项目依赖列表 |

## 5. 环境安装

项目依赖包括 `torch`、`torchvision`、`numpy`、`Pillow`、`scikit-learn`、`matplotlib` 和 `tensorboard`。

```bash
pip install -r requirements.txt
```

可先创建独立的虚拟环境：

```bash
python -m venv .venv
```

Windows PowerShell 激活命令：

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux/macOS 激活命令：

```bash
source .venv/bin/activate
```

激活后再执行依赖安装命令。依赖列表未固定具体版本；复现实验时建议记录 Python、依赖版本和运行硬件，便于核对结果。

## 6. 训练实验

### Baseline：全参数微调

```bash
python train.py
```

使用第 3 节中的默认配置，输出目录为 `./outputs`。每轮打印 Train Loss、Train Accuracy、Val Loss 和 Val Accuracy；训练结束打印最佳验证准确率、总训练时间和最佳模型路径。

### Label Smoothing 0.1

```bash
python train.py --label_smoothing 0.1 --output_dir ./outputs/label_smoothing
```

使用 `nn.CrossEntropyLoss(label_smoothing=0.1)`，其余训练设置与 Baseline 相同。

### Freeze Backbone

```bash
python train.py --freeze_backbone --output_dir ./outputs/freeze_backbone
```

冻结除 `fc` 外的全部参数，仅训练新的 37 分类 `fc` 层，其余超参数与 Baseline 相同。

### 训练参数

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--epochs` | `10` | 训练轮数 |
| `--batch_size` | `32` | 每批样本数量 |
| `--lr` | `1e-4` | AdamW 学习率 |
| `--label_smoothing` | `0.0` | 标签平滑系数 |
| `--freeze_backbone` | 默认关闭 | 加入该开关后仅训练 `fc` 参数 |
| `--data_root` | `./data` | 数据下载根目录 |
| `--output_dir` | `./outputs` | 模型、历史及日志保存目录 |

## 7. 测试集评估

### 评估 Baseline

```bash
python evaluate.py --model_path ./outputs/best_model.pth
```

### 评估 Label Smoothing

```bash
python evaluate.py --model_path ./outputs/label_smoothing/best_model.pth --output_dir ./outputs/label_smoothing/evaluation
```

### 评估 Freeze Backbone

```bash
python evaluate.py --model_path ./outputs/freeze_backbone/best_model.pth --output_dir ./outputs/freeze_backbone/evaluation
```

各实验分别保存评估结果。评估时复用训练阶段的 `seed=42` 分层划分，并加载已训练的完整权重。

终端输出 Test Top-1 Accuracy、Test Macro-F1、测试样本数量，以及误分类次数最多的前 10 组“真实类别 → 预测类别”。如果实际误分类类别对不足 10 组，只输出存在的类别对。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--model_path` | `./outputs/best_model.pth` | 待评估的模型权重 |
| `--data_root` | `./data` | 数据下载根目录 |
| `--batch_size` | `32` | 测试批大小 |
| `--output_dir` | `./outputs/evaluation` | 测试结果保存目录 |

Top-1 Accuracy 是预测正确的样本比例；Macro-F1 对各类别的 F1 等权平均。测试集包含全部 37 个类别。`metrics.json` 中的准确率和 Macro-F1 使用 `0～1` 的数值表示。

## 8. 真实实验结果与结论

以下数值来自项目作者提供的实际运行记录。

| 实验 | Test Accuracy | Macro-F1 |
| --- | --- | --- |
| Baseline | **91.75%** | **0.9172** |
| Label Smoothing 0.1 | 91.66% | 0.9161 |
| Freeze Backbone | 88.30% | 0.8814 |

Baseline 的 **Best Val Accuracy 为 93.01%**，**Training Time 为 3.59 min**。训练时间为该次实验记录，具体用时随运行硬件而变化。

- **Baseline 表现最好**：在本次固定随机种子的三个实验中，测试准确率和 Macro-F1 均最高。
- **Label Smoothing 基本无明显收益**：相较 Baseline，测试准确率降低 0.09 个百分点，Macro-F1 也略有下降。
- **Freeze Backbone 性能明显下降**：相较 Baseline，测试准确率降低 3.45 个百分点，说明在当前细粒度分类任务与实验设置下，全参数微调更适合学习宠物品种间的细微差异。

## 9. 输出文件与日志

| 输出内容 | Baseline 默认路径 | 说明 |
| --- | --- | --- |
| 最佳模型 | `outputs/best_model.pth` | 验证准确率最高模型的纯 `state_dict` |
| 训练历史 | `outputs/history.json` | 逐轮记录 epoch、训练/验证损失及准确率 |
| TensorBoard 日志 | `outputs/tensorboard/` | `Loss/Train`、`Loss/Validation`、`Accuracy/Train`、`Accuracy/Validation` |
| 混淆矩阵 | `outputs/evaluation/confusion_matrix.png` | 完整 37×37 矩阵，完整类别名称，300 dpi |
| 评估结果 | `outputs/evaluation/metrics.json` | `test_accuracy`、`macro_f1`、`num_test_samples`、`top_confusions` |
| Grad-CAM 结果图 | `outputs/evaluation/gradcam_example.png` | 按第 10 节示例单独生成的原图与热力图叠加图 |

混淆矩阵的横轴为 Predicted Label，纵轴为 True Label，横轴类别名称旋转 90 度。主要混淆类别对的统计忽略对角线及零计数，按误分类次数降序排列。

查看 Baseline 的 TensorBoard 曲线：

```bash
tensorboard --logdir ./outputs/tensorboard
```

打开终端显示的访问地址。查看三个实验的日志可使用：

```bash
tensorboard --logdir ./outputs
```

## 10. Grad-CAM 可视化

Grad-CAM 图通过 `GradCAM` 和 `save_gradcam()` 单独生成，默认解释层为 ResNet-18 的 `model.layer4[-1]`。工具返回归一化到 `0～1` 的 224×224 热力图，并使用 ImageNet 参数反标准化原图，以 `jet` 配色和 `alpha=0.45` 叠加，标题显示 True 与 Pred，保存分辨率为 300 dpi。

以下示例可在项目根目录保存为临时 Python 脚本运行，生成测试集第一张图片的可视化：

```python
import matplotlib

matplotlib.use("Agg")
import torch

from data.dataset import get_dataloaders
from models.model import create_model
from utils.gradcam import GradCAM, save_gradcam


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(num_classes=37, pretrained=False).to(device)
    state_dict = torch.load(
        "./outputs/best_model.pth", map_location=device, weights_only=True
    )
    model.load_state_dict(state_dict)
    model.eval()

    _, _, test_loader, class_names = get_dataloaders(
        root="./data", batch_size=1, num_workers=0
    )
    images, labels = next(iter(test_loader))
    images = images.to(device)
    true_idx = int(labels[0].item())
    with torch.no_grad():
        pred_idx = int(model(images).argmax(dim=1).item())

    # 生成时工具会启用梯度；上下文管理器在结束时自动移除 hooks。
    with GradCAM(model) as cam:
        heatmap = cam.generate(images, class_idx=pred_idx)
    save_gradcam(
        image_tensor=images,
        heatmap=heatmap,
        true_class=class_names[true_idx],
        pred_class=class_names[pred_idx],
        save_path="./outputs/evaluation/gradcam_example.png",
    )
    print("已保存 ./outputs/evaluation/gradcam_example.png")


if __name__ == "__main__":
    main()
```

省略 `class_idx` 时自动解释 Top-1 预测类别；传入 `true_idx` 可观察真实类别对应的关注区域。

## 11. AI辅助编程声明

本项目使用 AI 辅助生成部分代码、协助排查错误，以及整理项目结构和说明文档。实验结果来自项目作者实际运行的模型训练与测试，上表数值依据作者提供的真实实验记录整理，未使用 AI 模拟或生成实验数据。
