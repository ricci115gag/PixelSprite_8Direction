# PixelSprite_4Direction Developer Notes

> [!CAUTION]
> **CRITICAL AGENT CONSTRAINT**: **DO NOT RUN ANY TRAINING SCRIPTS (like `Train.py`) UNDER ANY CIRCUMSTANCES. The USER has strictly forbidden the AI agent from running model training. This rule must be strictly followed.**

当前项目先用已有 Octopath 两方向素材训练一个方向迁移 baseline。LPC 仓库暂时只作为后续 4-direction 数据管线参考，不进入当前训练。

## 数据约定

主数据目录：

```text
TrainningData/Octopath/
```

当前数据：

- 角色目录数：334
- PNG 数量：668
- 方向名：`down` 与 `left`
- 图片尺寸：`32x48`
- 像素格式：`Format32bppArgb`

旧 8-direction 项目里这批 `left` 对应过 `left_down` 视角；当前 4-direction 项目统一叫 `left`，代码、文件名和报告都保持这个名字。

## MVC 结构

项目源码现在收敛到 `pixel_sprite/`，按 MVC 划分：

```text
pixel_sprite/
  model/
    GAN/
      discriminator.py
      generator.py
    GANv2/
      discriminatorv2.py
      generatorv2.py
    blocks.py
    dataset.py
    factory.py
    image_ops.py
    loss.py
    network.py
  controller/
    extract.py
    infer.py
    report.py
    split.py
    train.py
  view/
    gradio_app.py
  config.py
```

### model

`model/` 放领域对象和核心算法，支持多模型版本管理：

- `factory.py`
  - 模型工厂 `get_model_class`，根据指定的 version (1 或 2) 获取对应的 Generator 或 Discriminator 类。
- `GAN/` (v1初代模型)
  - `generator.py` (UNetGenerator): 标准 U-Net 结构，包含 Skip Connections 并直接输出完整的 target RGBA 图像，在 Bottleneck 处使用 Multi-Axis Attention 自注意力。
  - `discriminator.py` (Discriminator): 接受 concat(down, target) 图像输入（8 通道），经过卷积下采样和 Adaptive Average Pooling 后用 Sigmoid 预测二元分布。
- `GANv2/` (v2改进版模型)
  - `generatorv2.py` (UNetGenerator): v2 生成器。采用与 V1 相同的标准 U-Net 连续表征结构（去除了 Soft Palette Selector/画笔法），通过 Sigmoid 激活直接输出完整的 RGBA 图像，以维持连续的像素分布空间，确保 AI 能正常学习。
  - `discriminatorv2.py` (Discriminator): v2 判别器。采用 **PatchGAN** 架构。
    - **全卷积 Patch 分类**：摒弃全局池化，采用 $4$ 层 Spectral Normalization (谱归一化) 卷积层，最后一层卷积输出 $6 \times 4$ 的 Patch 局部真假判别概率网格 `[B, 1, 6, 4]`。
    - **Patch 级别 Loss 计算**：直接将输出的 Patch 概率网格送入 BCE Loss 计算（使用 `torch.ones_like` 和 `torch.zeros_like`），确保每个局部的真假都能受到精确惩罚，避免全局均值稀释梯度。
- `dataset.py`
  - `DirectionPairDataset`
  - 从角色目录中读取 `down/left` 图片，生成有向 source-target pairs。
- `network.py`
  - 存放注意力机制 block（`HorizontalAxisAttention`, `VerticalAxisAttention`, `GlobalAttention`）。
- `blocks.py`
  - 条件残差块（`ResBlock`）。
- `loss.py`
  - 包含自定义损失函数定义（例如 `PixelArtLoss`）。
- `image_ops.py`
  - palette 映射、alpha 二值化、透明区域 RGB 清理。

### controller

`controller/` 放流程编排：

- `train.py`
  - 训练流程。
  - 按角色划分 train/val。
  - 记录 `train_loss` 和 `val_loss`。
  - 周期性保存 source/pred/target 样本图。
- `infer.py`
  - 加载模型并对单张图片推理。
- `extract.py`
  - 从 Octopath 原始资源裁出 `down/left` 训练图。
- `report.py`
  - 生成数据集统计。
- `split.py`
  - 按角色划分 dataset，避免数据泄漏。

### view

`view/` 只放 UI：

- `gradio_app.py`
  - Gradio 界面。
  - 不直接包含模型细节，只调用 controller。

## 顶层入口

顶层文件只作为核心的主程序运行入口：

- `Train.py`
  - 调用 `pixel_sprite.controller.train.train_model` 启动模型训练。
- `app.py`
  - 启动 Gradio 推理与可视化界面。

## 关于 scripts

`scripts/` 不再作为源码目录使用。之前模型代码放在 `scripts/` 是命名错误；模型代码现在已经移动到 `pixel_sprite/model/`。

后续如果重新创建 `scripts/`，它只能放一次性维护脚本或 repo 工具，不放可 import 的业务/模型代码。

## 常用命令

报告数据集：

```bash
python -m pixel_sprite.controller.report
```

保存数据集报告：

```bash
python -m pixel_sprite.controller.report --json reports/dataset_report.json
```

提取原始素材为训练对：

```bash
python -m pixel_sprite.controller.extract <root_dir> <output_dir>
```

训练：

```bash
python Train.py
```

启动推理 UI：

```bash
python app.py
```


## 当前风险

- 当前只有二方向数据，不能期望模型可靠泛化到四向或八向。
- `left` 是当前项目约定名，来源语义对应旧项目的左下视角。
- 当前模型不使用 RoPE、连续角度 sinusoidal embedding 或离散方向 embedding。两方向孤立数据不足以学习连续视角插值；当前 baseline 只学习输入视角到另一视角的映射。
- 默认启用 train/val split；独立 test split 暂未启用。
- `TrainningData/`、`PaperAssets/`、`models/`、`reports/` 等本地数据已加入 Git 和 Seafile ignore。
- 当前环境中功能训练未实际跑；本次主要做架构整理和 Python 编译检查。

## 下一步建议

1. 跑 `python -m pixel_sprite.controller.report` 确认 pair 统计。
2. 开始二方向 conditional GAN baseline 训练。
3. 查看 `models/.../samples/epoch_*.png`，肉眼检查 source (down)/pred (fake left)/target (real left)。
4. 视训练效果决定是否加入 test split。
5. 再考虑接入 LPC 4-direction 管线。
