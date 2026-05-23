# PixelSprite_4Direction

基于 Conditional GAN (cGAN) 实现的像素角色视角转换工具，目前支持将下视角（`down`）的角色精灵图转换为左视角（`left`）。

---

## 核心特性

- **结构与细节保留**：生成器使用标准 U-Net 结构，配合跳跃连接（Skip Connections）直接生成 RGBA 图像，保留高频细节并防止边缘羽化。
- **全局自注意力**：在 Bottleneck 处引入 Multi-Axis Attention（横向、纵向及全局自注意力），捕捉旋转时的宏观结构变化。
- **像素画优化 Loss**：集成 L1 重建损失、对抗损失、特征匹配损失（Feature Matching Loss）以及专门针对像素画边缘的 **Alpha 二值化惩罚损失**（防止半透明虚边）。
- **鲁棒的训练持久化**：
  - 支持 `safetensors` 格式保存，避免 pickle 安全风险。
  - 自动记录训练步数 `global_step` 与时间戳至模型元数据。
  - 写入时采用临时文件占位，写完后原子替换（Write-then-Replace），规避中断写入导致的文件损坏。
  - 支持 `Ctrl+C` 安全中断，自动在退出前保存最新权重。
- **Gradio 交互界面**：提供图像推理转换以及模型的动态热重载功能。

---

## 项目结构

```text
├── pixel_sprite/
│   ├── model/           # 核心模型与算法层 (U-Net, Discriminator, Loss, Dataset)
│   ├── controller/      # 任务编排层 (训练、推理、数据集提取与划分)
│   └── view/            # UI 界面 (Gradio App)
├── TrainningData/       # 训练数据集存放目录
├── config.yaml          # 模型及训练超参数配置
├── Train.py             # 训练入口脚本
└── app.py               # Gradio 推理及管理入口
```

---

## 快速开始

### 1. 数据准备
确保数据存放在 `TrainningData/Octopath/` 目录下，并以角色分类存放：
```text
TrainningData/Octopath/<character_id>/
  ├── *_down.png
  └── *_left.png
```

运行以下命令验证并生成数据集统计报告：
```bash
python -m pixel_sprite.controller.report
```

### 2. 模型训练
修改 `config.yaml` 调整超参数，然后执行：
```bash
python Train.py
```
- 模型权重默认以时间戳和 `global_step` 命名并保存在 `./checkpoints/` 目录下。
- TensorBoard 日志保存在 `./runs/run_YYYYMMDD_HHMMSS/` 下。启动 TensorBoard：
  ```bash
  tensorboard --logdir runs
  ```
- 训练期间每 5 秒会自动刷新并输出一张预测样本对比图（`Generated_Image`）到 TensorBoard 中。

### 3. 可视化推理与模型管理
启动 Gradio 交互面板：
```bash
python app.py
```
- **Direction Transfer**: 填入本地待预测图片路径，选择输入和输出方向即可生成转换图。
- **Model Settings**: 自动扫描 `./checkpoints/` 下的所有模型文件，支持下拉选择并**一键热重载**模型权重，无需重启服务。
