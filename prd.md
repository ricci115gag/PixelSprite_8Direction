# PRD: Down-to-Left Pixel Sprite Direction Transfer

## 1. Goal

先训练一个固定方向模型：

```text
down sprite -> left sprite
```

当前阶段不做连续角度、不做 4 向/8 向条件模型、不接入 LPC。只用现有 Octopath 配对素材，验证一个更适合像素角色转方向的 adversarial baseline。

核心假设：

- 纯 MSE 容易得到模糊、平均化、缺少结构判断的结果。
- 转方向任务有很强的视觉约束：轮廓、头身比例、服装颜色、透明边界、像素风格、角色身份一致性。
- GAN 的 discriminator 可以学习“down-left 配对是否像真实方向转换”，比单点像素误差更接近任务本身。
- 采用带有 Skip Connections 的 U-Net 生成器直接预测完整 RGBA 目标图像，能够有效保留低频细节，避免残差相加（residual add）引起的“消色鬼影”和边缘羽化问题。

## 2. Dataset Assumption

当前训练样本：

```text
TrainningData/Octopath/<character_id>/
  *_down.png
  *_left.png
```

当前项目中统一叫：

```text
down
left
```

虽然旧 8-direction 项目里这批 `left` 语义接近 `left_down`，但当前模型不暴露这个历史命名。

训练 pair：

```text
source = down (RGBA)
target = left (RGBA)
```

生成图像：

```text
fake_left = G(down)  # G直接预测完整的RGBA目标图像
```

输出时通过 Sigmoid 激活函数归一化到 [0, 1] 范围。

## 3. Model Family

当前建议从 conditional GAN 开始：

```text
down image
  -> Generator (U-Net with Attention Bottleneck)
  -> fake left = G(down)

real pair: concat(down, real left)
fake pair: concat(down, fake left)
  -> Discriminator
  -> true/fake confidence for target direction conversion
```

## 4. Discriminator Design

### 4.1 Input

D 接受配对信息，不只看单张图。

第一版输入：

```text
concat(down, target)
```

通道：

```text
down RGBA:      4 channels
target RGBA:    4 channels  (real left 或 fake left)
total:          8 channels
```

### 4.2 Output

二元分布输出（输出大小为 1，经由 Sigmoid 归一化，代表真实样本的概率）：

```text
D output = 1 (probability of being real)
```

### 4.3 D Architecture

口述结构：

```text
input: concat(down, residual)
  -> shallow conv stem
  -> downsample block
  -> downsample block
  -> residual conv loop
  -> global pooling or final spatial squeeze
  -> binary head (1 logit)
```

### 4.4 D Label

真实样本标签为 `1.0`（Real），伪造样本标签为 `0.0`（Fake）。

真实样本：

```text
D(concat(down, real_left)) -> target label: 1.0
```

伪造样本：

```text
D(concat(down, fake_left)) -> target label: 0.0
```

使用标准的二元交叉熵损失（BCELoss）进行训练。


## 5. Generator Design

### 5.1 My Recommendation

第一版 G 采用标准的 U-Net 结构，包含 Skip Connections（跳跃连接）直接输出完整的 target RGBA 图像，并在 Bottleneck 处保留 Multi-Axis Attention（纵向、横向和全局自注意力）。

为什么：

- down 和 left 是同一角色，颜色、材质、透明区域、头身比例大量共享。
- Skip connections 可以将 Encoder 的低级特征（高频细节、精确像素位置、色彩）直接拉到 Decoder 端，防止画面模糊和边缘羽化，同时避免了相加残差（residual add）引起的“消色鬼影”问题。
- 自注意力机制在低分辨率的 Bottleneck（16x24）运行，计算高效，且能够捕捉到旋转时的大跨度结构关联。

### 5.2 G Structure

口述结构：

```text
input down RGBA (32x48)
  -> Encoder L0: Conv3x3 (out_channels=32)
     -> Skip Connection 1 (32x48)
  -> Encoder L1: Conv3x3 Stride=2 (out_channels=64)
     -> Skip Connection 2 (16x24)
  -> Bottleneck (16x24, channels=128):
     -> Conv3x3 to 128 channels
     -> ResBlocks (x2)
     -> Horizontal + Vertical + Global Attention (並行相加)
     -> ResBlocks (x2)
  -> Decoder L1 (16x24):
     -> Concat with Skip Connection 2 (128 + 64 = 192 channels)
     -> Conv3x3 (out_channels=128)
     -> Upsample (PixelShuffle(2)) (spatially 32x48, out_channels=32)
  -> Decoder L0 (32x48):
     -> Concat with Skip Connection 1 (32 + 32 = 64 channels)
     -> Conv3x3 (out_channels=32)
     -> Final Conv1x1 (out_channels=4)
     -> Sigmoid -> fake left RGBA (32x48)
```

无序结构表：

| Stage | Purpose |
| --- | --- |
| Encoder L0/L1 | 提取浅层高频细节，保存精确色彩与边界位置，提供 Skip 连接 |
| Bottleneck | 学习视角转换带来的非线性形变与横向/纵向位移（注意力机制） |
| Skip Connections | 传递底层细节，确保角色身份、原画质和透明度边界的保真度，消除羽化现象 |
| Decoder L1/L0 | 融合高低频特征，重建并恢复至原始图像大小 |
| Output Layer | 输出合法的 RGBA 全尺寸 `left` 图像 |

### 5.3 G Output Options

建议第一版：

```text
G 直接预测完整的 target RGBA 图像。
fake_left = G(down) (经 Sigmoid 激活到 [0, 1] 范围)
```


## 6. Loss Design

```text
G loss =
  reconstruction loss
  + adversarial loss
  + Feature Matching loss (中间特征匹配)
  + Alpha Binarization loss (Alpha二值化约束)
```

### 6.1 Reconstruction

使用 L1 距离作为重建损失，比 MSE 更能防止像素画产生模糊的边缘和混色：

```text
L_recon = L1(fake_left, real_left)
```

### 6.2 Adversarial

使用标准的二元交叉熵损失（BCELoss），通过全局信息聚合（Global Squeeze）的判别器进行博弈：

- **D Loss**：
  ```text
  D(concat(down, real_left)) -> 1.0 (Real)
  D(concat(down, fake_left)) -> 0.0 (Fake)
  ```
- **G Loss**：
  ```text
  D(concat(down, fake_left)) -> 1.0 (Real)
  ```

### 6.3 Feature Matching Loss

为了使生成器能更好地重建局部细节并稳定对抗训练，引入特征匹配损失（Feature Matching Loss）。它度量 real_pair 和 fake_pair 在判别器 $D$ 的各中间层激活特征图上的 $L_1$ 距离：

```text
L_fm = sum_{i} mean( | D_i(real_pair).detach() - D_i(fake_pair) | )
```

其中 $D_i$ 为判别器第 $i$ 层的特征输出，对其 `real_pair` 的特征采用 `.detach()`，确保梯度只传导至生成器 $G$。

### 6.4 Alpha Binarization Loss

像素画的 alpha 通道原则上应是明确的二值状态（0.0 或 1.0），而生成器通过 Sigmoid 容易输出大量半透明（例如 0.5）的模糊边缘。为了惩罚半透明像素、使边缘干净利落，引入 Alpha 二值化损失：

```text
L_alpha = mean( alpha * (1.0 - alpha) )
```

其中 $\alpha$ 为生成 RGBA 图像的 alpha 通道。当 $\alpha \in \{0, 1\}$ 时，损失为 0，当 $\alpha = 0.5$ 时损失最大。

### 6.5 Training Recipe

最终的联合优化目标：

```text
G loss = L_recon + lambda_adv * L_gan + lambda_fm * L_fm + lambda_alpha * L_alpha
D loss = ( BCE(D_real, 1) + BCE(D_fake, 0) ) / 2
```

默认超参数权重：
- `lambda_adv` = 0.01
- `lambda_fm` = 0.1
- `lambda_alpha` = 0.1

## 7. Training Loop

固定任务：

```text
down -> left
```

每个 step：

```text
1. load down, left (real_left)
2. fake_left = G(down)

3. train D:
   D(concat(down, real_left)) -> label: 1.0
   D(concat(down, fake_left.detach())) -> label: 0.0
   loss_d = (loss_d_real + loss_d_fake) / 2

4. train G:
   reconstruction loss = L1(fake_left, real_left)
   adversarial loss = BCE(D(concat(down, fake_left)), 1.0)
   loss_g = reconstruction loss + lambda_adv * adversarial loss
```

## 8. Evaluation

保存固定样本 contact sheet：

```text
down | fake left | real left
```

重点肉眼检查：

- 角色身份是否保持。
- 轮廓是否像 left。
- alpha 边界是否干净。
- 颜色是否漂移。
- 是否产生 GAN 噪点。
- 是否过度复制 down，没有真正转向。

## 9. Risks

- 数据只有两个方向，G 可能学成“记忆角色局部变化”而不是可泛化转向。
- GAN 数据少时容易不稳定。
- residual 如果直接加 RGBA，alpha 可能出现边界脏点。
- D 输出 7 个方向槽位目前只有一个 active slot，未来扩展时要小心 inactive slots 的训练策略。

## 10. Non-Goals

当前不做：

- 连续角度建模。
- RoPE。
- direction embedding。
- 4-direction/8-direction 条件模型。
- LPC 数据训练。
- 大模型/扩散模型。

