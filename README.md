# Houston 2013 双流 HSI–LiDAR 分类

本工程使用 Houston 2013 同一场景的高光谱图像（HSI）和 LiDAR 高程二维特征图，完成 15 类地物像素分类。两路 CNN 分别提取 HSI 与 LiDAR 的深度空间特征，在分类层前拼接 128 维和 128 维特征，再通过全连接融合模块输出类别。

## 数据与划分

- `Houston2013/HSI.mat`：`349 × 1905 × 144` 高光谱数据。
- `Houston2013/LiDAR.mat`：`349 × 1905` 高程特征图。
- `Houston2013/TRLabel.mat`：2,832 个训练像素，共 15 类。
- `Houston2013/TSLabel.mat`：12,197 个独立测试像素。
- `Houston2013/gt.mat`：15,029 个有标签像素，用于分类图可视化。

训练集内部再按类别分层抽取 15% 作为验证集，用于早停和模型选择。官方测试集只用于最终评估。

## 预处理

1. 只用训练标签对应的像素拟合 HSI 标准化参数和 PCA，避免测试集泄漏；默认将 144 个波段压缩为 20 个主成分。
2. LiDAR 数值线性归一化到 `[0, 1]`。
3. 以每个目标像素为中心提取同位置的 `21 × 21` HSI/LiDAR 邻域，边界使用反射填充。
4. 训练时对两个模态同步进行随机旋转和翻转。

## 网络结构

```text
HSI (20x21x21) -> HSI CNN encoder -> 128-d feature
                                               \
                                                concat -> FC(256->128) -> 15 classes
                                               /
LiDAR (1x21x21) -> LiDAR CNN encoder -> 128-d feature
```

每个编码器由 6 个 `3x3 Conv + BatchNorm + ReLU` 卷积块、2 次最大池化和全局平均池化组成。融合发生在深度特征层，而不是输入层或决策层。

## 注意力融合版本

`attention_model.py` 保留原来的 HSI/LiDAR 双流 CNN，但不再直接把两个 128 维向量拼接。两路卷积特征图会先投影到统一维度，然后执行双向多头交叉注意力：

```text
HSI CNN feature map  <->  LiDAR CNN feature map
         双向 cross-attention
                   ↓
       注意力池化 + 门控加权融合
                   ↓
             分类器（15 类）
```

该版本会额外保存融合门控值和可选的注意力矩阵，便于分析两种模态在不同空间位置的贡献。

训练：

```bash
cd /home/temp/xiangsi
PYTHON_BIN=/home/temp/miniforge3/envs/cnn/bin/python \
  ./dual_stream_hsi_lidar/scripts/train_attention.sh
```

训练结果默认保存到：

```text
dual_stream_hsi_lidar/runs/attention_fusion/
```

独立测试：

```bash
cd /home/temp/xiangsi
PYTHON_BIN=/home/temp/miniforge3/envs/cnn/bin/python \
  ./dual_stream_hsi_lidar/scripts/test_attention.sh
```

也可以直接指定模型文件：

```bash
PYTHON_BIN=/home/temp/miniforge3/envs/cnn/bin/python \
CHECKPOINT=/path/to/best_model.pt \
  ./dual_stream_hsi_lidar/scripts/test_attention.sh
```

## 融合消融对比图

使用同一次训练划分下的 `fusion`、`hsi_only` 和 `lidar_only` 结果生成总体指标、逐类精度、混淆矩阵差异和训练曲线：

```bash
cd /home/temp/xiangsi
PYTHON_BIN=/home/temp/miniforge3/envs/cnn/bin/python \
  ./dual_stream_hsi_lidar/scripts/compare_ablations.sh
```

图片和分析报告保存在：

```text
dual_stream_hsi_lidar/runs/comparison/
```

## 运行

### 1. 创建并激活环境

以下命令从零创建名为 `houston-dual-cnn` 的 Python 3.10 环境：

```bash
cd /home/temp/xiangsi
conda create -n houston-dual-cnn python=3.10 -y
conda activate houston-dual-cnn
python -m pip install --upgrade pip
```

没有 CUDA GPU 时，安装 CPU 版 PyTorch 和其他依赖：

```bash
python -m pip install torch==2.4.1 \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r dual_stream_hsi_lidar/requirements.txt
```

有 CUDA 12.1 GPU 时，将 PyTorch 安装命令替换为：

```bash
python -m pip install torch==2.4.1 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r dual_stream_hsi_lidar/requirements.txt
```

### 2. 验证环境

```bash
python - <<'PY'
import numpy, scipy, sklearn, matplotlib, torch
print("numpy:", numpy.__version__)
print("scipy:", scipy.__version__)
print("sklearn:", sklearn.__version__)
print("matplotlib:", matplotlib.__version__)
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
PY
```

### 3. 训练融合模型

确认 `Houston2013/` 位于项目根目录，然后执行：

```bash
cd /home/temp/xiangsi
./dual_stream_hsi_lidar/scripts/train.sh
```

也可以指定参数：

```bash
python -m dual_stream_hsi_lidar.train \
  --dataset-dir Houston2013 \
  --output-dir dual_stream_hsi_lidar/runs/fusion \
  --fusion-mode fusion \
  --epochs 40 \
  --patch-size 21 \
  --pca-components 20 \
  --batch-size 128
```

如需强制指定设备：

```bash
python -m dual_stream_hsi_lidar.train \
  --dataset-dir Houston2013 \
  --output-dir dual_stream_hsi_lidar/runs/fusion \
  --device cpu
```

将 `--device cpu` 改为 `--device cuda` 可强制使用 GPU；`--device auto` 会自动选择。

单模态消融实验只需将 `--fusion-mode` 改为 `hsi` 或 `lidar`，并更换输出目录。

### 4. 运行拼接权重扫描

在 HSI 与 LiDAR 的 128 维特征拼接前分别加入权重，并运行 5 组归一化权重和
原始 `1:1` 基线：

```bash
PYTHON_BIN=/home/temp/miniforge3/envs/cnn/bin/python \
  ./dual_stream_hsi_lidar/scripts/run_weight_sweep.sh
```

各次训练位于 `dual_stream_hsi_lidar/runs/weight_sweep/`。汇总表、报告和图片
位于 `dual_stream_hsi_lidar/runs/weight_sweep_comparison/`，其中
`summary.csv` 包含每组的 OA、AA、Kappa 和宏平均 F1。

需要注意，加权拼接后立即进入线性融合头，分支尺度原则上可以被该层参数吸收；
因此该扫描主要衡量权重对优化和正则化的影响。若要严格约束各模态贡献，应使用
加权特征求和、独立分类后的决策融合，或带权重正则的融合头。

### 5. 查看数据内容

生成 HSI 假彩色图、LiDAR 图、标签分布、波段切片、类别平均光谱和类别样本数量：

```bash
./dual_stream_hsi_lidar/scripts/visualize_data.sh
```

自定义用于 RGB 假彩色合成的三个 HSI 波段：

```bash
./dual_stream_hsi_lidar/scripts/visualize_data.sh \
  --rgb-bands 20 60 100 \
  --band-grid 10 30 50 70 90 110 130 144
```

图片保存在 `dual_stream_hsi_lidar/visualizations/`。

也可以一次运行完整融合和两组消融：

```bash
./dual_stream_hsi_lidar/scripts/run_ablations.sh
```

## 输出

运行结果保存在 `dual_stream_hsi_lidar/runs/fusion/`：

- `best_model.pt`：验证集表现最好的模型。
- `metrics.json`：测试集 OA、AA、Kappa、逐类精度和混淆矩阵。
- `test_predictions.npz`：测试像素坐标、真实类别、预测类别和概率。
- `classification_map.npz/.mat/.png`：所有有标签像素的分类预测图。
- `confusion_matrix.png`、`class_accuracy.png`、`training_history.png`：结果图。

## 已复现结果

固定随机种子 `42`，使用官方训练/测试划分和相同的网络宽度；结果如下。

| 输入 | OA | AA | Kappa |
| --- | ---: | ---: | ---: |
| LiDAR only | 71.70% | 72.48% | 0.6930 |
| HSI only | 83.62% | 85.68% | 0.8221 |
| HSI + LiDAR feature fusion | **88.17%** | **89.64%** | **0.8717** |

融合结果位于 `runs/fusion/`。训练集内部验证集与训练像素空间相邻，容易因邻域相关性而过早达到饱和；模型因此使用固定上限和早停规则，最终报告始终采用官方独立测试集。


