# LGF-EEG-Emotion

[English][en] | 简体中文

**LGF-EEG-Emotion** 是一个基于 PyTorch 实现的**局部-全局特征融合框架**，用于在**留一被试交叉验证（Leave-One-Subject-Out, LOSO）**协议下进行**被试无关的 EEG 情绪识别**。

---

## 项目概述

跨被试 EEG 情绪识别具有很强的被试间差异和 EEG 信号的非平稳性，因此极具挑战性。本项目提出了一种**局部-全局特征融合框架**，整合以下三类特征：

- **局部特征**：通道级 EEG 表征
- **全局特征**：基于图连接的描述符
- **辅助特征**：多重分形特征，用于捕捉长程时序动态

模型采用基于 Transformer 的架构，以有效融合异构模态信息，提升被试无关泛化能力。

---

## 亮点与成果

### 论文与认可

- 📄 论文被 **IEEE Engineering in Medicine and Biology Conference（EMBC）** 接收为**口头报告（Oral Presentation）**。
- **Local–Global Feature Fusion for Subject-Independent EEG Emotion Recognition**  
  预印本：https://arxiv.org/abs/2601.08094

### 核心亮点

- **局部-全局特征融合**：结合通道级 EEG、图连接特征和多重分形特征。
- **被试无关评估**：严格遵循 LOSO 协议。
- 采用 **Transformer 架构**融合异构模态。
- **完整的 PyTorch 流水线**：特征提取 → 模型训练 → 评估。

---

## 项目结构

```
LGF-EEG-Emotion/
│
├── datasets/
│ └── dataset_MAET.py
│
├── feature_extraction/
│ ├── extract_EEG_features.py
│ ├── extract_aux25_features.py
│ ├── compute_graph_features_per_trial.py
│ └── multifractal_features.py
│
├── preprocessing/
│ ├── aggregate_npz_to_trial25.py
│ └── generate_labels_csv.py
│
├── models/
│ └── MAET_model.py
│
├── training/
│ ├── train_MAET_LOSO.py
│ └── train_baselines_LOSO.py
│
├── evaluation/
│ ├── compute_loso_mean_confusion_and_metrics.py
│ ├── infer_metrics_from_checkpoints.py
│ ├── subject_accuracy_bar.py
│ └── summarize_baselines.py
│
├── .gitignore
├── LICENSE
└── README.md
```

---

## 数据集

本代码库面向**多被试 EEG 情绪数据集**设计（例如 **SEED-VII** 或类似基准数据集）。

⚠️ **重要说明**：
- 本仓库**不包含原始 EEG 数据、提取好的特征、标签或预训练模型**。
- 由于许可和隐私限制，用户需自行获取并预处理数据集。
- 目录路径和文件格式需根据实际情况调整。

---

## 特征提取

该框架支持多种互补特征类型：

### 局部 EEG 特征
- 从滑动窗口中提取
- 通道级表征（例如差分熵 Differential Entropy）

### 全局图特征
- 将 EEG 通道视为节点
- 按试次计算连接描述符

### 多重分形特征
- 捕捉非线性和长程时序动态
- 作为辅助信息以增强鲁棒性

所有窗口级特征在模型训练前聚合为**试次级表征**。

---

## 训练

### 提出的模型（LOSO）

```bash
python training/train_MAET_LOSO.py
```

### 基线模型（LOSO）

```bash
python training/train_baselines_LOSO.py
```

每次训练运行：
- 留出一名被试作为测试集
- 在其余被试上训练
- 对所有被试重复上述过程

---

## 评估

计算聚合指标和混淆矩阵：

```bash
python evaluation/compute_loso_mean_confusion_and_metrics.py
```

---

## 依赖

```bash
pip install -r requirements.txt
```

---

## 可复现性指南

为确保公平且可复现的结果：
- 为所有实验固定随机种子
- 在所有被试间使用一致的窗口长度和步长
- 采用相同的预处理和特征提取流水线
- 严格在 LOSO 协议下评估

---

## 引用

如果你认为本仓库对你的研究有帮助，请考虑引用。论文正式发表后将提供 BibTeX 条目。

---

## 致谢

- 本实现作为一项学术研究项目开发完成。
- 部分设计灵感来源于开源的基于 Transformer 的架构。

---

## 联系方式

如有问题、建议或发现 bug，请在 GitHub 上提交 issue。

[en]: README.md
