# Catalyst Framework — 催化剂跨体系预测框架

> A w↔T↔z Tri-directional Iterative Framework for Catalyst Cross-System Prediction

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

##  简介 | Overview

催化剂研发面临一个核心挑战：**换一个环境（电位、电解质、温度）和换一个催化剂，如何直接预测产率/法拉第效率（FE）？** 传统 DFT 计算耗时长，纯数据驱动模型又缺乏可解释性和跨体系泛化能力。

本项目提出一个**自优化闭环框架**，将打分公式（w）、决策树（T）、隐变量（z）三者耦合迭代，自动发现关键特征-阈值组合，并通过神经网络精确拟合，实现跨体系预测。

---

##  核心创新：w ↔ T ↔ z 三向交替迭代闭环

Phase 1 内部的三向闭环（z 在 w 稳定后引入）：

```
┌─────────────────────────────────────────────────────────────────┐
│                Phase1: w ↔ T ↔ z 三向交替迭代                    │
│                                                                 │
│   ┌──────────┐   Adam更新w+z(中后期)  ┌──────────┐             │
│   │  打分公式  │ ─────────────────────→│  决策树   │             │
│   │ f(x;w,z) │                        │   T      │             │
│   └──────────┘                        └──────────┘             │
│        ↑                                   │                    │
│        │        叶内方差减少反馈             │                    │
│        └───────────────────────────────────┘                    │
│                                                                 │
│   打分公式 = Σ w_i*term_i + z_proj(z) 标量偏置                   │
│   term_i = 基础项 + 交互项 + 惩罚项(显式扣分)                     │
│   z 由 Adam 梯度下降优化，L2 正则防过拟合                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

Phase 3 的特征回注闭环（独立于 z 优化）：

┌─────────────────────────────────────────────────────────────────┐
│                Phase3: 特征迭代器发现新特征                       │
│                                                                 │
│   ┌──────────┐    条件分布分析    ┌──────────┐                  │
│   │  特征集   │ ────────────────→ │  新候选   │ ──→ 回注Phase1  │
│   └──────────┘                    └──────────┘                  │
│   FeatureIterator: 独热分箱 + 卡方/Fisher检验 + FDR校正          │
│   注意：这是发现新特征，和 z 的优化完全无关                        │
└─────────────────────────────────────────────────────────────────┘
```

**三个组件的实际优化方式**：

| 组件 | 作用 | 实际优化方式 |
|---|---|---|
| **w 打分公式** | 把多维特征压成一个标量分数（基础项+交互项+惩罚项） | Adam梯度下降 + 树反馈耦合，L1正则稀疏化 |
| **T 决策树** | 自动发现关键特征-阈值组合，叶内方差给w反馈 | 多树×10次重复×稳定性选择，Phase2加置换检验 |
| **z 隐变量** | 捕捉未显式指定的催化剂微观状态（全局标量偏置） | Phase1中后期(z_start_round后)引入，Adam梯度下降，L2正则 |

---

## 快速开始 | Quick Start

### 安装

```bash
git clone https://github.com/用户名/catalyst-framework.git
cd catalyst-framework
pip install -r requirements.txt
```

### 最小示例

```python
import numpy as np
import pandas as pd
from main import CatalystFramework

# 假设你有催化剂特征数据和目标性质（如FE）
feature_names = ['Fe_content', 'Ni_content', 'temperature', 'pH', 'electrolyte_conc']

# X: (n_samples, n_features) 特征矩阵
# y: (n_samples,) 目标性质（如法拉第效率）
# df: pandas DataFrame，包含所有特征列（用于特征迭代器分析）
X = np.array([...])  # 你的特征数据
y = np.array([...])  # 你的目标性质
df = pd.DataFrame(X, columns=feature_names)

# 初始化框架
framework = CatalystFramework(feature_names=feature_names)

# 运行完整的5+2阶段流程
# 注意：df, X, y 三个参数必传，df用于特征分析模块
results = framework.run(df, X, y, test_size=0.15)

# 返回值结构（results字典）：
print(f"最终状态: {results['final_status']}")        # GLOBAL_SUCCESS / LOCAL_SUCCESS / LOCAL_DESCRIPTION
print(f"结论: {results['final_message']}")
print(f"发现的新特征: {results['discovered_features']}")
print(f"Phase1-3结果: {results['phases']}")          # 包含p1/p2/p3/p4各阶段详细结果

# 稳定特征（Phase2稳定性选择的结果）
stable_feats = [feature_names[i] for i in results['phases']['p2']['stability']['stable_features']]
print(f"稳定特征: {stable_feats}")

# 神经网络预测性能（Phase4）
best_test_r2 = results['phases']['p4']['best_test_r2']
print(f"最佳测试 R²: {best_test_r2:.4f}")

# 保存结果到JSON
framework.save_results("output/results.json")
```

### 流程概览

框架运行 5 个主阶段 + 2 个中间辅助阶段：

| 阶段 | 说明 |
|---|---|
| Phase 1 | 打分公式 ↔ 决策树 ↔ 隐变量 三向交替迭代 |
| Phase 1.5 | 排除偏离样本（3×MAD 稳健异常检测） |
| Phase 2 | 稳定性选择 + 置换检验筛选关键规则 |
| Phase 3 | 条件分布分析 + 迭代找特征（含特征回注闭环） |
| Phase 4 | 神经网络拟合（ExactFit + RuleRegularized） |
| Phase 4.5 | 多元判别（二元→多元跃迁，需体系标签） |
| Phase 5 | 最终判定（GLOBAL_SUCCESS / LOCAL_SUCCESS / LOCAL_DESCRIPTION） |

---

##  项目结构 | Structure

```
catalyst_framework_standalone/
├── main.py                 # 主框架 CatalystFramework，5+2 Phase 完整流程
├── config.py               # 全局配置（打分/树/置换/NN/收敛参数）
├── scoring.py              # 打分公式 + ScoringOptimizer（w↔T耦合核心）
├── tree_fitting.py         # 决策树包装器（路径提取/叶内方差/分裂增益）
├── stability.py            # 稳定性选择（多树×多种子）
├── permutation_test.py     # 置换检验（叶内方差减少量零分布）
├── feature_analysis.py     # 独热编码+条件分布分析+FDR+特征迭代器
├── nn_models.py            # 神经网络（精确拟合/可微箱/规则正则/多元判别）
├── DESIGN.md               # 框架设计思路说明
├── README.md               # 本文件
├── requirements.txt        # 依赖列表
├── LICENSE                 # MIT License
└── .gitignore              # Git忽略规则
```

---

## 核心模块说明 | Core Modules

### 1. scoring.py — 打分公式 + 优化器

```python
class ScoringFormula:
    """打分公式：f(x;w,z) = Σ w_i * term_i(x) + (z启用时) z_proj(z) 标量偏置
       term_i = 基础项 + 交互项 + 惩罚项；z_proj是MLP(z_dim→16→1)"""

class ScoringOptimizer:
    """w↔T耦合优化器：Adam梯度下降更新w（z启用时一起更新），
       损失=MSE + L1正则(w) + L2正则(z)，同时接受树的叶内方差反馈"""
```

### 2. tree_fitting.py — 决策树包装器

```python
class DecisionTreeWrapper:
    """提取树路径、计算叶内方差、分裂增益"""

class TreeEnsembleFeedback:
    """多树集成反馈，把树的分裂信息回传给w优化器"""
```

### 3. feature_analysis.py — 特征分析 + 新特征发现（Phase3）

```python
class FeatureEncoder:
    """独热编码 + 数值特征离散化"""

class ConditionalDistributionAnalyzer:
    """条件分布分析，发现特征间关联（lift/卡方/Fisher检验），提出新特征候选"""

class FeatureIterator:
    """特征组合迭代器，自动搜索关键特征组合并回注Phase1"""
```

### 4. nn_models.py — 神经网络精确拟合

```python
class ExactFitNet:        # 精确拟合网络
class SoftBinningNet:     # 可微软分箱
class RuleRegularizedNet: # 规则正则化（注入树发现的规则）
class MultiClassNet:      # 多元判别网络
class NNTrainer:          # 训练器
```

---

##  技术亮点 | Highlights

1. **三向耦合迭代**：w、T、z 不是独立训练——w和z用Adam联合优化，T通过叶内方差反馈影响w，自动收敛；
2. **可解释性强**：决策树提取的特征-阈值组合 = 可直接读的催化剂设计规则；
3. **稳定性选择**：多树×10次重复×置换检验，避免单棵树的运气成分；
4. **隐变量建模**：z是可学习的全局标量偏置，由MLP投影，捕捉显式特征之外的催化剂微观状态；
5. **规则注入NN**：把树发现的规则作为正则项注入神经网络，兼顾精度与解释；
6. **跨体系预测**：同一框架处理不同催化剂-环境体系。

---

## 适用场景 | Use Cases

- 催化剂性能预测（产率、法拉第效率、选择性）
- 材料体系跨环境迁移学习
- 小规模样本下的局域规律发现
- 需要可解释 AI 的材料/化学研发场景

---

##  注意事项 | Notes

- 框架为**探索性研究代码**，核心算法逻辑完整，但需要根据实际数据调整配置；
- 默认配置针对中小规模数据集（< 5000 样本）设计；
- 隐变量 z 的建模依赖条件分布假设，需根据领域知识验证。

---

##  许可证 | License

MIT License — 详见 [LICENSE](LICENSE)

---

##  设计心路 | Design Notes

详细设计思路（独立推导过程）见 [DESIGN.md](DESIGN.md)。

##  其他问题 | Other Issues
详见Design.md

---

*本项目为材料领域学习实践中的思考性工作。核心架构独立完成。如有高度相似或完全撞车的早于本项目的工作，欢迎告知，将及时补充引用说明。*
