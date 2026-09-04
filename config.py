# -*- coding: utf-8 -*-
"""配置文件：统一管理所有参数和路径"""
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))  # 项目根目录
DATA_DIR = os.path.join(PROJECT_ROOT, "data")               # 数据目录
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")           # 输出目录
os.makedirs(DATA_DIR, exist_ok=True)                        # 创建数据目录
os.makedirs(OUTPUT_DIR, exist_ok=True)                      # 创建输出目录

SCORING_CONFIG = {
    "base_features": ["potential", "FE", "yield", "concentration"],  # 基础特征
    "penalty_features": ["potential", "concentration"],              # 惩罚特征
    "penalty_coeffs": [0.5, 0.3],                                    # 惩罚系数
    "interaction_pairs": [                                            # 交互项
        ("FE", "yield")
    ],
    "lr_w": 0.01,            # w学习率
    "lr_z": 0.005,           # z学习率
    "max_epochs": 200,       # 最大训练轮数
    "early_stop_patience": 20,  # 早停耐心值
    "z_dim": 6,              # z维度
    "z_init": "zero",        # z初始化方式
    "z_start_round": 3,      # z启用轮次
}

TREE_CONFIG = {
    "n_trees": 100,              # 平行树数量
    "max_depth": 6,              # 最大树深
    "min_samples_leaf": 5,       # 叶节点最小样本数
    "criterion": "variance",     # 分裂标准
    "n_repeats": 10,             # 重复次数
    "bootstrap": True,           # 是否有放回抽样
    "max_features": 0.7,         # 特征采样比例
    "random_state_seed": 42,     # 随机种子
}

PERMUTATION_CONFIG = {
    "n_permutations": 1000,      # 置换次数
    "alpha": 0.01,               # 显著性水平
    "scoring_metric": "leaf_variance_reduction",  # 检验统计量
}

FEATURE_ITER_CONFIG = {
    "max_iterations": 5,         # 最大迭代轮数
    "min_support": 10,           # 最小支持度
    "fdr_alpha": 0.05,           # FDR显著性水平
    "bin_method": "quantile",    # 分箱方法
    "n_bins": 5,                 # 分箱数
    "interaction_order": 3,      # 交互分析阶数
}

NN_CONFIG = {
    "hidden_dims": [64, 32],     # 隐藏层维度
    "lr": 0.001,                 # 学习率
    "batch_size": 32,            # 批量大小
    "max_epochs": 300,           # 最大训练轮数
    "patience": 30,              # 早停耐心值
    "z_dim": 6,                  # z维度
    "use_soft_binning": True,    # 是否可微软分箱
    "n_bins": 10,                # 软分箱数
    "bin_temperature": 0.5,      # 软分箱温度
    "lambda_reg": 1e-4,          # L2正则系数
    "rule_reg_weight": 0.1,      # 规则正则权重
}

DATA_SPLIT_CONFIG = {
    "train_ratio": 0.7,          # 训练集比例
    "val_ratio": 0.15,           # 验证集比例
    "test_ratio": 0.15,          # 测试集比例
    "stratify": True,            # 是否分层抽样
}

CONVERGENCE_CONFIG = {
    "w_tol": 1e-4,               # w收敛阈值
    "z_tol": 1e-4,               # z收敛阈值
    "tree_change_tol": 0.05,     # 树变化率阈值
    "max_outer_iterations": 10,  # 最大外层迭代
    "success_r2": 0.85,          # R²成功阈值
    "success_mae": 0.05,         # MAE成功阈值
}
