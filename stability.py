# -*- coding: utf-8 -*-
"""稳定性选择：训练多棵树，反复出现的特征才算稳定可靠。"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.tree import DecisionTreeRegressor
from tree_fitting import DecisionTreeWrapper
from config import TREE_CONFIG


class StabilitySelector:
    """稳定性选择器：多棵树统计特征出现频率，筛出稳定特征。"""

    def __init__(self, n_trees: int = None, n_repeats: int = None,
                 config: Optional[Dict] = None):
        self.config = config or TREE_CONFIG
        self.n_trees = n_trees or self.config["n_trees"]
        self.n_repeats = n_repeats or self.config["n_repeats"]
        self.trees: List[DecisionTreeWrapper] = []
        self.feature_frequencies: np.ndarray = np.array([])
        self.stable_features: List[int] = []
        self.feature_paths: Dict[int, List[List[Tuple]]] = {}

    def fit(self, X: np.ndarray, y: np.ndarray,
            feature_names: List[str] = None,
            stability_threshold: float = 0.5):
        # 训练 n_trees × n_repeats 棵树，统计每个特征被用到的频率
        self.n_samples, self.n_features = X.shape
        self.feature_names = feature_names or [f"feat{i}" for i in range(self.n_features)]

        feature_counts = np.zeros(self.n_features)
        total_trees = self.n_trees * self.n_repeats

        print(f"训练 {self.n_trees} 棵树 × {self.n_repeats} 次重复 = {total_trees} 棵树...")

        for tree_idx in range(self.n_trees):
            for repeat_idx in range(self.n_repeats):
                # 每棵树用不同种子，保证有差异
                seed = self.config["random_state_seed"] + tree_idx * 1000 + repeat_idx
                cfg = dict(self.config)
                cfg["random_state_seed"] = seed

                tree = DecisionTreeWrapper(cfg)
                if cfg["bootstrap"]:
                    boot_idx = np.random.choice(self.n_samples, self.n_samples, replace=True)
                    X_boot = X[boot_idx]
                    y_boot = y[boot_idx]
                else:
                    X_boot, y_boot = X, y

                tree.fit(X_boot, y_boot, self.feature_names)
                self.trees.append(tree)

                # 统计这棵树用了哪些特征
                used = tree.feature_importances > 0
                feature_counts += used

                # 顺手记录每个特征的路径
                for path in tree.paths:
                    for cond in path.conditions:
                        feat_idx = cond[0]
                        if feat_idx not in self.feature_paths:
                            self.feature_paths[feat_idx] = []
                        self.feature_paths[feat_idx].append(path.conditions)

        # 频率 = 出现次数 / 总树数，超过阈值的就是稳定特征
        self.feature_frequencies = feature_counts / total_trees
        self.stable_features = np.where(
            self.feature_frequencies >= stability_threshold
        )[0].tolist()

        print(f"稳定特征数: {len(self.stable_features)} / {self.n_features}")

    def get_feature_stability(self) -> Dict[str, float]:
        return {
            self.feature_names[i]: float(self.feature_frequencies[i])
            for i in range(self.n_features)
        }

    def get_aggregated_importances(self) -> np.ndarray:
        all_imps = np.array([t.feature_importances for t in self.trees])
        return np.mean(all_imps, axis=0)

    def get_top_features(self, k: int = 5) -> List[Tuple[int, float]]:
        # 综合得分 = 稳定性 × 平均重要性
        mean_imp = self.get_aggregated_importances()
        combined = self.feature_frequencies * mean_imp
        ranking = sorted(
            enumerate(combined),
            key=lambda x: x[1],
            reverse=True
        )
        return ranking[:k]

    def get_representative_paths(self, top_k_paths: int = 5) -> List:
        all_paths = []
        for tree in self.trees:
            for path in tree.paths:
                all_paths.append(path)

        # 叶内方差越小，路径分得越干净
        all_paths.sort(key=lambda p: p.var_score)
        return all_paths[:top_k_paths]


if __name__ == "__main__":
    print("=== 稳定性选择模块自测 ===")

    np.random.seed(42)
    n_samples = 300
    n_features = 8
    X = np.random.randn(n_samples, n_features)
    # 目标值只依赖前3个特征，后面是噪声
    y = (2.0 * X[:, 0] + 1.5 * X[:, 1] - 1.0 * X[:, 2] +
         0.3 * np.random.randn(n_samples))
    feature_names = [f"feat{i}" for i in range(n_features)]

    selector = StabilitySelector(n_trees=20, n_repeats=3)
    selector.fit(X, y, feature_names, stability_threshold=0.5)

    print("\n特征稳定性:")
    for name, freq in selector.get_feature_stability().items():
        marker = " ★稳定" if freq >= 0.5 else ""
        print(f"  {name}: {freq:.3f}{marker}")

    print(f"\n稳定特征索引: {selector.stable_features}")
    print(f"Top-5特征: {selector.get_top_features(5)}")

    print("\n代表性路径（叶内方差最小）:")
    for i, path in enumerate(selector.get_representative_paths(3)):
        print(f"  路径{i+1}: {path}")

    print("=== 自测完成 ===")
