# -*- coding: utf-8 -*-
"""置换检验：打乱标签重算，判断好结果是不是靠运气蒙的。"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from tree_fitting import DecisionTreeWrapper
from config import PERMUTATION_CONFIG, TREE_CONFIG


class PermutationTest:
    """置换检验器：用叶内方差减少量做统计量，打乱y构造零分布算p值。"""

    def __init__(self, n_permutations: int = None, alpha: float = None,
                 tree_config: Optional[Dict] = None):
        self.n_permutations = n_permutations or PERMUTATION_CONFIG["n_permutations"]
        self.alpha = alpha or PERMUTATION_CONFIG["alpha"]

        # 故意用浅树，太深会把噪声也学进去
        self.tree_config = tree_config or dict(TREE_CONFIG)
        self.tree_config["max_depth"] = 4
        self.tree_config["min_samples_leaf"] = 10

    def _compute_statistic(self, X: np.ndarray, y: np.ndarray) -> float:
        # 统计量 = 1 - 叶内加权方差/总方差，越大说明分得越好
        tree = DecisionTreeWrapper(self.tree_config)
        tree.fit(X, y)

        total_var = np.var(y)
        leaf_variances = tree.get_leaf_variances()
        leaf_sample_counts = np.array([len(p.samples) for p in tree.paths])
        weighted_leaf_var = np.sum(leaf_variances * leaf_sample_counts) / np.sum(leaf_sample_counts)

        stat = 1.0 - weighted_leaf_var / (total_var + 1e-10)
        return stat

    def test(self, X: np.ndarray, y: np.ndarray,
             feature_subset: Optional[List[int]] = None) -> Dict:
        # 核心流程：真实数据算一次，打乱y算n次构成零分布，再算p值
        if feature_subset is not None:
            X_test = X[:, feature_subset]
        else:
            X_test = X

        observed_stat = self._compute_statistic(X_test, y)

        # 零分布：把y打乱，重复算很多次
        null_stats = []
        for i in range(self.n_permutations):
            y_permuted = np.random.permutation(y)
            perm_stat = self._compute_statistic(X_test, y_permuted)
            null_stats.append(perm_stat)

        null_stats = np.array(null_stats)

        # p值 = 零分布中 >= 观察值的比例，加1平滑防0
        p_value = (np.sum(null_stats >= observed_stat) + 1) / (self.n_permutations + 1)

        # 效应量：观察值比零分布均值高几个标准差
        null_mean = np.mean(null_stats)
        null_std = np.std(null_stats) + 1e-10
        effect_size = (observed_stat - null_mean) / null_std

        significant = p_value < self.alpha

        return {
            "observed_stat": observed_stat,
            "null_distribution": null_stats,
            "p_value": p_value,
            "significant": significant,
            "effect_size": effect_size,
            "null_mean": null_mean,
            "null_std": null_std,
            "n_permutations": self.n_permutations,
        }

    def test_feature_importance(self, X: np.ndarray, y: np.ndarray,
                                 n_top_features: int = 5) -> List[Dict]:
        # 先按重要性排序，再对前n个特征单独做置换检验
        tree = DecisionTreeWrapper(self.tree_config)
        tree.fit(X, y)
        importance_order = np.argsort(tree.feature_importances)[::-1]

        results = []
        for feat_idx in importance_order[:n_top_features]:
            result = self.test(X, y, feature_subset=[feat_idx])
            result["feature_idx"] = int(feat_idx)
            results.append(result)

        return results


if __name__ == "__main__":
    print("=== 置换检验模块自测 ===")

    np.random.seed(42)
    n_samples = 200
    n_features = 6
    X = np.random.randn(n_samples, n_features)

    # 目标值只依赖前2个特征，后面是噪声
    y_true = 2.0 * X[:, 0] + 1.0 * X[:, 1]
    y = y_true + 0.5 * np.random.randn(n_samples)

    tester = PermutationTest(n_permutations=100)

    print("\n检验全部特征...")
    result = tester.test(X, y)
    print(f"  观察统计量: {result['observed_stat']:.4f}")
    print(f"  零分布均值: {result['null_mean']:.4f}")
    print(f"  p值: {result['p_value']:.4f}")
    print(f"  效应量: {result['effect_size']:.4f}")
    print(f"  显著: {result['significant']}")

    print("\n单特征显著性检验:")
    feat_results = tester.test_feature_importance(X, y, n_top_features=4)
    for r in feat_results:
        sig = "★" if r["significant"] else " "
        print(f"  feat{r['feature_idx']}: p={r['p_value']:.4f} effect={r['effect_size']:.3f} {sig}")

    print("=== 自测完成 ===")
