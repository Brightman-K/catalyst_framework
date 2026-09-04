# -*- coding: utf-8 -*-
"""用决策树分析打分分数的分布结构"""
import numpy as np
from sklearn.tree import DecisionTreeRegressor, _tree
from typing import Dict, List, Tuple, Optional

from config import TREE_CONFIG


class TreePath:
    """存一条从根到叶的路径"""
    
    def __init__(self, leaf_id: int, samples: np.ndarray, 
                 mean_score: float, var_score: float):
        """初始化路径信息"""
        self.leaf_id = leaf_id
        self.samples = samples
        self.mean_score = mean_score
        self.var_score = var_score      # 叶内方差，越小分得越好
        self.conditions = []            # [(特征下标, 运算符, 阈值), ...]
    
    def add_condition(self, feature_idx: int, operator: str, threshold: float):
        self.conditions.append((feature_idx, operator, threshold))
    
    def __repr__(self):
        cond_str = " AND ".join(
            [f"feat{c[0]} {c[1]} {c[2]:.4f}" for c in self.conditions]
        )
        return f"Leaf{self.leaf_id}: [{cond_str}] var={self.var_score:.4f}"


class DecisionTreeWrapper:
    """包装 sklearn 决策树，提取叶内方差、路径和分裂增益"""
    
    def __init__(self, config: Optional[Dict] = None, feature_subset: Optional[List[int]] = None):
        """初始化树模型"""
        self.config = config or TREE_CONFIG
        self.feature_subset = feature_subset  # 固定特征子集，跨轮锁定树种类
        
        self.tree = DecisionTreeRegressor(
            criterion="squared_error",        # 用方差做分裂准则
            max_depth=self.config["max_depth"],
            min_samples_leaf=self.config["min_samples_leaf"],
            max_features=self.config["max_features"],
            random_state=self.config["random_state_seed"],
        )
        self.paths: List[TreePath] = []
        self.feature_importances: np.ndarray = np.array([])
        self.split_gains: Dict[int, float] = {}
        self.actual_feature_subset: List[int] = []
    
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: List[str] = None):
        """拟合树，提取路径和统计量"""
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]
        self.feature_names = feature_names or [f"feat{i}" for i in range(self.n_features)]
        
        # 固定特征子集，跨轮强制树用同样的特征
        if self.feature_subset is not None and len(self.feature_subset) > 0:
            X_subset = X[:, self.feature_subset]
            self._subset_to_global = {i: global_i for i, global_i in enumerate(self.feature_subset)}
        else:
            X_subset = X
            self._subset_to_global = {i: i for i in range(self.n_features)}
            
        self.tree.fit(X_subset, y)
        self._X_subset = X_subset
        
        # 记录树实际用了哪些特征
        tree_obj = self.tree.tree_
        used_local = set()
        for node_id in range(tree_obj.node_count):
            if tree_obj.feature[node_id] != _tree.TREE_UNDEFINED:
                used_local.add(tree_obj.feature[node_id])
        self.actual_feature_subset = [self._subset_to_global[l] for l in sorted(used_local)]
        
        self.paths = self._extract_paths(X, y)
        self.feature_importances = np.zeros(self.n_features)
        for local_i, global_i in self._subset_to_global.items():
            if local_i < len(self.tree.feature_importances_):
                self.feature_importances[global_i] = self.tree.feature_importances_[local_i]
        self.split_gains = self._compute_split_gains(X, y)
    
    def _extract_paths(self, X: np.ndarray, y: np.ndarray) -> List[TreePath]:
        """从树里提取所有根到叶的路径"""
        tree = self.tree.tree_
        paths = []
        
        def recurse(node_id: int, conditions: List[Tuple]):
            if tree.feature[node_id] == _tree.TREE_UNDEFINED:
                # 碰到叶子，算叶内方差存起来
                leaf_samples = np.where(self.tree.apply(self._X_subset) == node_id)[0]
                if len(leaf_samples) > 0:
                    leaf_y = y[leaf_samples]
                    path = TreePath(
                        leaf_id=node_id,
                        samples=leaf_samples,
                        mean_score=np.mean(leaf_y),
                        var_score=np.var(leaf_y)     # 叶内方差，核心指标
                    )
                    path.conditions = list(conditions)
                    paths.append(path)
                return
            
            feat_idx_local = tree.feature[node_id]
            feat_idx = self._subset_to_global[feat_idx_local]
            threshold = tree.threshold[node_id]
            
            recurse(tree.children_left[node_id], conditions + [(feat_idx, "<=", threshold)])
            recurse(tree.children_right[node_id], conditions + [(feat_idx, ">", threshold)])
        
        recurse(0, [])
        return paths
    
    def _compute_split_gains(self, X: np.ndarray, y: np.ndarray) -> Dict[int, float]:
        """算每个特征的分裂增益，后面反馈给权重 w"""
        tree = self.tree.tree_
        gains = {i: 0.0 for i in range(self.n_features)}
        
        def recurse(node_id: int, parent_samples: np.ndarray):
            if tree.feature[node_id] == _tree.TREE_UNDEFINED:
                return
            
            feat_idx_local = tree.feature[node_id]
            feat_idx = self._subset_to_global[feat_idx_local]
            threshold = tree.threshold[node_id]
            
            left_mask = self._X_subset[parent_samples, feat_idx_local] <= threshold
            left_samples = parent_samples[left_mask]
            right_samples = parent_samples[~left_mask]
            
            # 分裂增益 = 父方差 - 加权子方差，越小说明分叉越有用
            parent_var = np.var(y[parent_samples])
            n_parent = len(parent_samples)
            
            n_left = len(left_samples)
            n_right = len(right_samples)
            weighted_child_var = (
                (n_left / n_parent) * np.var(y[left_samples]) + 
                (n_right / n_parent) * np.var(y[right_samples])
            )
            
            gain = parent_var - weighted_child_var
            gains[feat_idx] += gain
            
            recurse(tree.children_left[node_id], left_samples)
            recurse(tree.children_right[node_id], right_samples)
        
        all_samples = np.arange(self.n_samples)
        recurse(0, all_samples)
        return gains
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.tree.predict(X)
    
    def get_leaf_variances(self) -> np.ndarray:
        return np.array([p.var_score for p in self.paths])
    
    def get_mean_leaf_variance(self) -> float:
        return np.mean(self.get_leaf_variances())  # 平均叶内方差，越小越好
    
    def get_weight_feedback(self) -> np.ndarray:
        """把分裂增益归一化成反馈向量，告诉 w 哪些特征重要"""
        total_gain = sum(self.split_gains.values()) + 1e-10
        feedback = np.array([
            self.split_gains[i] / total_gain if i in self.split_gains else 0.0 
            for i in range(self.n_features)
        ])
        return feedback


class TreeEnsembleFeedback:
    """多棵树的反馈聚合，降低单棵树的随机性"""
    
    def __init__(self, n_features: int, feature_names: List[str] = None):
        self.n_features = n_features
        self.feature_names = feature_names or [f"feat{i}" for i in range(n_features)]
        self.feedback_history = []
    
    def aggregate_feedback(self, trees: List[DecisionTreeWrapper]) -> np.ndarray:
        """把多棵树的反馈取平均"""
        all_feedbacks = []
        for tree in trees:
            fb = tree.get_weight_feedback()
            all_feedbacks.append(fb)
        
        aggregated = np.mean(all_feedbacks, axis=0)
        self.feedback_history.append(aggregated)
        return aggregated
    
    def get_stable_features(self, trees: List[DecisionTreeWrapper], 
                            threshold: float = 0.5) -> List[int]:
        """找在多棵树里都出现的稳定特征"""
        feature_counts = np.zeros(self.n_features)
        for tree in trees:
            used = tree.feature_importances > 0
            feature_counts += used
        
        frequencies = feature_counts / len(trees)
        stable_indices = np.where(frequencies >= threshold)[0].tolist()
        return stable_indices
    
    def get_feature_ranking(self, trees: List[DecisionTreeWrapper]) -> List[Tuple[int, float]]:
        """按平均重要性给特征排名"""
        all_importances = np.array([t.feature_importances for t in trees])
        mean_importance = np.mean(all_importances, axis=0)
        ranking = sorted(
            enumerate(mean_importance),
            key=lambda x: x[1],
            reverse=True
        )
        return ranking


if __name__ == "__main__":
    print("=== 决策树模块自测 ===")
    
    np.random.seed(0)
    n_samples = 200
    X = np.random.randn(n_samples, 5)
    y = 3.0 * X[:, 0] + 2.0 * X[:, 1] + 0.5 * np.random.randn(n_samples)
    feature_names = ["feat0", "feat1", "feat2", "feat3", "feat4"]
    
    tree = DecisionTreeWrapper()
    tree.fit(X, y, feature_names)
    print(f"平均叶内方差: {tree.get_mean_leaf_variance():.4f}")
    print(f"特征重要性: {tree.feature_importances}")
    print(f"分裂增益: {tree.split_gains}")
    
    for i, path in enumerate(sorted(tree.paths, key=lambda p: p.var_score)[:3]):
        print(f"路径{i+1}: {path}")
    
    trees = []
    for seed in range(5):
        cfg = dict(TREE_CONFIG)
        cfg["random_state_seed"] = seed
        t = DecisionTreeWrapper(cfg)
        t.fit(X, y, feature_names)
        trees.append(t)
    
    feedback = TreeEnsembleFeedback(5, feature_names)
    agg = feedback.aggregate_feedback(trees)
    print(f"聚合反馈: {agg}")
    stable = feedback.get_stable_features(trees, threshold=0.6)
    print(f"稳定特征索引: {stable}")
    ranking = feedback.get_feature_ranking(trees)
    print(f"特征排名: {ranking}")
    print("=== 自测完成 ===")
