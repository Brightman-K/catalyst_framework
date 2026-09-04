# -*- coding: utf-8 -*-
"""框架主入口，w↔T↔z三向交替迭代闭环，跑完整五阶段流程。"""
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional
from config import (
    SCORING_CONFIG, TREE_CONFIG, PERMUTATION_CONFIG,
    FEATURE_ITER_CONFIG, NN_CONFIG, DATA_SPLIT_CONFIG, CONVERGENCE_CONFIG
)
from scoring import ScoringFormula, ScoringOptimizer, dict_to_tensor
from tree_fitting import DecisionTreeWrapper, TreeEnsembleFeedback
from stability import StabilitySelector
from permutation_test import PermutationTest
from feature_analysis import (
    FeatureEncoder, ConditionalDistributionAnalyzer, FeatureIterator
)
from nn_models import ExactFitNet, SoftBinningNet, RuleRegularizedNet, NNTrainer, build_rule_features, MultiClassNet, MultiClassTrainer


class CatalystFramework:
    """框架主类，封装w↔T↔z三向迭代闭环的所有步骤。"""

    def __init__(self, feature_names: List[str],
                 numeric_cols: List[str] = None,
                 categorical_cols: List[str] = None,
                 ablation: Dict = None):
        """初始化框架，把所有子模块都创建出来。"""
        self.feature_names = feature_names
        self.numeric_cols = numeric_cols or feature_names
        self.categorical_cols = categorical_cols or []
        self.ablation = ablation or {}

        self.scoring_model = ScoringFormula(feature_names)
        self.scoring_optimizer = ScoringOptimizer(self.scoring_model)

        self.tree_feedback = TreeEnsembleFeedback(len(feature_names), feature_names)
        self.stability_selector = StabilitySelector()
        self.permutation_tester = PermutationTest()

        self.encoder = FeatureEncoder()
        self.analyzer = ConditionalDistributionAnalyzer()
        self.feature_iterator = FeatureIterator(self.encoder, self.analyzer)

        self.results = {
            "phase1_scoring_tree": [],
            "phase2_stability": None,
            "phase2_permutation": None,
            "phase3_feature_iter": [],
            "phase4_nn": {},
            "phase5_validation": None,
            "final_status": None,
        }
    
    def _prepare_input_dict(self, X: np.ndarray) -> Dict[str, torch.Tensor]:
        """把NumPy特征矩阵转成特征名字典。"""
        return {
            name: torch.tensor(X[:, i], dtype=torch.float32)
            for i, name in enumerate(self.feature_names)
        }

    def _map_feedback_to_terms(self, feedback_feat: np.ndarray) -> np.ndarray:
        """把特征级反馈映射成打分公式的term级反馈（交互项取两特征平均）。"""
        interaction_pairs = self.scoring_model.interaction_pairs
        penalty_features = self.scoring_model.penalty_features
        n_terms = self.scoring_model.n_terms

        feedback_terms = np.zeros(n_terms)
        term_idx = 0

        # 基础项直接搬
        for i, name in enumerate(self.feature_names):
            feedback_terms[term_idx] = feedback_feat[i] if i < len(feedback_feat) else 0.0
            term_idx += 1

        # 交互项取两特征反馈平均
        for pair in interaction_pairs:
            feat_a, feat_b = pair
            idx_a = self.feature_names.index(feat_a) if feat_a in self.feature_names else -1
            idx_b = self.feature_names.index(feat_b) if feat_b in self.feature_names else -1
            fa = feedback_feat[idx_a] if idx_a >= 0 else 0.0
            fb = feedback_feat[idx_b] if idx_b >= 0 else 0.0
            feedback_terms[term_idx] = (fa + fb) / 2.0
            term_idx += 1

        # 惩罚项直接映射
        for feat in penalty_features:
            idx = self.feature_names.index(feat) if feat in self.feature_names else -1
            feedback_terms[term_idx] = feedback_feat[idx] if idx >= 0 else 0.0
            term_idx += 1

        return feedback_terms
    
    def build_feature_matrix(self, df: pd.DataFrame, indices: np.ndarray = None) -> np.ndarray:
        """根据当前feature_names从DataFrame构建特征矩阵，支持原始列和衍生列。"""
        cols = []
        one_hot_matrix = None
        encoder_names = self.encoder.feature_names if hasattr(self, 'encoder') else []

        for name in self.feature_names:
            if name in df.columns:
                col = df[name].values
                if indices is not None:
                    col = col[indices]
                cols.append(col.reshape(-1, 1).astype(np.float32))
            elif name in encoder_names:
                if one_hot_matrix is None:
                    full_one_hot = self.encoder.transform(df)
                    if indices is not None:
                        one_hot_matrix = full_one_hot[indices]
                    else:
                        one_hot_matrix = full_one_hot
                col_idx = encoder_names.index(name)
                cols.append(one_hot_matrix[:, col_idx].reshape(-1, 1).astype(np.float32))
            else:
                n = len(indices) if indices is not None else len(df)
                cols.append(np.zeros((n, 1), dtype=np.float32))
        return np.hstack(cols) if cols else np.zeros((0, 0))

    def rebuild_model_with_new_features(self, new_feature_names: List[str],
                                         X: np.ndarray, df: pd.DataFrame = None,
                                         train_indices: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """Phase3闭环核心：把新特征加进来，重建打分模型和树反馈器。"""
        if not new_feature_names:
            return X, []

        truly_new = [f for f in new_feature_names if f not in self.feature_names]
        if not truly_new:
            return X, []

        if df is not None:
            new_cols = []
            added = []
            encoder_names = self.encoder.feature_names if hasattr(self, 'encoder') else []
            one_hot_matrix = None
            need_one_hot = any(name in encoder_names for name in truly_new)
            if need_one_hot:
                full_one_hot = self.encoder.transform(df)
                if train_indices is not None:
                    one_hot_matrix = full_one_hot[train_indices]
                else:
                    one_hot_matrix = full_one_hot

            for name in truly_new:
                if name in df.columns:
                    col = df[name].values
                    if train_indices is not None:
                        col = col[train_indices]
                    new_cols.append(col.reshape(-1, 1).astype(np.float32))
                    added.append(name)
                elif name in encoder_names and one_hot_matrix is not None:
                    col_idx = encoder_names.index(name)
                    new_cols.append(one_hot_matrix[:, col_idx].reshape(-1, 1).astype(np.float32))
                    added.append(name)
            if new_cols:
                new_block = np.hstack(new_cols)
                X_new = np.hstack([X, new_block])
                self.feature_names = self.feature_names + added
                self.scoring_model = ScoringFormula(self.feature_names)
                self.scoring_optimizer = ScoringOptimizer(self.scoring_model)
                self.tree_feedback = TreeEnsembleFeedback(len(self.feature_names), self.feature_names)
                print(f"  → 已加入新特征 {added}，模型已重建，回到 Phase1 重新迭代")
                return X_new, added

        print(f"  → 候选特征 {truly_new} 在数据中未找到对应列，跳过")
        return X, []
    
    def phase1_scoring_tree_loop(self, X: np.ndarray, y: np.ndarray,
                                  max_outer_iterations: int = None) -> Dict:
        """Phase1核心：w↔T交替迭代，树反馈调w，稳定后引入z，直到收敛。"""
        max_outer = max_outer_iterations or CONVERGENCE_CONFIG["max_outer_iterations"]
        x_dict = self._prepare_input_dict(X)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        # 树种类锁定：第一轮探索出特征子集后锁死，后续轮次控制变量公平比较
        locked_tree_subsets = None

        print(f"\n{'='*60}")
        print("Phase 1: 打分公式 ↔ 决策树 ↔ 隐变量 三向交替迭代")
        print(f"{'='*60}")

        prev_weights = self.scoring_model.get_weights().copy()
        prev_z = np.zeros(SCORING_CONFIG["z_dim"])
        prev_leaf_var = None
        outer_history = []

        for round_idx in range(max_outer):
            print(f"\n  --- 外层迭代 {round_idx + 1}/{max_outer} ---")

            # 步骤1：梯度下降更新w
            losses = self.scoring_optimizer.update(x_dict, y_tensor)
            current_weights = self.scoring_model.get_weights()
            w_change = np.linalg.norm(current_weights - prev_weights)
            print(f"  [w更新] 损失={losses[-1]:.6f}, ||Δw||={w_change:.6f}")

            # 步骤2：算打分公式的分数给树拟合，树分析分数分布结构不是预测
            scores = self.scoring_optimizer.predict(x_dict)

            # 步骤3：每棵树跑10次不同种子取平均，削减运气成分
            trees = []
            tree_repeat_vars = []
            new_locked_subsets = []

            for tree_idx in range(min(TREE_CONFIG["n_trees"], 20)):
                # 第一轮自由探索，后续轮次用锁定的特征子集
                if locked_tree_subsets is not None and tree_idx < len(locked_tree_subsets):
                    feat_subset = locked_tree_subsets[tree_idx]
                else:
                    feat_subset = None

                # 每棵树跑10次不同种子
                repeat_vars = []
                best_tree = None
                best_var = float('inf')
                for repeat_idx in range(10):
                    cfg = dict(TREE_CONFIG)
                    cfg["random_state_seed"] = TREE_CONFIG["random_state_seed"] + tree_idx * 100 + repeat_idx
                    tree = DecisionTreeWrapper(cfg, feature_subset=feat_subset)
                    tree.fit(X, scores, self.feature_names)
                    leaf_var = tree.get_mean_leaf_variance()
                    repeat_vars.append(leaf_var)
                    if leaf_var < best_var:
                        best_var = leaf_var
                        best_tree = tree
                avg_var = np.mean(repeat_vars)
                tree_repeat_vars.append(repeat_vars)
                trees.append(best_tree)
                # 第一轮记录实际用的特征子集，后面锁定
                if locked_tree_subsets is None:
                    new_locked_subsets.append(best_tree.actual_feature_subset)
                subset_info = f" 特征子集={[self.feature_names[i] for i in feat_subset]}" if feat_subset else " 自由探索"
                print(f"  [树#{tree_idx+1}] 10次平均叶内方差={avg_var:.6f}{subset_info}")

            # 树种类锁定！后续轮次不再探索新的特征子集
            if locked_tree_subsets is None:
                locked_tree_subsets = new_locked_subsets
                print(f"  [树种类锁定] 第一轮探索出{len(locked_tree_subsets)}种树种类，后续轮次不再变动")

            mean_leaf_var = np.mean([t.get_mean_leaf_variance() for t in trees])
            print(f"  [树拟合] 全部{len(trees)}棵树平均叶内方差={mean_leaf_var:.6f}")

            # 步骤4：树门控——≥5棵树10次平均都提升才保留本轮打分公式
            tree_accepted = True
            if prev_leaf_var is not None:
                worse_count = 0
                for tree_idx, repeat_vars in enumerate(tree_repeat_vars):
                    avg_var = np.mean(repeat_vars)
                    improvement = (prev_leaf_var - avg_var) / (prev_leaf_var + 1e-10)
                    if improvement < -0.1:
                        worse_count += 1

                worse_ratio = worse_count / len(tree_repeat_vars) if tree_repeat_vars else 0
                if worse_ratio > 0.5:
                    tree_accepted = False
                    print(f"  [树门控] 拒绝：{worse_count}/{len(tree_repeat_vars)}棵树10次平均都恶化>10% (比例={worse_ratio:.0%})")
                else:
                    accepted_count = len(tree_repeat_vars) - worse_count
                    print(f"  [树门控] 接受：{accepted_count}/{len(tree_repeat_vars)}棵树10次平均都更好")

            # 步骤5：树反馈更新w——w↔T双向耦合的核心，树告诉w哪些特征重要
            if tree_accepted:
                if not self.ablation.get("no_tree_feedback", False):
                    feedback_feat = self.tree_feedback.aggregate_feedback(trees)
                    feedback_terms = self._map_feedback_to_terms(feedback_feat)
                    self.scoring_optimizer.apply_tree_feedback(feedback_terms, feedback_lr=0.15)
                    current_weights = self.scoring_model.get_weights()
                    top_features = self.tree_feedback.get_feature_ranking(trees)
                    print(f"  [树→w反馈] 已更新w, Top特征: {[(self.feature_names[i], f'{v:.4f}') for i, v in top_features[:3]]}")
                else:
                    top_features = self.tree_feedback.get_feature_ranking(trees)
                    print(f"  [消融模式] 跳过w↔T耦合（w只梯度下降）")
            else:
                top_features = self.tree_feedback.get_feature_ranking(trees)
                print(f"  [树→w反馈] 跳过（树未被接受）")

            # 步骤6：记录本轮结果
            z_current = self.scoring_optimizer.get_z_value()
            z_change = np.linalg.norm(z_current - prev_z) if self.scoring_model.z_active else 0.0

            round_result = {
                "round": round_idx,
                "w_change": float(w_change),
                "z_change": float(z_change),
                "final_loss": float(losses[-1]),
                "mean_leaf_var": float(mean_leaf_var),
                "tree_accepted": tree_accepted,
                "top_features": top_features,
                "weights": current_weights.tolist(),
                "z_active": self.scoring_model.z_active,
                "z_value": z_current.tolist() if self.scoring_model.z_active else None,
            }
            outer_history.append(round_result)

            # 步骤7：z引入——前几轮先让w稳定，到指定轮数再引入z
            if (not self.scoring_model.z_active and
                not self.ablation.get("no_z", False) and
                round_idx + 1 >= SCORING_CONFIG["z_start_round"]):
                print(f"  → 引入隐变量 z (dim={SCORING_CONFIG['z_dim']})")
                self.scoring_model.activate_z()
                self.scoring_optimizer = ScoringOptimizer(self.scoring_model)
                prev_z = np.zeros(SCORING_CONFIG["z_dim"])
                z_current = self.scoring_optimizer.get_z_value()
                z_change = 0.0

            # 步骤8：收敛判断——w和z变化都很小就停
            converged_w = w_change < CONVERGENCE_CONFIG["w_tol"]
            converged_z = True
            if self.scoring_model.z_active:
                converged_z = z_change < CONVERGENCE_CONFIG["z_tol"]

            if converged_w and converged_z and round_idx > 0:
                print(f"  → 收敛：||Δw||={w_change:.6f}, ||Δz||={z_change:.6f}")
                break

            prev_weights = current_weights.copy()
            prev_z = z_current.copy() if self.scoring_model.z_active else prev_z
            prev_leaf_var = mean_leaf_var

        phase_result = {
            "history": outer_history,
            "final_weights": self.scoring_model.get_weights().tolist(),
            "final_z": self.scoring_model.z.detach().numpy().tolist() if self.scoring_model.z_active else None,
            "z_active": self.scoring_model.z_active,
            "converged": w_change < CONVERGENCE_CONFIG["w_tol"],
        }
        self.results["phase1_scoring_tree"] = outer_history

        return phase_result
    
    def phase2_stability_permutation(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Phase2：稳定性选择找反复出现的特征 + 置换检验验证显著性。"""
        print(f"\n{'='*60}")
        print("Phase 2: 稳定性选择 + 置换检验")
        print(f"{'='*60}")

        print("\n[1/2] 稳定性选择...")
        self.stability_selector = StabilitySelector(
            n_trees=min(TREE_CONFIG["n_trees"], 30),
            n_repeats=min(TREE_CONFIG["n_repeats"], 5)
        )
        self.stability_selector.fit(X, y, self.feature_names, stability_threshold=0.5)

        stability_result = {
            "frequencies": self.stability_selector.get_feature_stability(),
            "stable_features": self.stability_selector.stable_features,
            "top_features": self.stability_selector.get_top_features(10),
        }
        print(f"  稳定特征: {[self.feature_names[i] for i in self.stability_selector.stable_features]}")

        print("\n[2/2] 置换检验...")
        self.permutation_tester = PermutationTest(n_permutations=min(PERMUTATION_CONFIG["n_permutations"], 200))
        perm_result = self.permutation_tester.test(X, y)
        print(f"  观察统计量: {perm_result['observed_stat']:.4f}")
        print(f"  p值: {perm_result['p_value']:.4f}")
        print(f"  效应量: {perm_result['effect_size']:.4f}")
        print(f"  显著: {perm_result['significant']}")

        print("\n  单特征显著性:")
        feat_results = self.permutation_tester.test_feature_importance(X, y, n_top_features=5)
        for r in feat_results:
            sig = "★" if r["significant"] else " "
            print(f"    {self.feature_names[r['feature_idx']]}: p={r['p_value']:.4f} effect={r['effect_size']:.3f} {sig}")

        phase_result = {
            "stability": stability_result,
            "permutation_full": perm_result,
            "permutation_single": feat_results,
        }
        self.results["phase2_stability"] = stability_result
        self.results["phase2_permutation"] = phase_result

        return phase_result

    def phase3_feature_iteration(self, df: pd.DataFrame, X: np.ndarray, y: np.ndarray) -> Dict:
        """Phase3：条件分布分析找特征间关联，迭代发现新候选特征。"""
        print(f"\n{'='*60}")
        print("Phase 3: 条件分布分析 + 迭代找特征")
        print(f"{'='*60}")

        iteration_results = []

        for iter_idx in range(min(FEATURE_ITER_CONFIG["max_iterations"], 3)):
            print(f"\n  --- 特征迭代 {iter_idx + 1} ---")

            result = self.feature_iterator.iterate(
                df, self.feature_names, self.numeric_cols, self.categorical_cols,
                y, self.feature_names
            )

            print(f"  显著关联数: {result['n_associations']}")
            if result["top_associations"]:
                for assoc in result["top_associations"][:3]:
                    print(f"    {assoc['feat_a']} → {assoc['feat_b']}: "
                          f"lift={assoc.get('lift', 0):.2f}, p_corr={assoc.get('p_value_corrected', 1):.4f}")
            print(f"  新候选特征: {result['new_candidates']}")

            iteration_results.append(result)

            if self.feature_iterator.should_stop():
                print("  → 迭代停止（无新特征或达到最大次数）")
                break

        phase_result = {
            "iterations": iteration_results,
            "discovered_features": self.feature_iterator.discovered_features,
        }
        self.results["phase3_feature_iter"] = iteration_results

        return phase_result
    
    def phase4_nn_fitting(self, X_train: np.ndarray, y_train: np.ndarray,
                          X_val: np.ndarray, y_val: np.ndarray,
                          X_test: np.ndarray, y_test: np.ndarray,
                          tree_paths: list = None) -> Dict:
        """Phase4：训练三种神经网络拟合，取最佳测试R²判断是否成功。"""
        print(f"\n{'='*60}")
        print("Phase 4: 神经网络拟合")
        print(f"{'='*60}")

        X_tr = torch.tensor(X_train, dtype=torch.float32)
        y_tr = torch.tensor(y_train, dtype=torch.float32)
        X_va = torch.tensor(X_val, dtype=torch.float32)
        y_va = torch.tensor(y_val, dtype=torch.float32)
        X_te = torch.tensor(X_test, dtype=torch.float32)
        y_te = torch.tensor(y_test, dtype=torch.float32)

        input_dim = X_train.shape[1]
        nn_results = {}

        print("\n[1/3] 精确拟合网络 (ExactFitNet)...")
        exact_net = ExactFitNet(input_dim, hidden_dims=NN_CONFIG["hidden_dims"])
        trainer1 = NNTrainer(exact_net)
        hist1 = trainer1.train(X_tr, y_tr, X_va, y_va, max_epochs=NN_CONFIG["max_epochs"])
        pred1 = trainer1.predict(X_te)
        test_r2_1 = 1 - np.sum((y_test - pred1) ** 2) / (np.sum((y_test - np.mean(y_test)) ** 2) + 1e-10)
        print(f"  验证R²: {hist1['val_r2'][-1]:.4f}, 测试R²: {test_r2_1:.4f}")
        nn_results["exact_fit"] = {
            "val_r2": hist1["val_r2"][-1],
            "test_r2": float(test_r2_1),
            "history": hist1,
        }

        print("\n[2/3] 可微箱范围网络 (SoftBinningNet)...")
        binning_net = SoftBinningNet(input_dim, n_bins=NN_CONFIG["n_bins"],
                                      hidden_dims=NN_CONFIG["hidden_dims"])
        trainer2 = NNTrainer(binning_net)
        hist2 = trainer2.train(X_tr, y_tr, X_va, y_va, max_epochs=NN_CONFIG["max_epochs"])
        pred2 = trainer2.predict(X_te)
        test_r2_2 = 1 - np.sum((y_test - pred2) ** 2) / (np.sum((y_test - np.mean(y_test)) ** 2) + 1e-10)
        print(f"  验证R²: {hist2['val_r2'][-1]:.4f}, 测试R²: {test_r2_2:.4f}")
        edges = binning_net.soft_binning.bin_edges[0, :4].detach().numpy()
        print(f"  学习到的箱边界（特征0前4个）: {edges}")
        nn_results["soft_binning"] = {
            "val_r2": hist2["val_r2"][-1],
            "test_r2": float(test_r2_2),
            "learned_edges": edges.tolist(),
            "history": hist2,
        }

        if tree_paths and len(tree_paths) > 0:
            print("\n[3/3] 规则正则化网络 (RuleRegularizedNet)...")
            rule_train = build_rule_features(tree_paths[:20], X_train, input_dim)
            rule_val = build_rule_features(tree_paths[:20], X_val, input_dim)
            rule_test = build_rule_features(tree_paths[:20], X_test, input_dim)

            rule_tr = torch.tensor(rule_train, dtype=torch.float32)
            rule_va = torch.tensor(rule_val, dtype=torch.float32)
            rule_te = torch.tensor(rule_test, dtype=torch.float32)

            rule_net = RuleRegularizedNet(input_dim, len(tree_paths[:20]),
                                           hidden_dims=NN_CONFIG["hidden_dims"])
            trainer3 = NNTrainer(rule_net)
            hist3 = trainer3.train(X_tr, y_tr, X_va, y_va,
                                    rule_features_train=rule_tr,
                                    rule_features_val=rule_va,
                                    max_epochs=NN_CONFIG["max_epochs"])
            pred3 = trainer3.predict(X_te, rule_features=rule_te)
            test_r2_3 = 1 - np.sum((y_test - pred3) ** 2) / (np.sum((y_test - np.mean(y_test)) ** 2) + 1e-10)
            print(f"  验证R²: {hist3['val_r2'][-1]:.4f}, 测试R²: {test_r2_3:.4f}")
            nn_results["rule_regularized"] = {
                "val_r2": hist3["val_r2"][-1],
                "test_r2": float(test_r2_3),
                "history": hist3,
            }

        best_r2 = max(r.get("test_r2", 0) for r in nn_results.values())
        success = best_r2 >= CONVERGENCE_CONFIG["success_r2"]
        print(f"\n  最佳测试R²: {best_r2:.4f} (成功判据: ≥ {CONVERGENCE_CONFIG['success_r2']})")
        print(f"  NN拟合{'成功' if success else '未达成功判据（局域成立）'}")

        phase_result = {
            "models": nn_results,
            "best_test_r2": float(best_r2),
            "success": success,
        }
        self.results["phase4_nn"] = phase_result

        return phase_result
    
    def run(self, df: pd.DataFrame, X: np.ndarray, y: np.ndarray,
            test_size: float = 0.15, system_labels: np.ndarray = None) -> Dict:
        """运行完整分析流程：数据划分→三向迭代→特征回注闭环→门控→NN→最终判定。"""
        print(f"\n{'#'*60}")
        print("# 催化剂-环境体系分析框架：w ↔ T ↔ z 三向迭代闭环")
        print(f"{'#'*60}")

        # 数据划分：训练/验证/测试三份
        n = len(y)
        n_test = int(n * test_size)
        n_val = int(n * 0.15)
        indices = np.random.permutation(n)
        test_idx = indices[:n_test]
        val_idx = indices[n_test:n_test + n_val]
        train_idx = indices[n_test + n_val:]

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        print(f"\n数据划分: 训练{len(X_train)}, 验证{len(X_val)}, 测试{len(X_test)}")

        # Phase1-3闭环：发现新特征就加回去重跑，最多3次防死循环
        X_current = X_train
        max_feedback_loops = 3
        all_new_features = []

        for feedback_loop in range(max_feedback_loops):
            print(f"\n{'#'*60}")
            print(f"特征回注闭环 {feedback_loop + 1}/{max_feedback_loops}")
            print(f"{'#'*60}")

            p1 = self.phase1_scoring_tree_loop(X_current, y_train)

            # Phase1.5：3×MAD剔离群点，MAD比均值抗干扰
            print(f"\n{'='*60}")
            print("Phase 1.5: 排除偏离样本（用3×MAD稳健异常检测）")
            print(f"{'='*60}")
            scores_full = self.scoring_optimizer.predict(self._prepare_input_dict(X_current))
            median_score = np.median(scores_full)
            mad = np.median(np.abs(scores_full - median_score))
            outlier_mask = np.abs(scores_full - median_score) > 3 * (mad + 1e-10)
            n_outliers = np.sum(outlier_mask)
            print(f"  打分数: {len(scores_full)}, 中位数={median_score:.4f}, MAD={mad:.4f}")
            print(f"  偏离样本数: {n_outliers} ({n_outliers/len(scores_full)*100:.1f}%)")

            keep_mask = ~outlier_mask
            if n_outliers > 0 and n_outliers < len(y_train) * 0.5:
                X_train_clean = X_current[keep_mask]
                y_train_clean = y_train[keep_mask]
                print(f"  剔除后训练集: {len(X_train_clean)} 样本")
            else:
                X_train_clean = X_current
                y_train_clean = y_train
                if n_outliers >= len(y_train) * 0.5:
                    print(f"  异常样本超过50%，不剔除（保持数据完整性）")

            p2 = self.phase2_stability_permutation(X_train_clean, y_train_clean)
            p3 = self.phase3_feature_iteration(df, X_train_clean, y_train_clean)

            # Phase3特征回注闭环：发现新特征就重建模型回Phase1
            discovered = self.feature_iterator.discovered_features
            if self.ablation.get("no_phase3_loop", False):
                print(f"\n  [消融模式] 跳过Phase3特征回注闭环")
                break
            elif discovered and feedback_loop < max_feedback_loops - 1:
                print(f"\n  [Phase3闭环] 发现新特征候选: {discovered}")
                X_new, added = self.rebuild_model_with_new_features(discovered, X_current, df, train_idx)
                if added:
                    all_new_features.extend(added)
                    X_current = X_new
                    print(f"  [Phase3闭环] 回到 Phase1 重新迭代...")
                    continue
                else:
                    print(f"  [Phase3闭环] 候选特征无法加入，结束闭环")
                    break
            else:
                print(f"\n  [Phase3闭环] 无新特征发现，结束闭环")
                break

        # NN门控判据：置换显著(p<0.05)且稳定特征≥3个才进NN，不然直接判局域
        perm_significant = p2.get("permutation_full", {}).get("significant", False)
        n_stable = len(p2.get("stability", {}).get("stable_features", []))
        gate_pass = perm_significant and n_stable >= 3
        print(f"\n[门控判据] 置换显著={perm_significant}, 稳定特征数={n_stable}, 通过={gate_pass}")

        if not gate_pass:
            print(f"  [门控未通过] 特征区分力不足，跳过NN精确拟合，直接判局域成立")
            p4 = {
                "success": False,
                "best_test_r2": 0.0,
                "reason": f"门控未通过: 置换显著={perm_significant}, 稳定特征数={n_stable}"
            }
            tree_paths = []
        else:
            tree_paths = self.stability_selector.get_representative_paths(top_k_paths=20)
            # 加了新特征要重建验证/测试集特征矩阵
            X_val_current = self.build_feature_matrix(df, val_idx)
            X_test_current = self.build_feature_matrix(df, test_idx)
            X_nn_train = X_current if 'X_train_clean' not in dir() else X_train_clean
            y_nn_train = y_train if 'y_train_clean' not in dir() else y_train_clean
            p4 = self.phase4_nn_fitting(X_nn_train, y_nn_train, X_val_current, y_val, X_test_current, y_test, tree_paths)

        # 存树路径组合
        saved_paths = []
        for p in tree_paths:
            cond_list = []
            for cond in p.conditions:
                feat_idx, operator, threshold = cond
                feat_name = self.feature_names[feat_idx] if feat_idx < len(self.feature_names) else f"feat{feat_idx}"
                cond_list.append({
                    "feature": feat_name,
                    "operator": operator,
                    "threshold": float(threshold)
                })
            saved_paths.append({
                "conditions": cond_list,
                "leaf_variance": float(p.var_score),
            })
        self.results["top_paths"] = saved_paths

        # Phase4.5：有体系标签就跑多元判别
        p4_5 = None
        if system_labels is not None and gate_pass:
            print(f"\n{'='*60}")
            print("Phase 4.5: 多元判别（二元→多元跃迁）")
            print(f"{'='*60}")

            y_sys_train = system_labels[train_idx]
            y_sys_val = system_labels[val_idx]
            y_sys_test = system_labels[test_idx]

            # 对齐清洗后的训练集
            if 'keep_mask' in dir() and len(y_sys_train) != X_nn_train.shape[0]:
                y_sys_train = y_sys_train[keep_mask]

            # 体系标签编码成数字索引
            unique_systems = sorted(set(y_sys_train))
            sys_to_idx = {s: i for i, s in enumerate(unique_systems)}
            n_classes = len(unique_systems)
            print(f"  体系类别数: {n_classes}, 体系: {unique_systems}")

            y_tr_cls = torch.tensor([sys_to_idx[s] for s in y_sys_train], dtype=torch.long)
            y_va_cls = torch.tensor([sys_to_idx[s] for s in y_sys_val], dtype=torch.long)
            y_te_cls = torch.tensor([sys_to_idx[s] for s in y_sys_test], dtype=torch.long)

            multi_net = MultiClassNet(X_nn_train.shape[1], n_classes,
                                       hidden_dims=NN_CONFIG["hidden_dims"])
            multi_trainer = MultiClassTrainer(multi_net)
            multi_hist = multi_trainer.train(
                torch.tensor(X_nn_train, dtype=torch.float32), y_tr_cls,
                torch.tensor(X_val_current, dtype=torch.float32), y_va_cls,
                max_epochs=NN_CONFIG["max_epochs"]
            )

            y_te_pred = multi_trainer.predict(torch.tensor(X_test_current, dtype=torch.float32))
            test_acc = np.mean(y_te_pred == y_te_cls.numpy())
            print(f"  验证准确率: {multi_hist['val_acc'][-1]:.4f}, 测试准确率: {test_acc:.4f}")

            p4_5 = {
                "n_classes": n_classes,
                "systems": unique_systems,
                "val_acc": multi_hist["val_acc"][-1],
                "test_acc": float(test_acc),
                "history": multi_hist,
            }
        else:
            print(f"\n[Phase 4.5] 未提供体系类别标签，跳过多元判别")

        # Phase5：最终判定
        print(f"\n{'#'*60}")
        print("Phase 5: 最终判定")
        print(f"{'#'*60}")

        if p4["success"] and p2["permutation_full"]["significant"]:
            final_status = "GLOBAL_SUCCESS"
            status_msg = "✓ 找到了隐变量分类方法，全局成立"
        elif p4["success"]:
            final_status = "LOCAL_SUCCESS"
            status_msg = "○ NN拟合成功，但置换检验未达显著，局域成立"
        else:
            final_status = "LOCAL_DESCRIPTION"
            status_msg = "○ NN拟合未达成功判据，作为局域精描述/快速筛选工具"

        print(f"\n最终状态: {final_status}")
        print(f"结论: {status_msg}")
        print(f"最佳测试R²: {p4['best_test_r2']:.4f}")
        print(f"置换检验p值: {p2['permutation_full']['p_value']:.4f}")
        print(f"稳定特征: {[self.feature_names[i] for i in p2['stability']['stable_features']]}")
        print(f"回注新特征: {all_new_features}")

        self.results["final_status"] = final_status
        self.results["final_message"] = status_msg
        self.results["discovered_features"] = all_new_features
        self.results["multiclass"] = p4_5
        self.results["phases"] = {"p1": p1, "p2": p2, "p3": p3, "p4": p4}

        return self.results
    
    def save_results(self, filepath: str):
        """把结果保存成JSON文件。"""
        import json

        def convert(obj):
            if isinstance(obj, np.ndarray):
                return [convert(x) for x in obj.tolist()]
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.int32, np.int64, np.int8, np.int16, np.uint8, np.uint16, np.uint32, np.uint64)):
                return int(obj)
            if isinstance(obj, np.bool_):
                return bool(obj)
            if isinstance(obj, np.str_):
                return str(obj)
            if isinstance(obj, dict):
                return {kk: convert(vv) for kk, vv in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [convert(item) for item in obj]
            return obj

        results_copy = convert(self.results)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results_copy, f, indent=2, ensure_ascii=False)

        print(f"\n结果已保存到: {filepath}")


if __name__ == "__main__":
    # 主流程自测
    print("=== 催化剂框架主流程自测 ===")

    np.random.seed(42)
    torch.manual_seed(42)

    n_samples = 300
    feature_names = ["potential", "FE", "yield", "concentration"]
    numeric_cols = feature_names
    categorical_cols = ["catalyst_type"]

    df = pd.DataFrame({
        "potential": np.random.uniform(1.0, 3.0, n_samples),
        "FE": np.random.uniform(0.5, 1.0, n_samples),
        "yield": np.random.uniform(0.3, 1.0, n_samples),
        "concentration": np.random.uniform(0.1, 1.0, n_samples),
        "catalyst_type": np.random.choice(["A", "B", "C"], n_samples),
    })

    X = df[numeric_cols].values.astype(np.float32)

    y = (2.0 * X[:, 0] + 1.5 * X[:, 1] - 1.0 * X[:, 2] +
         0.3 * np.random.randn(n_samples)).astype(np.float32)

    system_labels = np.array([
        "A" if p < 1.7 else ("B" if p < 2.4 else "C")
        for p in X[:, 0]
    ])

    framework = CatalystFramework(feature_names, numeric_cols, categorical_cols)

    results = framework.run(df, X, y, test_size=0.15, system_labels=system_labels)

    framework.save_results("output/test_results.json")

    try:
        from visualization import generate_all_visualizations
        generate_all_visualizations(results, save_dir="output/figures")
    except Exception as e:
        print(f"[可视化] 跳过（{e}）")

    print("\n=== 自测完成 ===")
