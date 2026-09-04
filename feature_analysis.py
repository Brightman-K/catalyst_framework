# -*- coding: utf-8 -*-
"""条件分布分析模块：独热编码 + 关联检验 + FDR校正 + 特征迭代"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from itertools import combinations
from scipy.stats import fisher_exact, chi2_contingency
from sklearn.preprocessing import KBinsDiscretizer
from config import FEATURE_ITER_CONFIG


class FeatureEncoder:
    """把原始特征编码成统一的0/1独热矩阵"""
    
    def __init__(self, n_bins: int = None, bin_method: str = None):
        self.n_bins = n_bins or FEATURE_ITER_CONFIG["n_bins"]
        self.bin_method = bin_method or FEATURE_ITER_CONFIG["bin_method"]
        self.encoders: Dict[str, KBinsDiscretizer] = {}
        self.categories: Dict[str, List] = {}
        self.feature_names: List[str] = []
    
    def fit(self, df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str]):
        """拟合编码器，记录分箱边界和类别取值"""
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        
        # 数值特征分箱后独热
        for col in numeric_cols:
            vals = df[col].values.reshape(-1, 1)
            discretizer = KBinsDiscretizer(
                n_bins=self.n_bins,
                encode="ordinal",
                strategy=self.bin_method
            )
            discretizer.fit(vals)
            self.encoders[col] = discretizer
            
            bin_edges = discretizer.bin_edges_[0]
            for i in range(self.n_bins):
                name = f"{col}_bin{i}"
                self.feature_names.append(name)
        
        # 类别特征直接独热
        for col in categorical_cols:
            cats = sorted(df[col].unique().tolist())
            self.categories[col] = cats
            for cat in cats:
                name = f"{col}_{cat}"
                self.feature_names.append(name)
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """把数据转成独热矩阵"""
        n_samples = len(df)
        encoded_blocks = []
        
        for col in self.numeric_cols:
            vals = df[col].values.reshape(-1, 1)
            bin_idx = self.encoders[col].transform(vals).astype(int).flatten()
            one_hot_block = np.zeros((n_samples, self.n_bins))
            one_hot_block[np.arange(n_samples), bin_idx] = 1.0  # 对应箱的位置设1
            encoded_blocks.append(one_hot_block)
        
        for col in self.categorical_cols:
            cats = self.categories[col]
            one_hot_block = np.zeros((n_samples, len(cats)))
            for i, cat in enumerate(cats):
                one_hot_block[:, i] = (df[col].values == cat).astype(float)
            encoded_blocks.append(one_hot_block)
        
        return np.hstack(encoded_blocks) if encoded_blocks else np.zeros((n_samples, 0))
    
    def fit_transform(self, df: pd.DataFrame, numeric_cols: List[str], 
                      categorical_cols: List[str]) -> np.ndarray:
        self.fit(df, numeric_cols, categorical_cols)
        return self.transform(df)


class ConditionalDistributionAnalyzer:
    """分析特征间的条件分布关联：A出现时B会不会出现"""
    
    def __init__(self, min_support: int = None, fdr_alpha: float = None):
        self.min_support = min_support or FEATURE_ITER_CONFIG["min_support"]
        self.fdr_alpha = fdr_alpha or FEATURE_ITER_CONFIG["fdr_alpha"]
    
    def _fisher_test(self, a: int, b: int, c: int, d: int) -> Tuple[float, float]:
        """Fisher精确检验，小样本用"""
        table = np.array([[a, b], [c, d]])
        odds_ratio, p_value = fisher_exact(table, alternative="two-sided")
        return odds_ratio, p_value
    
    def _chi2_test(self, table: np.ndarray) -> Tuple[float, float]:
        """卡方检验，大样本用"""
        chi2, p_value, _, _ = chi2_contingency(table)
        return chi2, p_value
    
    def analyze_pair(self, one_hot: np.ndarray, feat_a: int, feat_b: int,
                     feature_names: List[str]) -> Dict:
        """算 P(B|A) vs P(B|¬A)，看A对B有没有影响"""
        n_samples = one_hot.shape[0]
        
        # 四格计数：A和B的四种组合各多少样本
        a_and_b = np.sum((one_hot[:, feat_a] == 1) & (one_hot[:, feat_b] == 1))
        a_and_not_b = np.sum((one_hot[:, feat_a] == 1) & (one_hot[:, feat_b] == 0))
        not_a_and_b = np.sum((one_hot[:, feat_a] == 0) & (one_hot[:, feat_b] == 1))
        not_a_and_not_b = np.sum((one_hot[:, feat_a] == 0) & (one_hot[:, feat_b] == 0))
        
        support_a = a_and_b + a_and_not_b
        if support_a < self.min_support:
            return {
                "valid": False,
                "reason": f"支持度不足: A出现 {support_a} < {self.min_support}"
            }
        
        p_b_given_a = a_and_b / support_a if support_a > 0 else 0  # P(B|A)
        p_b_given_not_a = not_a_and_b / (not_a_and_b + not_a_and_not_b) if (not_a_and_b + not_a_and_not_b) > 0 else 0  # P(B|¬A)
        
        total = a_and_b + a_and_not_b + not_a_and_b + not_a_and_not_b
        if total < 100:  # 小样本用Fisher，大样本用卡方
            odds_ratio, p_value = self._fisher_test(a_and_b, a_and_not_b, not_a_and_b, not_a_and_not_b)
        else:
            table = np.array([[a_and_b, a_and_not_b], [not_a_and_b, not_a_and_not_b]])
            _, p_value = self._chi2_test(table)
    
        # 提升度 lift = P(B|A) / P(B)，A让B的概率提高几倍
        p_b = (a_and_b + not_a_and_b) / total if total > 0 else 0
        lift = p_b_given_a / (p_b + 1e-10) if p_b > 0 else 0
        
        p_a = support_a / total if total > 0 else 0
        independence_expected_p_b_given_a = p_b  # 独立时 P(B|A)=P(B)
        observed_deviation = p_b_given_a - independence_expected_p_b_given_a
        
        return {
            "valid": True,
            "feat_a": feature_names[feat_a],
            "feat_b": feature_names[feat_b],
            "p_b_given_a": p_b_given_a,
            "p_b_given_not_a": p_b_given_not_a,
            "odds_ratio": odds_ratio if total < 100 else None,
            "lift": lift,
            "independence_expected_p_b": independence_expected_p_b_given_a,
            "observed_deviation": observed_deviation,
            "p_value": p_value,
            "support_a": support_a,
            "counts": {
                "a_and_b": int(a_and_b),
                "a_and_not_b": int(a_and_not_b),
                "not_a_and_b": int(not_a_and_b),
                "not_a_and_not_b": int(not_a_and_not_b),
            }
        }
    
    def analyze_triple(self, one_hot: np.ndarray, feat_a: int, feat_b: int, feat_c: int,
                       feature_names: List[str]) -> Dict:
        """看A和B的各种组合下C出现的概率变化"""
        n_samples = one_hot.shape[0]
        results = {}
        
        # 遍历4种 A/B 组合
        for a_val in [0, 1]:
            for b_val in [0, 1]:
                mask = (one_hot[:, feat_a] == a_val) & (one_hot[:, feat_b] == b_val)
                subset = one_hot[mask]
                n_subset = len(subset)
                
                if n_subset < self.min_support:
                    results[f"A={a_val},B={b_val}"] = {
                        "valid": False,
                        "n_samples": n_subset,
                        "reason": f"支持度不足: {n_subset} < {self.min_support}"
                    }
                    continue
                
                p_c = np.sum(subset[:, feat_c] == 1) / n_subset
                results[f"A={a_val},B={b_val}"] = {
                    "valid": True,
                    "p_c": p_c,
                    "n_samples": n_subset
                }
        
        all_valid = all(r.get("valid", False) for r in results.values())
        if all_valid:
            table = np.zeros((2, 2, 2), dtype=int)
            for a_val in [0, 1]:
                for b_val in [0, 1]:
                    mask = (one_hot[:, feat_a] == a_val) & (one_hot[:, feat_b] == b_val)
                    table[a_val, b_val, 1] = np.sum(mask & (one_hot[:, feat_c] == 1))
                    table[a_val, b_val, 0] = np.sum(mask & (one_hot[:, feat_c] == 0))
            _, p_cond_indep, _, _ = chi2_contingency(table.reshape(4, 2))
            results["conditional_independence_p"] = p_cond_indep
        
        return {
            "feat_a": feature_names[feat_a],
            "feat_b": feature_names[feat_b],
            "feat_c": feature_names[feat_c],
            "conditions": results
        }
    
    def bh_fdr_correction(self, p_values: List[float]) -> np.ndarray:
        """BH-FDR校正，多次检验控制假阳性"""
        p_values = np.array(p_values)
        n = len(p_values)
        sorted_idx = np.argsort(p_values)
        sorted_p = p_values[sorted_idx]
        
        # BH公式：p_corrected(k) = p(k) * n / k
        ranks = np.arange(1, n + 1)
        corrected_sorted = sorted_p * n / ranks
        corrected_sorted = np.minimum.accumulate(corrected_sorted[::-1])[::-1]  # 保证单调不减
        corrected_sorted = np.clip(corrected_sorted, 0, 1)
        
        corrected = np.zeros(n)
        corrected[sorted_idx] = corrected_sorted
        return corrected
    
    def find_associations(self, one_hot: np.ndarray, feature_names: List[str],
                          max_pairs: int = 100, top_k: int = 20) -> List[Dict]:
        """批量分析特征对，FDR校正后返回显著关联"""
        n_features = one_hot.shape[1]
        all_pairs = list(combinations(range(n_features), 2))
        np.random.shuffle(all_pairs)
        all_pairs = all_pairs[:max_pairs]
        
        results = []
        p_values = []
        
        for feat_a, feat_b in all_pairs:
            result = self.analyze_pair(one_hot, feat_a, feat_b, feature_names)
            if result.get("valid", False):
                results.append(result)
                p_values.append(result["p_value"])
        
        # FDR校正，压假阳性
        if p_values:
            corrected_p = self.bh_fdr_correction(p_values)
            for i, result in enumerate(results):
                result["p_value_corrected"] = corrected_p[i]
                result["significant"] = corrected_p[i] < self.fdr_alpha
        
        significant = [r for r in results if r.get("significant", False)]
        significant.sort(key=lambda x: x.get("lift", 0), reverse=True)
        
        return significant[:top_k]


class FeatureIterator:
    """不断发现新特征加入打分公式，迭代到收敛"""
    
    def __init__(self, encoder: FeatureEncoder, analyzer: ConditionalDistributionAnalyzer,
                 max_iterations: int = None):
        self.encoder = encoder
        self.analyzer = analyzer
        self.max_iterations = max_iterations or FEATURE_ITER_CONFIG["max_iterations"]
        self.discovered_features: List[str] = []
        self.iteration_history: List[Dict] = []
    
    def iterate(self, df: pd.DataFrame, scoring_features: List[str], 
                numeric_cols: List[str], categorical_cols: List[str],
                y: np.ndarray, current_feature_names: List[str]) -> Dict:
        """一轮特征发现：编码→找关联→挑新特征"""
        one_hot = self.encoder.fit_transform(df, numeric_cols, categorical_cols)
        encoded_names = self.encoder.feature_names
        
        associations = self.analyzer.find_associations(one_hot, encoded_names)
        
        # 从显著关联里挑还没用过的特征当新候选
        new_candidates = []
        for assoc in associations:
            feat_a = assoc["feat_a"]
            feat_b = assoc["feat_b"]
            if feat_a not in current_feature_names and feat_a not in self.discovered_features:
                new_candidates.append(feat_a)
            if feat_b not in current_feature_names and feat_b not in self.discovered_features:
                new_candidates.append(feat_b)
        
        new_candidates = list(set(new_candidates))
        new_candidates = new_candidates[:3]  # 每轮最多加3个，稳一点
        
        result = {
            "n_associations": len(associations),
            "top_associations": associations[:5],
            "new_candidates": new_candidates,
            "n_new": len(new_candidates),
        }
        self.iteration_history.append(result)
        
        self.discovered_features.extend(new_candidates)
        return result
    
    def should_stop(self) -> bool:
        """达到最大次数或连续两轮没新特征就停"""
        if len(self.iteration_history) >= self.max_iterations:
            return True
        if len(self.iteration_history) >= 2:
            last_two = self.iteration_history[-2:]
            if all(r["n_new"] == 0 for r in last_two):
                return True
        return False


if __name__ == "__main__":
    print("=== 特征分析模块自测 ===")
    
    np.random.seed(42)
    n_samples = 200
    df = pd.DataFrame({
        "potential": np.random.uniform(1.0, 3.0, n_samples),
        "FE": np.random.uniform(0.5, 1.0, n_samples),
        "yield": np.random.uniform(0.3, 1.0, n_samples),
        "catalyst_type": np.random.choice(["A", "B", "C"], n_samples),
        "electrolyte": np.random.choice(["KOH", "NaOH", "H2SO4"], n_samples),
    })
    
    encoder = FeatureEncoder(n_bins=3)
    one_hot = encoder.fit_transform(df, 
                                     numeric_cols=["potential", "FE", "yield"],
                                     categorical_cols=["catalyst_type", "electrolyte"])
    print(f"独热编码矩阵形状: {one_hot.shape}")
    print(f"编码特征名: {encoder.feature_names}")
    
    analyzer = ConditionalDistributionAnalyzer(min_support=10, fdr_alpha=0.05)
    
    feat_names = encoder.feature_names
    result = analyzer.analyze_pair(one_hot, 0, 5, feat_names)
    print(f"\n单对分析: {result}")
    
    print("\n批量关联分析（前100对）:")
    sig = analyzer.find_associations(one_hot, feat_names, max_pairs=100, top_k=5)
    for s in sig:
        print(f"  {s['feat_a']} → {s['feat_b']}: "
              f"P(B|A)={s['p_b_given_a']:.3f} vs P(B|¬A)={s['p_b_given_not_a']:.3f}, "
              f"lift={s['lift']:.2f}, p_corr={s['p_value_corrected']:.4f}")
    
    print("=== 自测完成 ===")
