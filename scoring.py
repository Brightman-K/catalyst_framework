# -*- coding: utf-8 -*-
"""打分公式 f(x;w,z) 和它的优化器：基础项 + 交互项 + 惩罚项 + z投影偏置。"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from config import SCORING_CONFIG


class ScoringFormula(nn.Module):
    """打分公式模型 f(x;w,z)：基础项 + 交互项 + 惩罚项 + z投影偏置，w和z都可学习。"""

    def __init__(self, feature_names: List[str], config: Optional[Dict] = None):
        """初始化打分公式的各个零件。"""
        super().__init__()
        self.config = config or SCORING_CONFIG
        self.feature_names = feature_names

        # 惩罚项（拉齐项）：显式减号 + 可学习力度
        # 高电解液浓度天然产率高，所以要扣分，让不同浓度之间可比
        # penalty_coeff_raw 是原始参数，经过 softplus 保证正数=惩罚力度
        # 最终效果：scores -= penalty_coeff * feat（显式扣分，力度可学习）
        self.penalty_features = self.config["penalty_features"]
        self.penalty_coeff_raw = nn.Parameter(
            torch.tensor(self.config["penalty_coeffs"], dtype=torch.float32)
        )

        # 交互项：哪些特征两两相乘
        self.interaction_pairs = self.config["interaction_pairs"]

        # 总项数 = 基础项 + 交互项 + 惩罚项
        n_base = len(feature_names)
        n_interaction = len(self.interaction_pairs)
        n_penalty = len(self.penalty_features)
        self.n_terms = n_base + n_interaction + n_penalty

        # 权重 w，初始全1
        self.weights = nn.Parameter(torch.ones(self.n_terms, dtype=torch.float32))

        # 隐变量 z，先建好不启用，等w稳了再上
        self.z_dim = self.config["z_dim"]
        self.z_active = False
        if self.config["z_init"] == "zero":
            self.z = nn.Parameter(torch.zeros(self.z_dim, dtype=torch.float32))
        else:
            self.z = nn.Parameter(torch.randn(self.z_dim, dtype=torch.float32) * 0.01)

        # z投影网络：z_dim -> 16 -> 1，输出标量偏置
        self.z_proj = nn.Sequential(
            nn.Linear(self.z_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

        self.current_round = 0
    
    def build_terms(self, x_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """把所有打分项拼出来，返回 [batch, n_terms] 矩阵。"""
        batch_size = x_dict[list(x_dict.keys())[0]].shape[0]
        terms = []

        # 基础项
        for name in self.feature_names:
            if name in x_dict:
                terms.append(x_dict[name])
            else:
                terms.append(torch.zeros(batch_size, dtype=torch.float32))

        # 交互项
        for pair in self.interaction_pairs:
            feat_a, feat_b = pair
            if feat_a in x_dict and feat_b in x_dict:
                interaction = x_dict[feat_a] * x_dict[feat_b]
                terms.append(interaction)
            else:
                terms.append(torch.zeros(batch_size, dtype=torch.float32))

        # 惩罚项（拉齐项）：显式扣分，力度=softplus(原始参数)保证正数
        for i, feat in enumerate(self.penalty_features):
            if feat in x_dict:
                penalty_coeff = torch.nn.functional.softplus(self.penalty_coeff_raw[i])  # 保证正数=惩罚力度
                penalty = -x_dict[feat] * penalty_coeff  # 显式减号，高浓度扣分
                terms.append(penalty)
            else:
                terms.append(torch.zeros(batch_size, dtype=torch.float32))

        return torch.stack(terms, dim=1)

    def forward(self, x_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """前向算分数：terms乘w求和，z启用时再加z投影偏置。"""
        terms = self.build_terms(x_dict)
        scores = torch.matmul(terms, self.weights)  # f = Σ w_i * term_i

        # z启用时加全局偏置
        if self.z_active:
            z_proj = self.z_proj(self.z)
            scores = scores + z_proj.squeeze()

        return scores
    
    def activate_z(self):
        """启用隐变量z。"""
        self.z_active = True

    def deactivate_z(self):
        """禁用隐变量z。"""
        self.z_active = False

    def get_weights(self) -> np.ndarray:
        """返回权重w的NumPy数组。"""
        return self.weights.detach().numpy()

    def get_penalty_coeffs(self) -> np.ndarray:
        """返回实际惩罚力度（softplus后的值，保证正数）。"""
        return torch.nn.functional.softplus(self.penalty_coeff_raw).detach().numpy()

    def get_weight_names(self) -> List[str]:
        """返回每个权重对应的项名。"""
        names = []
        for name in self.feature_names:
            names.append(f"base_{name}")
        for pair in self.interaction_pairs:
            names.append(f"inter_{pair[0]}_{pair[1]}")
        for feat in self.penalty_features:
            names.append(f"penalty_{feat}")
        return names


class ScoringOptimizer:
    """打分公式优化器，用梯度下降调w和z，损失含MSE+L1(w)+L2(z)正则。"""

    def __init__(self, model: ScoringFormula, lr_w: float = None, lr_z: float = None):
        """初始化优化器，给w和z配不同学习率。"""
        self.model = model
        lr_w = lr_w or SCORING_CONFIG["lr_w"]
        lr_z = lr_z or SCORING_CONFIG["lr_z"]

        # w和惩罚系数一组，惩罚系数学得慢一点
        param_groups = [
            {"params": [model.weights], "lr": lr_w},
            {"params": [model.penalty_coeff_raw], "lr": lr_w * 0.5},  # 惩罚力度学得慢
        ]
        # z启用时把z和z_proj也加进来
        self.includes_z = False
        if model.z_active:
            param_groups.append({"params": [model.z], "lr": lr_z})
            param_groups.append({"params": model.z_proj.parameters(), "lr": lr_z})
            self.includes_z = True

        self.optimizer = torch.optim.Adam(param_groups)
        self.criterion = nn.MSELoss()
        self.lambda_w = 1e-4  # L1正则，让w稀疏
        self.lambda_z = 1e-4  # L2正则，防z过大

    def rebuild_for_z(self):
        """z状态变了就重建优化器，保证参数组对得上。"""
        if self.model.z_active and not self.includes_z:
            self.__init__(self.model)
        elif not self.model.z_active and self.includes_z:
            self.__init__(self.model)
    
    def update(self, x_dict: Dict[str, torch.Tensor], y: torch.Tensor,
               epochs: int = None, patience: int = None) -> List[float]:
        """梯度下降更新w（z启用时一起更新），带早停，返回每步loss。"""
        epochs = epochs or SCORING_CONFIG["max_epochs"]
        patience = patience or SCORING_CONFIG["early_stop_patience"]

        self.rebuild_for_z()

        losses = []
        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            self.model.train()
            self.optimizer.zero_grad()

            scores = self.model(x_dict)  # 前向算分数

            loss = self.criterion(scores, y)  # MSE损失

            # L1正则，鼓励w稀疏
            l1_reg = self.lambda_w * torch.norm(self.model.weights, p=1)
            loss = loss + l1_reg

            # z启用时加L2正则
            if self.model.z_active:
                l2_reg = self.lambda_z * torch.norm(self.model.z, p=2)
                loss = loss + l2_reg

            loss.backward()  # 反向传播
            self.optimizer.step()  # Adam更新参数

            losses.append(loss.item())

            # 早停：损失没改善就计数，到patience就停
            if loss.item() < best_loss - 1e-6:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        return losses

    def predict(self, x_dict: Dict[str, torch.Tensor]) -> np.ndarray:
        """预测打分，只算不训。"""
        self.model.eval()
        with torch.no_grad():
            scores = self.model(x_dict)
        return scores.numpy()
    
    def apply_tree_feedback(self, feedback: np.ndarray,
                            feedback_lr: float = 0.1,
                            momentum: float = 0.9):
        """w↔T耦合核心：树分裂增益归一化后按比例调w，高增益增权低增益衰减，带动量平滑。"""
        with torch.no_grad():
            # 归一化反馈到[-1,1]
            fb_norm = feedback / (np.max(np.abs(feedback)) + 1e-10)
            # 缩放因子：1 + lr*fb_norm，clip防崩
            scale = 1.0 + feedback_lr * fb_norm
            scale = np.clip(scale, 0.5, 2.0)

            # 动量平滑，避免振荡
            if not hasattr(self, "_feedback_velocity"):
                self._feedback_velocity = np.zeros_like(feedback)
            self._feedback_velocity = momentum * self._feedback_velocity + (1 - momentum) * scale
            effective_scale = self._feedback_velocity

            # 乘w，再L2归一化保持尺度
            w_tensor = self.model.weights.data
            w_np = w_tensor.numpy()
            w_np = w_np * effective_scale
            w_norm = np.linalg.norm(w_np) + 1e-10
            original_norm = np.linalg.norm(self.model.weights.data.numpy()) + 1e-10
            w_np = w_np * (original_norm / w_norm)
            self.model.weights.data.copy_(torch.tensor(w_np, dtype=torch.float32))

    def get_z_value(self) -> np.ndarray:
        """返回当前z值，没启用返回空数组。"""
        return self.model.z.detach().numpy() if self.model.z_active else np.array([])


def dict_to_tensor(x_dict: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
    """NumPy特征字典转PyTorch张量字典。"""
    return {k: torch.tensor(v, dtype=torch.float32) for k, v in x_dict.items()}


if __name__ == "__main__":
    # 自测
    print("=== 打分公式模块自测 ===")

    np.random.seed(0)
    batch_size = 50
    x_dict_np = {
        "potential": np.random.uniform(1.0, 3.0, batch_size),
        "FE": np.random.uniform(0.5, 1.0, batch_size),
        "yield": np.random.uniform(0.3, 1.0, batch_size),
        "concentration": np.random.uniform(0.1, 1.0, batch_size),
    }
    y_np = np.random.uniform(0.5, 2.0, batch_size)

    x_dict = dict_to_tensor(x_dict_np)
    y = torch.tensor(y_np, dtype=torch.float32)

    feature_names = ["potential", "FE", "yield", "concentration"]
    model = ScoringFormula(feature_names)
    print(f"打分项数量: {model.n_terms}")
    print(f"权重名称: {model.get_weight_names()}")

    scores_before = model(x_dict).detach().numpy()
    print(f"训练前打分范围: [{scores_before.min():.3f}, {scores_before.max():.3f}]")

    optimizer = ScoringOptimizer(model)
    losses = optimizer.update(x_dict, y, epochs=50, patience=10)
    print(f"最终损失: {losses[-1]:.6f}")
    print(f"权重 w: {model.get_weights()}")

    # 启用z再训一轮
    model.activate_z()
    optimizer = ScoringOptimizer(model)
    losses_z = optimizer.update(x_dict, y, epochs=50, patience=10)
    print(f"启用z后最终损失: {losses_z[-1]:.6f}")
    print("=== 自测完成 ===")
