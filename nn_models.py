# -*- coding: utf-8 -*-
"""神经网络模块：包含精确拟合、可微箱、规则正则化和多元判别四种网络。"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from config import NN_CONFIG


class SoftBinningLayer(nn.Module):
    """可微箱分层：用可学习边界+softmax软概率，把连续值映射成箱代表值。"""

    def __init__(self, n_bins: int = None, n_features: int = None,
                 temperature: float = None):
        super().__init__()
        self.n_bins = n_bins or NN_CONFIG["n_bins"]
        self.n_features = n_features
        self.temperature = temperature or NN_CONFIG["bin_temperature"]

        # 箱边界和代表值都可学习，初始化成均匀分布
        if n_features is not None:
            edges = torch.linspace(0, 1, self.n_bins + 1).unsqueeze(0).repeat(n_features, 1)
            self.bin_edges = nn.Parameter(edges)
            representatives = torch.linspace(0, 1, self.n_bins).unsqueeze(0).repeat(n_features, 1)
            self.representatives = nn.Parameter(representatives)
        else:
            edges = torch.linspace(0, 1, self.n_bins + 1)
            self.bin_edges = nn.Parameter(edges)
            representatives = torch.linspace(0, 1, self.n_bins)
            self.representatives = nn.Parameter(representatives)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 算每个样本到各箱中心的软概率，再加权求和得到分箱值
        if x.dim() == 1:
            x = x.unsqueeze(1)

        batch_size, n_feat = x.shape
        bin_centers = (self.bin_edges[..., :-1] + self.bin_edges[..., 1:]) / 2
        x_expanded = x.unsqueeze(-1)
        centers_expanded = bin_centers.unsqueeze(0)
        distances = -((x_expanded - centers_expanded) ** 2) / (2 * self.temperature ** 2)
        soft_probs = F.softmax(distances, dim=-1)
        reps_expanded = self.representatives.unsqueeze(0)
        binned = torch.sum(soft_probs * reps_expanded, dim=-1)
        return binned


class FreeEmbedding(nn.Module):
    """全局可学习的自由隐向量z，扩展到整个batch。"""

    def __init__(self, dim: int = None):
        super().__init__()
        self.dim = dim or NN_CONFIG["z_dim"]
        self.embedding = nn.Parameter(torch.randn(self.dim) * 0.01)

    def forward(self, batch_size: int) -> torch.Tensor:
        return self.embedding.unsqueeze(0).expand(batch_size, -1)


class ExactFitNet(nn.Module):
    """精确拟合网络：MLP回归，可拼接自由隐向量z。"""

    def __init__(self, input_dim: int, hidden_dims: List[int] = None,
                 z_dim: int = None, use_free_z: bool = True):
        super().__init__()
        hidden_dims = hidden_dims or NN_CONFIG["hidden_dims"]
        z_dim = z_dim or NN_CONFIG["z_dim"]
        self.use_free_z = use_free_z

        # 拼上自由z，增加模型表达力
        if use_free_z:
            self.free_z = FreeEmbedding(z_dim)
            input_dim = input_dim + z_dim

        # 搭MLP：线性+ReLU+Dropout堆起来
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        if self.use_free_z:
            z = self.free_z(batch_size)
            x = torch.cat([x, z], dim=1)
        return self.mlp(x).squeeze(-1)


class SoftBinningNet(nn.Module):
    """可微箱网络：先软分箱，再拼z过MLP做回归。"""

    def __init__(self, input_dim: int, n_bins: int = None,
                 hidden_dims: List[int] = None, z_dim: int = None):
        super().__init__()
        n_bins = n_bins or NN_CONFIG["n_bins"]
        hidden_dims = hidden_dims or NN_CONFIG["hidden_dims"]
        z_dim = z_dim or NN_CONFIG["z_dim"]

        # 先过软分箱层，每个特征独立分箱
        self.soft_binning = SoftBinningLayer(n_bins=n_bins, n_features=input_dim)
        self.free_z = FreeEmbedding(z_dim)
        mlp_input_dim = input_dim + z_dim

        layers = []
        prev_dim = mlp_input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x_binned = self.soft_binning(x)
        z = self.free_z(batch_size)
        x_aug = torch.cat([x_binned, z], dim=1)
        return self.mlp(x_aug).squeeze(-1)


class RuleRegularizedNet(nn.Module):
    """规则正则化网络：把树路径规则当软先验注入网络，让网络别乱学。"""

    def __init__(self, input_dim: int, n_rules: int,
                 hidden_dims: List[int] = None, z_dim: int = None):
        super().__init__()
        hidden_dims = hidden_dims or NN_CONFIG["hidden_dims"]
        z_dim = z_dim or NN_CONFIG["z_dim"]

        self.free_z = FreeEmbedding(z_dim)
        total_input = input_dim + n_rules + z_dim

        layers = []
        prev_dim = total_input
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.mlp = nn.Sequential(*layers)

        # 每条规则一个可学习权重，网络自己学规则多重要
        self.n_rules = n_rules
        self.rule_weights = nn.Parameter(torch.ones(n_rules))

    def forward(self, x: torch.Tensor, rule_features: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        weighted_rules = rule_features * self.rule_weights.unsqueeze(0)
        z = self.free_z(batch_size)
        x_aug = torch.cat([x, weighted_rules, z], dim=1)
        return self.mlp(x_aug).squeeze(-1)


class NNTrainer:
    """神经网络训练器，支持早停和L2正则。"""

    def __init__(self, model: nn.Module, lr: float = None,
                 patience: int = None, lambda_reg: float = None):
        self.model = model
        self.lr = lr or NN_CONFIG["lr"]
        self.patience = patience or NN_CONFIG["patience"]
        self.lambda_reg = lambda_reg or NN_CONFIG["lambda_reg"]
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.lambda_reg)
        self.criterion = nn.MSELoss()

    def train(self, X_train: torch.Tensor, y_train: torch.Tensor,
              X_val: torch.Tensor, y_val: torch.Tensor,
              rule_features_train: Optional[torch.Tensor] = None,
              rule_features_val: Optional[torch.Tensor] = None,
              max_epochs: int = None) -> Dict:
        max_epochs = max_epochs or NN_CONFIG["max_epochs"]
        history = {"train_loss": [], "val_loss": [], "val_r2": []}
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(max_epochs):
            self.model.train()
            self.optimizer.zero_grad()

            if isinstance(self.model, RuleRegularizedNet):
                y_pred = self.model(X_train, rule_features_train)
            else:
                y_pred = self.model(X_train)

            train_loss = self.criterion(y_pred, y_train)
            train_loss.backward()
            self.optimizer.step()

            self.model.eval()
            with torch.no_grad():
                if isinstance(self.model, RuleRegularizedNet):
                    y_val_pred = self.model(X_val, rule_features_val)
                else:
                    y_val_pred = self.model(X_val)
                val_loss = self.criterion(y_val_pred, y_val)
                ss_res = torch.sum((y_val - y_val_pred) ** 2)
                ss_tot = torch.sum((y_val - torch.mean(y_val)) ** 2)
                val_r2 = 1 - ss_res / (ss_tot + 1e-10)

            history["train_loss"].append(train_loss.item())
            history["val_loss"].append(val_loss.item())
            history["val_r2"].append(val_r2.item())

            # 早停：验证损失连续不进步就停
            if val_loss.item() < best_val_loss - 1e-6:
                best_val_loss = val_loss.item()
                patience_counter = 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return history

    def predict(self, X: torch.Tensor,
                rule_features: Optional[torch.Tensor] = None) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            if isinstance(self.model, RuleRegularizedNet):
                pred = self.model(X, rule_features)
            else:
                pred = self.model(X)
        return pred.numpy()


class MultiClassNet(nn.Module):
    """多元判别网络：多分类MLP，输出n_classes个logits，支持自由z。"""

    def __init__(self, input_dim: int, n_classes: int,
                 hidden_dims: List[int] = None, z_dim: int = None):
        super().__init__()
        hidden_dims = hidden_dims or NN_CONFIG["hidden_dims"]
        z_dim = z_dim or NN_CONFIG["z_dim"]
        self.n_classes = n_classes

        self.free_z = FreeEmbedding(z_dim)
        mlp_input_dim = input_dim + z_dim

        layers = []
        prev_dim = mlp_input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        # 输出层不做softmax，CrossEntropyLoss内置了
        layers.append(nn.Linear(prev_dim, n_classes))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        z = self.free_z(batch_size)
        x_aug = torch.cat([x, z], dim=1)
        return self.mlp(x_aug)


class MultiClassTrainer:
    """多分类网络训练器，交叉熵损失+早停。"""

    def __init__(self, model: MultiClassNet, lr: float = None,
                 patience: int = None, lambda_reg: float = None):
        self.model = model
        self.lr = lr or NN_CONFIG["lr"]
        self.patience = patience or NN_CONFIG["patience"]
        self.lambda_reg = lambda_reg or NN_CONFIG["lambda_reg"]
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr,
                                           weight_decay=self.lambda_reg)
        self.criterion = nn.CrossEntropyLoss()

    def train(self, X_train: torch.Tensor, y_train: torch.Tensor,
              X_val: torch.Tensor, y_val: torch.Tensor,
              max_epochs: int = None) -> Dict:
        max_epochs = max_epochs or NN_CONFIG["max_epochs"]
        history = {"train_loss": [], "val_loss": [], "val_acc": []}
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(max_epochs):
            self.model.train()
            self.optimizer.zero_grad()
            logits = self.model(X_train)
            loss = self.criterion(logits, y_train.long())
            loss.backward()
            self.optimizer.step()

            self.model.eval()
            with torch.no_grad():
                val_logits = self.model(X_val)
                val_loss = self.criterion(val_logits, y_val.long())
                val_pred = torch.argmax(val_logits, dim=1)
                val_acc = (val_pred == y_val.long()).float().mean()

            history["train_loss"].append(loss.item())
            history["val_loss"].append(val_loss.item())
            history["val_acc"].append(val_acc.item())

            # 早停：验证损失连续不进步就停
            if val_loss.item() < best_val_loss - 1e-6:
                best_val_loss = val_loss.item()
                patience_counter = 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        if best_state:
            self.model.load_state_dict(best_state)

        return history

    def predict(self, X: torch.Tensor) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X)
            pred = torch.argmax(logits, dim=1)
        return pred.numpy()


def build_rule_features(tree_paths, X: np.ndarray, n_features: int) -> np.ndarray:
    """把树路径转成0/1规则特征矩阵。"""
    n_samples = X.shape[0]
    n_paths = len(tree_paths)
    rule_features = np.ones((n_samples, n_paths))

    # 向量化实现：每个条件生成布尔掩码，再AND合并
    for j, path in enumerate(tree_paths):
        if not path.conditions:
            continue
        masks = []
        for cond in path.conditions:
            feat_idx, operator, threshold = cond
            col = X[:, feat_idx]
            if operator == "<=":
                masks.append(col <= threshold)
            elif operator == ">":
                masks.append(col > threshold)
        path_mask = np.logical_and.reduce(masks)
        rule_features[:, j] = path_mask.astype(float)

    return rule_features


if __name__ == "__main__":
    print("=== NN拟合模块自测 ===")

    np.random.seed(42)
    torch.manual_seed(42)

    n_samples = 200
    input_dim = 4
    X = np.random.randn(n_samples, input_dim).astype(np.float32)
    y = (3 * X[:, 0] + 2 * X[:, 1] + 0.5 * np.random.randn(n_samples)).astype(np.float32)

    n_train = 160
    X_train = torch.tensor(X[:n_train])
    y_train = torch.tensor(y[:n_train])
    X_val = torch.tensor(X[n_train:])
    y_val = torch.tensor(y[n_train:])

    # 测试精确拟合网络
    print("\n1. 精确拟合网络 (ExactFitNet):")
    exact_net = ExactFitNet(input_dim, hidden_dims=[32, 16])
    trainer1 = NNTrainer(exact_net, lr=0.01, patience=20)
    hist1 = trainer1.train(X_train, y_train, X_val, y_val, max_epochs=100)
    print(f"  最终验证R²: {hist1['val_r2'][-1]:.4f}")

    # 测试可微箱网络
    print("\n2. 可微箱范围网络 (SoftBinningNet):")
    binning_net = SoftBinningNet(input_dim, n_bins=5, hidden_dims=[32, 16])
    trainer2 = NNTrainer(binning_net, lr=0.01, patience=20)
    hist2 = trainer2.train(X_train, y_train, X_val, y_val, max_epochs=100)
    print(f"  最终验证R²: {hist2['val_r2'][-1]:.4f}")
    print(f"  箱边界样本: {binning_net.soft_binning.bin_edges[0, :3].detach().numpy()}")

    # 测试规则正则化网络
    print("\n3. 规则正则化网络 (RuleRegularizedNet):")
    n_rules = 3
    rule_train = torch.randint(0, 2, (n_train, n_rules)).float()
    rule_val = torch.randint(0, 2, (n_samples - n_train, n_rules)).float()
    rule_net = RuleRegularizedNet(input_dim, n_rules, hidden_dims=[32, 16])
    trainer3 = NNTrainer(rule_net, lr=0.01, patience=20)
    hist3 = trainer3.train(X_train, y_train, X_val, y_val,
                            rule_features_train=rule_train,
                            rule_features_val=rule_val,
                            max_epochs=100)
    print(f"  最终验证R²: {hist3['val_r2'][-1]:.4f}")

    print("=== 自测完成 ===")
