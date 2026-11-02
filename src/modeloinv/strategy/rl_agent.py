"""Reinforcement Learning Execution and Sizing Agent for modeloINV.

Provides:
- RLStateBuilder: Compact 9-dimensional state vector constructor.
- PolicyNetwork & PolicyAgent: PyTorch Double Deep Q-Network (DDQN) selecting
  target portfolio weight w_t in [-1.0, 1.0].
- CostPenalizedReward: Transaction cost, slippage, and drawdown-penalized reward.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# Discrete action to target weight mapping
ACTION_WEIGHT_MAP: Dict[int, Optional[float]] = {
    0: 0.0,    # Flat / Exit
    1: 0.25,   # Cautious Long
    2: 0.50,   # Moderate Long
    3: 1.00,   # Full Long
    4: -0.25,  # Cautious Short
    5: -0.50,  # Moderate Short
    6: None,   # Hold current position
}


class RLStateBuilder:
    """Builds a compact, normalized 9-dimensional state vector s_t in R^9."""

    @staticmethod
    def build_state(
        quant_score: float,
        ml_score: float,
        ml_confidence: float,
        adx: float,
        atr: float,
        price: float,
        current_weight: float,
        unrealized_pnl_pct: float,
        drawdown_pct: float,
        holding_time_bars: float = 0.0,
        spread_bps: float = 0.0,
    ) -> np.ndarray:
        """Constructs and validates the 9-dimensional state vector.

        State elements:
            0: quant_score in [-1.0, 1.0]
            1: ml_score in [-1.0, 1.0]
            2: ml_confidence in [0.0, 1.0]
            3: adx_norm in [0.0, 1.0] (adx / 100.0)
            4: vol_ratio in [0.0, 0.1] (atr / price)
            5: current_weight in [-1.0, 1.0]
            6: unrealized_pnl_norm in [-1.0, 1.0] (uPnL / 100.0 or ratio)
            7: drawdown_norm in [0.0, 1.0]
            8: holding_time_norm or spread_norm in [0.0, 1.0]
        """
        # Element 0: Quant Score
        s0 = float(np.clip(quant_score, -1.0, 1.0))

        # Element 1: ML Score
        s1 = float(np.clip(ml_score, -1.0, 1.0))

        # Element 2: ML Confidence
        s2 = float(np.clip(ml_confidence, 0.0, 1.0))

        # Element 3: ADX normalized
        s3 = float(np.clip(adx / 100.0, 0.0, 1.0))

        # Element 4: Volatility ratio (ATR / Price)
        safe_price = max(price, 1e-6)
        s4 = float(np.clip(atr / safe_price, 0.0, 0.1))

        # Element 5: Current portfolio weight
        s5 = float(np.clip(current_weight, -1.0, 1.0))

        # Element 6: Unrealized P&L (if percentage > 1.0, normalize to ratio)
        upnl = unrealized_pnl_pct if abs(unrealized_pnl_pct) <= 1.0 else (unrealized_pnl_pct / 100.0)
        s6 = float(np.clip(upnl, -1.0, 1.0))

        # Element 7: Drawdown percentage
        dd = drawdown_pct if drawdown_pct <= 1.0 else (drawdown_pct / 100.0)
        s7 = float(np.clip(dd, 0.0, 1.0))

        # Element 8: Holding time normalized or spread
        if holding_time_bars > 0:
            s8 = float(np.clip(holding_time_bars / 60.0, 0.0, 1.0))
        else:
            s8 = float(np.clip(spread_bps / 100.0, 0.0, 1.0))

        state = np.array([s0, s1, s2, s3, s4, s5, s6, s7, s8], dtype=np.float32)
        return np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)


class CostPenalizedReward:
    """Calculates realistic execution reward accounting for fees, slippage, and drawdown."""

    def __init__(
        self,
        lambda_fee: float = 1.0,
        lambda_slippage: float = 1.0,
        lambda_drawdown: float = 10.0,
        drawdown_threshold: float = 0.05,
    ):
        self.lambda_fee = lambda_fee
        self.lambda_slippage = lambda_slippage
        self.lambda_drawdown = lambda_drawdown
        self.drawdown_threshold = drawdown_threshold

    def calculate_reward(
        self,
        delta_pnl: float,
        fee: float = 0.0,
        slippage: float = 0.0,
        drawdown: float = 0.0,
        drawdown_penalty: Optional[float] = None,
    ) -> float:
        """Computes cost-penalized reward:

        R_t = delta_pnl - lambda_f * fee - lambda_s * slippage - lambda_dd * max(0, DD - thr)^2
        """
        fee_cost = self.lambda_fee * fee
        slippage_cost = self.lambda_slippage * slippage

        if drawdown_penalty is not None:
            dd_penalty = drawdown_penalty
        else:
            excess_dd = max(0.0, drawdown - self.drawdown_threshold)
            dd_penalty = self.lambda_drawdown * (excess_dd ** 2)

        raw_reward = delta_pnl - fee_cost - slippage_cost - dd_penalty
        # Clip reward to [-10.0, 10.0] to prevent gradient explosion in RL
        return float(np.clip(raw_reward, -10.0, 10.0))


class PolicyNetwork(nn.Module):
    """Deep Q-Network for position sizing decisions."""

    def __init__(self, state_dim: int = 9, action_dim: int = 7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Returns Q-values for each of the discrete actions."""
        return self.net(state)


class PolicyAgent:
    """Double Deep Q-Network (DDQN) Policy Agent for target allocation."""

    def __init__(
        self,
        state_dim: int = 9,
        action_dim: int = 7,
        lr: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.005,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.device = torch.device("cpu")

        self.policy_net = PolicyNetwork(state_dim, action_dim).to(self.device)
        self.target_net = PolicyNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()  # Huber Loss

    def act(
        self,
        state: np.ndarray,
        epsilon: float = 0.0,
        current_weight: float = 0.0,
    ) -> Tuple[int, float]:
        """Chooses an action and resolves it to a target portfolio weight.

        Returns:
            (action_idx, target_weight in [-1.0, 1.0])
        """
        if np.random.rand() < epsilon:
            action = int(np.random.randint(0, self.action_dim))
        else:
            self.policy_net.eval()
            with torch.no_grad():
                s_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
                q_values = self.policy_net(s_t)
                action = int(torch.argmax(q_values, dim=1).item())

        weight = ACTION_WEIGHT_MAP.get(action)
        if weight is None:
            # Action 6 = Hold current position weight
            target_weight = current_weight
        else:
            target_weight = weight

        return action, float(np.clip(target_weight, -1.0, 1.0))

    def update(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ) -> float:
        """Executes a single DDQN update step using Huber loss."""
        self.policy_net.train()

        s_t = torch.from_numpy(states).float().to(self.device)
        a_t = torch.from_numpy(actions).long().unsqueeze(1).to(self.device)
        r_t = torch.from_numpy(rewards).float().unsqueeze(1).to(self.device)
        s_next = torch.from_numpy(next_states).float().to(self.device)
        d_t = torch.from_numpy(dones).float().unsqueeze(1).to(self.device)

        # Current Q-values: Q(s, a)
        curr_q = self.policy_net(s_t).gather(1, a_t)

        # DDQN Target: Q_target(s', argmax_a Q(s', a))
        with torch.no_grad():
            next_actions = self.policy_net(s_next).argmax(dim=1, keepdim=True)
            next_q = self.target_net(s_next).gather(1, next_actions)
            target_q = r_t + (1.0 - d_t) * self.gamma * next_q

        loss = self.loss_fn(curr_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Soft update target network: theta_target = tau * theta + (1 - tau) * theta_target
        for p_target, p_curr in zip(self.target_net.parameters(), self.policy_net.parameters()):
            p_target.data.copy_(self.tau * p_curr.data + (1.0 - self.tau) * p_target.data)

        return float(loss.item())

    def save(self, filepath: str) -> None:
        """Persists model checkpoint to disk."""
        dir_name = os.path.dirname(filepath)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, filepath)

    def load(self, filepath: str) -> None:
        """Restores model checkpoint from disk."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
