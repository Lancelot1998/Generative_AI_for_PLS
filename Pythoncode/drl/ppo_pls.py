# ppo_pls.py
# -*- coding: utf-8 -*-

import os
import json
import math
import random
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from channel.semantic_channelpro import SemanticChannelModel  # 保持和你现有结构一致


# =========================
# 1. Alice 的 PLS 环境
# =========================

class AlicePLSEnv:
    """
    单场景环境：
    - 给定:
        bob_models: {k: SemanticChannelModel}  (合法链路语义模型)
        eve_models: {k: SemanticChannelModel}  (窃听链路语义模型)
        k_values:   [k1, k2, ...]
        P_dB_list:  [P1_dB, P2_dB, ...]
        distance_bob, distance_eve
    - 动作: 一个整数 a，对应 (P_dB, k)
    - 状态: 简单取 [d_bob_norm, d_eve_norm]
    - 奖励: reward = xi_B - lambda_e * xi_E
      如果设置 bob_xi_min，则对 xi_B < bob_xi_min 的动作给予惩罚
    """

    def __init__(
        self,
        bob_models: Dict[int, SemanticChannelModel],
        eve_models: Dict[int, SemanticChannelModel],
        k_values: List[int],
        P_dB_list: List[float],
        distance_bob: float,
        distance_eve: float,
        sigma2_bob: float = 0.1,
        sigma2_eve: float = 0.1,
        n_mc: int = 200,
        lambda_e: float = 1.0,
        bob_xi_min: Optional[float] = None,
        d_norm: float = 100.0,
    ):
        self.bob_models = bob_models
        self.eve_models = eve_models
        self.k_values = sorted(k_values)
        self.P_dB_list = P_dB_list
        self.distance_bob = distance_bob
        self.distance_eve = distance_eve
        self.sigma2_bob = sigma2_bob
        self.sigma2_eve = sigma2_eve
        self.n_mc = n_mc
        self.lambda_e = lambda_e
        self.bob_xi_min = bob_xi_min
        self.d_norm = d_norm  # 距离归一化因子

        self.num_P = len(self.P_dB_list)
        self.num_k = len(self.k_values)
        self.action_dim = self.num_P * self.num_k
        self.state_dim = 2  # [d_bob_norm, d_eve_norm]

        # 方便后面 decode
        self._last_info = None

    def reset(self) -> np.ndarray:
        """
        这里场景固定，不重新采样 Bob / Eve，只重置为同一个 state。
        如果你以后想每个 episode 随机 Bob，可以改写这里。
        """
        d_bob_norm = self.distance_bob / self.d_norm
        d_eve_norm = self.distance_eve / self.d_norm
        state = np.array([d_bob_norm, d_eve_norm], dtype=np.float32)
        return state

    def _decode_action(self, action: int) -> Tuple[float, int]:
        """
        整数动作 -> (P_dB, k)
        """
        p_idx = action // self.num_k
        k_idx = action % self.num_k
        P_dB = self.P_dB_list[p_idx]
        k_val = self.k_values[k_idx]
        return P_dB, k_val

    def step(self, action: int):
        """
        执行动作:
          - 选择 (P_dB, k)
          - 计算 Bob/Eve 语义相似度
          - 计算 reward = diff + penalty，其中
              diff = xi_B - xi_E
              penalty = -max(0, bob_xi_min - xi_B)
        """
        P_dB, k_val = self._decode_action(action)
        P_linear = 10.0 ** (P_dB / 10.0)  # dB -> 线性功率

        bob_model = self.bob_models[k_val]
        eve_model = self.eve_models[k_val]

        xi_B = bob_model.semantic_similarity(
            P=P_linear,
            distance=self.distance_bob,
            sigma2=self.sigma2_bob,
            n_mc=self.n_mc
        )
        xi_E = eve_model.semantic_similarity(
            P=P_linear,
            distance=self.distance_eve,
            sigma2=self.sigma2_eve,
            n_mc=self.n_mc
        )

        # 语义差、比值
        diff = xi_B - xi_E
        ratio = xi_B / (xi_E + 1e-6)

        # --- 软约束部分：对 Bob 的最低语义质量 ---
        penalty = 0.0
        if (self.bob_xi_min is not None) and (xi_B < self.bob_xi_min):
            # 例如 bob_xi_min = 0.5, xi_B = 0.3 -> penalty = -0.2
            penalty = - (self.bob_xi_min - xi_B)

        reward = diff + penalty

        # 可以根据需要 clamp 一下，防止 reward 爆炸
        reward = max(min(reward, 1.0), -1.0)

        done = True  # 单步 episode
        next_state = self.reset()

        info = {
            "P_dB": P_dB,
            "k": k_val,
            "xi_B": xi_B,
            "xi_E": xi_E,
            "diff": diff,
            "ratio": ratio,
            "penalty": penalty,
            "reward": reward
        }
        self._last_info = info
        return next_state, reward, done, info

    
    def step1(self, action: int):
        """
        执行动作:
          - 选择 (P_dB, k)
          - 计算 Bob/Eve 语义相似度
          - 计算 reward = xi_B - lambda_e * xi_E （带可选 Bob 约束）
        """
        P_dB, k_val = self._decode_action(action)
        P_linear = 10.0 ** (P_dB / 10.0)  # dB -> 线性功率

        bob_model = self.bob_models[k_val]
        eve_model = self.eve_models[k_val]

        xi_B = bob_model.semantic_similarity(
            P=P_linear,
            distance=self.distance_bob,
            sigma2=self.sigma2_bob,
            n_mc=self.n_mc
        )
        xi_E = eve_model.semantic_similarity(
            P=P_linear,
            distance=self.distance_eve,
            sigma2=self.sigma2_eve,
            n_mc=self.n_mc
        )

        # 语义差、比值
        diff = xi_B - xi_E
        ratio = xi_B / (xi_E + 1e-6)

        # Bob 质量约束：如果不达标，可给一个较大负奖励
        if (self.bob_xi_min is not None) and (xi_B < self.bob_xi_min):
            reward = -1.0  # 你可以根据需要调节
        else:
            reward = diff - self.lambda_e * 0.0  # 目前等价于 diff，也可以改为 xi_B - λ xi_E

        done = True  # 单步 episode
        next_state = self.reset()

        info = {
            "P_dB": P_dB,
            "k": k_val,
            "xi_B": xi_B,
            "xi_E": xi_E,
            "diff": diff,
            "ratio": ratio,
            "reward": reward
        }
        self._last_info = info
        return next_state, reward, done, info


# =========================
# 2. PPO 网络 & Buffer
# =========================

class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        raise NotImplementedError

    def act(self, state):
        logits = self.actor(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        logprob = dist.log_prob(action)
        value = self.critic(state)
        return action, logprob, value

    def evaluate_actions(self, states, actions):
        logits = self.actor(states)
        dist = torch.distributions.Categorical(logits=logits)
        logprobs = dist.log_prob(actions)
        entropy = dist.entropy()
        values = self.critic(states).squeeze(-1)
        return logprobs, entropy, values


class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def clear(self):
        self.__init__()


# =========================
# 3. PPO Agent & 训练函数
# =========================

class PPOAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        K_epochs: int = 4,
        eps_clip: float = 0.2,
        device: Optional[torch.device] = None,
    ):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.buffer = RolloutBuffer()

        self.mse_loss = nn.MSELoss()

    def select_action(self, state: np.ndarray) -> int:
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, logprob, value = self.policy.act(state_t)
        self.buffer.states.append(state_t.squeeze(0).cpu().numpy())
        self.buffer.actions.append(int(action.cpu().item()))
        self.buffer.logprobs.append(float(logprob.cpu().item()))
        self.buffer.values.append(float(value.cpu().item()))
        return int(action.cpu().item())

    def update(self):
        # 把数据转成 tensor
        states = torch.tensor(self.buffer.states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(self.buffer.actions, dtype=torch.long, device=self.device)
        old_logprobs = torch.tensor(self.buffer.logprobs, dtype=torch.float32, device=self.device)
        rewards = torch.tensor(self.buffer.rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(self.buffer.dones, dtype=torch.float32, device=self.device)
        values = torch.tensor(self.buffer.values, dtype=torch.float32, device=self.device)

        # 单步任务，γ 影响不大，这里仍按一般 PPO 写
        # 计算 returns（G_t）和 advantages
        returns = []
        G = 0.0
        for r, d in zip(reversed(self.buffer.rewards), reversed(self.buffer.dones)):
            G = r + self.gamma * G * (1.0 - d)
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)

        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 多轮 epoch 更新
        for _ in range(self.K_epochs):
            logprobs, entropy, state_values = self.policy.evaluate_actions(states, actions)
            # ratio
            ratios = torch.exp(logprobs - old_logprobs)

            # surrogate loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = self.mse_loss(state_values, returns)
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # 清空 buffer
        self.buffer.clear()


def train_ppo(
    env: AlicePLSEnv,
    agent: PPOAgent,
    num_episodes: int = 1000,
    update_interval: int = 32,
    results_dir: str = "./results/ppo",
    run_name: str = "run1"
):
    os.makedirs(results_dir, exist_ok=True)
    log_json_path = os.path.join(results_dir, f"{run_name}_log.json")
    summary_txt_path = os.path.join(results_dir, f"{run_name}_summary.txt")

    episode_rewards = []
    episode_diff = []
    episode_ratio = []
    episode_xi_B = []
    episode_xi_E = []

    best_reward = -1e9
    best_info = None

    state = env.reset()

    for ep in range(num_episodes):
        # 单步 episode
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)

        # 记录到 buffer
        agent.buffer.rewards.append(reward)
        agent.buffer.dones.append(float(done))

        # 记录日志
        episode_rewards.append(reward)
        episode_diff.append(info["diff"])
        episode_ratio.append(info["ratio"])
        episode_xi_B.append(info["xi_B"])
        episode_xi_E.append(info["xi_E"])

        # 最优策略追踪
        if reward > best_reward:
            best_reward = reward
            best_info = info.copy()

        # 更新 PPO
        if (ep + 1) % update_interval == 0:
            agent.update()

        state = env.reset()

        if (ep + 1) % 50 == 0:
            print(f"[Episode {ep+1}] reward={reward:.4f}, "
                  f"xi_B={info['xi_B']:.4f}, xi_E={info['xi_E']:.4f}, "
                  f"P_dB={info['P_dB']:.2f}, k={info['k']}")

    # 保存 json 日志
    log_data = {
        "episode_rewards": episode_rewards,
        "episode_diff": episode_diff,
        "episode_ratio": episode_ratio,
        "episode_xi_B": episode_xi_B,
        "episode_xi_E": episode_xi_E,
        "best_strategy": best_info,
    }
    with open(log_json_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)

    # 保存 summary txt
    with open(summary_txt_path, "w", encoding="utf-8") as f:
        f.write("=== PPO Training Summary ===\n")
        f.write(f"Num episodes: {num_episodes}\n")
        f.write(f"Best reward: {best_reward:.6f}\n")
        if best_info is not None:
            f.write(f"Best P_dB: {best_info['P_dB']:.4f}\n")
            f.write(f"Best k: {best_info['k']}\n")
            f.write(f"Best xi_B: {best_info['xi_B']:.6f}\n")
            f.write(f"Best xi_E: {best_info['xi_E']:.6f}\n")
            f.write(f"Best diff (xi_B - xi_E): {best_info['diff']:.6f}\n")
            f.write(f"Best ratio (xi_B / xi_E): {best_info['ratio']:.6f}\n")

    print(f"\nPPO 训练结束，日志保存在：\n  {log_json_path}\n  {summary_txt_path}")

# ppo_pls.py 末尾新增

# ====== teacher frequency scheduling (ADD) ======
# teacher_builder: 一个函数，输入 env，返回 teacher_actions(list[(P_dB,k)]) 或者 None
# teacher_refresh_every: 每 N 个 episode 刷新一次 teacher_actions

# ====== end teacher scheduling ======


def train_ppo_multi_env(
    agent: PPOAgent,
    make_env,
    num_episodes: int = 100000,
    refresh_every: int = 1,
    update_interval: int = 32,
    results_dir: str = "./results/ppo_multi",
    run_name: str = "run1_multi",
    teacher_builder=None,
    teacher_refresh_every: int | None = None,
):
    """
    多环境版本训练：
    - make_env: 一个无参函数，每次调用返回一个新的 AlicePLSEnv 实例
    - num_episodes: 训练总轮数
    - refresh_every:
        = 1    -> 每一轮都重新创建环境
        = 5000 -> 每 5000 轮创建一个新环境
    - update_interval: 每多少步做一次 PPO 更新（和之前一样）

    注意：
    - 假设所有 env 的 state_dim 和 action_dim 都一致
    """

    os.makedirs(results_dir, exist_ok=True)
    log_json_path = os.path.join(results_dir, f"{run_name}_log.json")
    summary_txt_path = os.path.join(results_dir, f"{run_name}_summary.txt")

    episode_rewards = []
    episode_diff = []
    episode_ratio = []
    episode_xi_B = []
    episode_xi_E = []
    episode_env_id = []   # 记录这是第几个环境

    best_reward = -1e9
    best_info = None

    env = None
    current_env_id = -1

    def _maybe_refresh_teacher(env, episode_idx: int):
        if teacher_builder is None or teacher_refresh_every is None:
            return
        if teacher_refresh_every <= 0:
            return
        if (episode_idx % teacher_refresh_every) != 0:
            return
    
        new_actions = teacher_builder(env)
        if new_actions is None:
            return
    
        # 兼容两种写法：env.set_teacher_actions(...) 或直接 env.teacher_actions = ...
        if hasattr(env, "set_teacher_actions") and callable(getattr(env, "set_teacher_actions")):
            env.set_teacher_actions(new_actions)
        else:
            setattr(env, "teacher_actions", new_actions)
    
    for ep in range(num_episodes):
        # === 1. 按规则刷新环境 ===
        if ep == 0 or (refresh_every > 0 and (ep % refresh_every) == 0):
            env = make_env()
            current_env_id += 1
            # 这里假设 env.reset() 返回 state（形状固定）

            #_maybe_refresh_teacher(env, episode)
            state = env.reset()
        else:
            # 单步任务，每 episode 都 reset 一下同一个 env 即可

            #_maybe_refresh_teacher(env, episode)
            state = env.reset()

        _maybe_refresh_teacher(env, ep)
        # === 2. 与单环境版本相同：采样动作、交互一步 ===
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)

        agent.buffer.rewards.append(reward)
        agent.buffer.dones.append(float(done))

        # 记录日志
        episode_rewards.append(reward)
        episode_diff.append(info["diff"])
        episode_ratio.append(info["ratio"])
        episode_xi_B.append(info["xi_B"])
        episode_xi_E.append(info["xi_E"])
        episode_env_id.append(current_env_id)

        # 维护全局 best 策略
        if reward > best_reward:
            best_reward = reward
            best_info = info.copy()
            best_info["env_id"] = current_env_id

        # 定期更新 PPO
        if (ep + 1) % update_interval == 0:
            agent.update()

        if (ep + 1) % 500 == 0:
            print(
                f"[Episode {ep+1}] env_id={current_env_id}, "
                f"reward={reward:.4f}, "
                f"xi_B={info['xi_B']:.4f}, xi_E={info['xi_E']:.4f}, "
                f"P_dB={info['P_dB']:.2f}, k={info['k']}"
            )

    # === 3. 保存 json 日志 ===
    log_data = {
        "episode_rewards": episode_rewards,
        "episode_diff": episode_diff,
        "episode_ratio": episode_ratio,
        "episode_xi_B": episode_xi_B,
        "episode_xi_E": episode_xi_E,
        "episode_env_id": episode_env_id,
        "best_strategy": best_info,
    }
    with open(log_json_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)

    # === 4. 保存 summary txt ===
    with open(summary_txt_path, "w", encoding="utf-8") as f:
        f.write("=== PPO Multi-Env Training Summary ===\n")
        f.write(f"Num episodes: {num_episodes}\n")
        f.write(f"Refresh every: {refresh_every} episodes\n")
        f.write(f"Best reward: {best_reward:.6f}\n")
        if best_info is not None:
            f.write(f"Best env_id: {best_info.get('env_id', -1)}\n")
            f.write(f"Best P_dB: {best_info['P_dB']:.4f}\n")
            f.write(f"Best k: {best_info['k']}\n")
            f.write(f"Best xi_B: {best_info['xi_B']:.6f}\n")
            f.write(f"Best xi_E: {best_info['xi_E']:.6f}\n")
            f.write(f"Best diff (xi_B - xi_E): {best_info['diff']:.6f}\n")
            f.write(f"Best ratio (xi_B / xi_E): {best_info['ratio']:.6f}\n")

    print(f"\nPPO 多环境训练结束，日志保存在：\n  {log_json_path}\n  {summary_txt_path}")

