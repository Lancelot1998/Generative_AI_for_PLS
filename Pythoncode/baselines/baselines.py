# baselines.py
# -*- coding: utf-8 -*-

import random
from typing import Tuple, Sequence, Dict, Any, Optional, List


def db_to_linear(P_dB: float) -> float:
    """P_dB -> 线性功率 P"""
    return float(10.0 ** (float(P_dB) / 10.0))


def eval_xi_pair_from_env(env, P_dB: float, k: int) -> Tuple[float, float]:
    """
    不依赖 env.evaluate_action，直接用 env 的语义信道模型计算 (xi_B, xi_E)

    依赖 env 拥有以下字段：
      - env.bob_models: Dict[int, SemanticChannelModel]
      - env.eve_models: Dict[int, SemanticChannelModel]
      - env.distance_bob, env.distance_eve
      - env.sigma2_bob, env.sigma2_eve
      - env.n_mc
    """
    if k not in env.bob_models:
        raise KeyError(f"k={k} not in env.bob_models keys={list(env.bob_models.keys())}")
    if k not in env.eve_models:
        raise KeyError(f"k={k} not in env.eve_models keys={list(env.eve_models.keys())}")

    P_lin = db_to_linear(P_dB)

    xi_B = env.bob_models[k].semantic_similarity(
        P=P_lin, distance=env.distance_bob, sigma2=env.sigma2_bob, n_mc=env.n_mc
    )
    xi_E = env.eve_models[k].semantic_similarity(
        P=P_lin, distance=env.distance_eve, sigma2=env.sigma2_eve, n_mc=env.n_mc
    )
    return float(xi_B), float(xi_E)


class RandomBaseline:
    """随机策略：随机选 (P_dB, k)"""

    def select_action(self, env) -> Tuple[float, int]:
        P_dB = random.choice(list(env.P_dB_list))
        k = random.choice(list(env.k_values))
        return float(P_dB), int(k)


class GreedyBaseline:
    """
    贪心策略（oracle）：
      - 遍历所有动作
      - 计算 reward = xi_B - lambda_e * xi_E
      - 若设置 bob_xi_min：过滤 xi_B < bob_xi_min
      - 取 reward 最大的动作
    """

    def select_action(self, env) -> Tuple[float, int]:
        best_reward = -1e18
        best_action = None

        for P_dB in env.P_dB_list:
            for k in env.k_values:
                xi_B, xi_E = eval_xi_pair_from_env(env, P_dB=float(P_dB), k=int(k))

                if getattr(env, "bob_xi_min", None) is not None:
                    if xi_B < float(env.bob_xi_min):
                        continue

                reward = xi_B - float(env.lambda_e) * xi_E
                if reward > best_reward:
                    best_reward = reward
                    best_action = (float(P_dB), int(k))

        if best_action is None:
            P_dB = float(env.P_dB_list[len(env.P_dB_list) // 2])
            k = int(env.k_values[len(env.k_values) // 2])
            best_action = (P_dB, k)

        return best_action


class ThresholdHeuristicBaseline:
    """
    一个“比随机强，但通常不如 PPO”的弱启发式 baseline：

    思路：
      1) 对每个 k：
         - 用二分搜索在 P_dB_list 中找满足 xi_B >= bob_xi_min 的最小功率 P*
         - 只在 P* 及其邻近（P*+1 step）评估 reward
      2) 选 reward 最大的 (P_dB, k)

    优点：
      - 不需要遍历所有动作（只需 O(|k| log|P|) 次 xi 评估 + 少量邻域评估）
      - 通常显著优于 random（因为会尽量满足 Bob 质量）
      - 通常弱于 PPO（PPO 可学到更复杂的策略/更鲁棒）
    """

    def __init__(self, neighbor_steps: int = 1, max_eval_per_k: int = 2):
        """
        neighbor_steps: 在 P* 附近最多额外试几个功率点（例如 1 表示试 P* 和 P*+1）
        max_eval_per_k: 每个 k 最多评估多少个功率点（控制计算量）
        """
        self.neighbor_steps = int(max(0, neighbor_steps))
        self.max_eval_per_k = int(max(1, max_eval_per_k))

    def select_action(self, env) -> Tuple[float, int]:
        P_sorted = sorted([float(p) for p in env.P_dB_list])
        k_sorted = sorted([int(k) for k in env.k_values])
        lam = float(getattr(env, "lambda_e", 1.0))
        bob_min = getattr(env, "bob_xi_min", None)
        bob_min = None if bob_min is None else float(bob_min)

        # 如果没有 bob_xi_min，就用一个简单规则：
        # - 取中等功率
        # - 取中等 k
        if bob_min is None:
            P_dB = P_sorted[len(P_sorted) // 3]  # 偏低一些，减少 Eve
            k = k_sorted[len(k_sorted) // 2]
            return float(P_dB), int(k)

        best_reward = -1e18
        best_action: Optional[Tuple[float, int]] = None

        for k in k_sorted:
            # 1) 二分找最小满足 xi_B >= bob_min 的 P_dB 索引
            idx = self._binary_search_min_power(env, P_sorted, k, bob_min)

            # 2) 在 idx 附近评估少量点
            cand_indices = [idx]
            for t in range(1, self.neighbor_steps + 1):
                cand_indices.append(min(len(P_sorted) - 1, idx + t))

            # 控制每个 k 的评估数量
            cand_indices = cand_indices[: self.max_eval_per_k]

            for j in cand_indices:
                P_dB = P_sorted[j]
                xi_B, xi_E = eval_xi_pair_from_env(env, P_dB=P_dB, k=k)
                # 若仍不满足（例如 MC 抖动），就跳过
                if xi_B < bob_min:
                    continue

                reward = xi_B - lam * xi_E
                if reward > best_reward:
                    best_reward = reward
                    best_action = (float(P_dB), int(k))

        # 兜底：如果所有 k 都找不到满足 bob_min 的动作，就选最大发电功率+最大k
        if best_action is None:
            return float(P_sorted[-2]), int(k_sorted[-2])

        return best_action

    @staticmethod
    def _binary_search_min_power(env, P_sorted: List[float], k: int, bob_min: float) -> int:
        """
        二分找最小的 idx，使得 xi_B(P_sorted[idx], k) >= bob_min
        若全部不满足，则返回最后一个 idx（最大功率）
        """
        lo, hi = 0, len(P_sorted) - 1
        ans = hi

        while lo <= hi:
            mid = (lo + hi) // 2
            xi_B, _ = eval_xi_pair_from_env(env, P_dB=P_sorted[mid], k=k)

            if xi_B >= bob_min:
                ans = mid
                hi = mid - 1
            else:
                lo = mid + 1

        return int(ans)
