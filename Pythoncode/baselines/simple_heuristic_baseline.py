# weak_heuristic_baseline.py
# -*- coding: utf-8 -*-

from typing import Tuple, Sequence, Optional
import numpy as np


class WeakHeuristicBaseline:
    """
    目标：random < weak_heuristic << PPO

    核心策略：
      - 以 prob_guided 的概率：用非常粗的规则，从一个小候选集合里“随机挑一个更合理的动作”
      - 以 1 - prob_guided 的概率：完全 random

    这样：
      - 下限就是 random（prob_guided=0 时等价 random）
      - 只要 prob_guided 取 0.1~0.3，通常会比 random 稍好，但不会像小 greedy 那样过强
    """

    def __init__(
        self,
        prob_guided: float = 0.2,   # 建议 0.1~0.3，越大越强
        seed: Optional[int] = None,
    ):
        self.prob_guided = float(prob_guided)
        self.rng = np.random.default_rng(seed)

    def select_action(self, env) -> Tuple[float, int]:
        P_list = [float(p) for p in env.P_dB_list]
        K_list = [int(k) for k in env.k_values]

        # 1) 大概率保持随机（保证下限接近 random）
        if self.rng.random() > self.prob_guided:
            p = self.rng.choice(P_list)
            k = self.rng.choice(K_list)
            return float(p), int(k)

        # 2) 小概率启发式：只根据距离选一个“更可能合理”的档位，然后在候选中随机抽
        st = env.reset()
        d_norm = float(getattr(env, "d_norm", 100.0))
        d_eve = float(st[1]) * d_norm

        # 规则：Eve 越近 -> 倾向低功率、偏小/中 k；Eve 越远 -> 给一点功率、偏中/大 k
        if d_eve <= 10.0:         # 高风险
            p_targets = [0.0, 2.0, 4.0]
            k_targets = [K_list[0], K_list[min(1, len(K_list)-1)], K_list[len(K_list)//2]]
        elif d_eve <= 50.0:       # 中风险
            p_targets = [2.0, 4.0, 6.0]
            k_targets = [K_list[min(1, len(K_list)-1)], K_list[len(K_list)//2]]
        else:                     # 低风险
            p_targets = [4.0, 8.0, 12.0]
            k_targets = [K_list[len(K_list)//2], K_list[-1]]

        p_cands = self._closest_values(P_list, p_targets, max_keep=2)   # 只保留很少候选，避免太强
        k_cands = self._valid_ks(K_list, k_targets, max_keep=2)

        p = self.rng.choice(p_cands)
        k = self.rng.choice(k_cands)
        return float(p), int(k)

    # ---------- helpers ----------

    def _closest_values(self, values: Sequence[float], targets: Sequence[float], max_keep: int = 2):
        arr = np.asarray(values, dtype=float)
        out = []
        for t in targets:
            out.append(float(arr[int(np.argmin(np.abs(arr - float(t))))]))
        # 去重保序
        dedup = []
        for x in out:
            if x not in dedup:
                dedup.append(x)
        return dedup[:max_keep] if dedup else [float(values[len(values)//2])]

    def _valid_ks(self, k_list: Sequence[int], targets: Sequence[int], max_keep: int = 2):
        valid = []
        s = set(int(k) for k in k_list)
        for t in targets:
            if int(t) in s and int(t) not in valid:
                valid.append(int(t))
        if not valid:
            valid = [int(k_list[len(k_list)//2])]
        return valid[:max_keep]
