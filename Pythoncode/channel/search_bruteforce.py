# search_bruteforce.py
from typing import Dict, List, Tuple
from channel.semantic_channelpro import SemanticChannelModel


def brute_force_power_and_k(
    bob_models: Dict[int, SemanticChannelModel],
    eve_models: Dict[int, SemanticChannelModel],
    P_list: List[float],
    distance_bob: float,
    distance_eve: float,
    sigma2_bob: float = 0.1,
    sigma2_eve: float = 0.1,
    n_mc: int = 200,
    lambda_e: float = 1.0,
    bob_xi_min: float | None = None
) -> Tuple[float, int, float, float, float]:
    """
    穷举 Alice 的动作 (P, k)：
      目标: 最大化 reward = xi_B - lambda_e * xi_E
      如果 bob_xi_min 不为 None，则要求 xi_B >= bob_xi_min

    返回:
      best_P, best_k, best_xi_B, best_xi_E, best_reward
    """
    best_reward = -1e9
    best_P = None
    best_k = None
    best_xi_B = None
    best_xi_E = None

    for k, bob_model in bob_models.items():
        eve_model = eve_models[k]  # 假设 k 集合相同

        for P in P_list:
            xi_B = bob_model.semantic_similarity(
                P=P, distance=distance_bob,
                sigma2=sigma2_bob, n_mc=n_mc
            )
            xi_E = eve_model.semantic_similarity(
                P=P, distance=distance_eve,
                sigma2=sigma2_eve, n_mc=n_mc
            )

            if bob_xi_min is not None and xi_B < bob_xi_min:
                continue  # 不满足 Bob 最低语义质量要求

            reward = xi_B - lambda_e * xi_E

            if reward > best_reward:
                best_reward = reward
                best_P = P
                best_k = k
                best_xi_B = xi_B
                best_xi_E = xi_E

    return best_P, best_k, best_xi_B, best_xi_E, best_reward
