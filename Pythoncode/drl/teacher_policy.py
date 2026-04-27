# teacher_policy.py
from typing import List, Tuple, Dict
import numpy as np

from channel.semantic_channelpro import SemanticChannelModel


def build_teacher_actions_semantic(
    bob_models: Dict[int, SemanticChannelModel],
    eve_models: Dict[int, SemanticChannelModel],
    P_dB_list: List[float],
    k_values: List[int],
    distance_bob: float,
    distance_eve: float,
    sigma2_bob: float,
    sigma2_eve: float,
    lambda_e: float = 1.0,
    n_mc: int = 200,
    top_ratio: float = 0.2,
) -> List[Tuple[float, int]]:
    """
    语义“teacher”的动作建议：
      - 枚举所有 (P_dB, k) 组合
      - 对每个动作算 teacher reward:
            r_teacher = xi_B - lambda_e * xi_E
      - 按 r_teacher 排序，取前 top_ratio 比例的动作作为推荐集合

    返回:
      teacher_actions: List[(P_dB, k)]
    """

    actions = []
    rewards = []

    for P_dB in P_dB_list:
        # P_dB -> 线性 P
        P_linear = 10.0 ** (P_dB / 10.0)

        for k in k_values:
            # 对应 k 的语义信道模型
            scm_bob = bob_models[k]
            scm_eve = eve_models[k]

            # 计算 Bob / Eve 的平均语义相似度
            xi_B = scm_bob.semantic_similarity(
                P=P_linear,
                distance=distance_bob,
                sigma2=sigma2_bob,
                n_mc=n_mc,
            )
            xi_E = scm_eve.semantic_similarity(
                P=P_linear,
                distance=distance_eve,
                sigma2=sigma2_eve,
                n_mc=n_mc,
            )

            r_teacher = xi_B - lambda_e * xi_E

            actions.append((P_dB, k))
            rewards.append(r_teacher)

    rewards = np.array(rewards)
    actions = np.array(actions, dtype=object)

    # 选出 top_ratio 比例的动作（例如 0.2 就是前 20%）
    n = len(actions)
    n_select = max(1, int(n * top_ratio))
    idx_sorted = np.argsort(-rewards)  # 从大到小排序
    idx_top = idx_sorted[:n_select]

    teacher_actions = [tuple(actions[i]) for i in idx_top]

    print(f"[Teacher] 从 {n} 个动作中选出 top-{n_select} 作为推荐动作")
    for (P_dB, k), r_val in zip(teacher_actions[:5], rewards[idx_top][:5]):
        print(f"   (P_dB={P_dB:.1f}, k={k}) -> r_teacher={r_val:.4f}")

    return teacher_actions
