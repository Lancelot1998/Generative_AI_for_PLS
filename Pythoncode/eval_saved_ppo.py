# eval_saved_ppo.py
# -*- coding: utf-8 -*-

import os
import json
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from baselines.simple_heuristic_baseline import WeakHeuristicBaseline

import numpy as np
import torch
import matplotlib.pyplot as plt

from channel.env_legit_link import LegitLinkGenerator
from channel.semantic_channelpro import SemanticChannelModel
from drl.ppo_pls import AlicePLSEnv, PPOAgent

# ====== 关键：导入你的 baseline（按你项目结构二选一）======
# 方案 A：如果你是 baselines/baselines.py
try:
    from baselines.baselines import ThresholdHeuristicBaseline, RandomBaseline, eval_xi_pair_from_env
except Exception:
    # 方案 B：如果 baselines.py 就在同级目录
    from baselines import ThresholdHeuristicBaseline, RandomBaseline, eval_xi_pair_from_env


# =========================
# 0) 配置区：路径 + 评估参数
# =========================

PPO_PATH_LLM = "/models/Qwen/lyq_data/pythoncode/DeepSCmaster/results/ppo_Qwen_z/ppo_llm_agent_final.pth"
PPO_PATH_NOLLM = "/models/Qwen/lyq_data/pythoncode/DeepSCmaster/results/ppo_random_z/ppo_nollm_agent_final.pth"
PPO_PATH_LLM_TEACHER = "/models/Qwen/lyq_data/pythoncode/DeepSCmaster/results/ppo_Qwen_teacher_z/ppo_llm_teacher_agent_final.pth"

OUT_DIR = "./results/eval_saved_ppo"
OUT_JSON = os.path.join(OUT_DIR, "eval_saved_ppo_5levels.json")
OUT_PNG_PREFIX = os.path.join(OUT_DIR, "bar_level")  # 输出: bar_level_1.png ... bar_level_5.png

N_TRIALS_PER_LEVEL = 100

# 动作集合（建议跟训练一致）
K_VALUES = [4, 8, 16, 32, 64]
P_DB_LIST = [i for i in range(0, 21, 2)]  # 0,2,...,20 dB

# 环境参数（建议跟训练一致）
SIGMA2_BOB = 0.1
SIGMA2_EVE = 0.1
N_MC = 200
LAMBDA_E = 1.0
BOB_XI_MIN = 0.5
D_NORM = 100.0
def plot_bar_all_levels1(level_summaries: Dict[str, Dict], out_png: str):
    """
    level_summaries:
      {
        "1": { "random": {...}, "heuristic_threshold": {...}, ... },
        "2": { ... },
        ...
        "5": { ... }
      }
    """

    levels = [1, 2, 3, 4, 5]
    methods = [
        ("random", "Random"),
        ("heuristic_threshold", "Heuristic"),
        ("ppo_nollm", "PPO(NoLLM)"),
        ("ppo_llm", "PPO(LLM)"),
        ("ppo_llm_teacher", "PPO(LLM+Teacher)"),
    ]

    n_levels = len(levels)
    n_methods = len(methods)

    x = np.arange(n_levels)
    width = 0.16  # 每个柱子的宽度

    plt.figure(figsize=(10, 5))

    for i, (key, label) in enumerate(methods):
        means = [
            level_summaries[str(l)]["summary"][key]["mean_reward"]
            for l in levels
        ]

        plt.bar(
            x + (i - (n_methods - 1) / 2) * width,
            means,
            width=width,
            capsize=3,
            label=label,
        )

    plt.xticks(x, levels)
    plt.xlabel("Difficulty Level")
    plt.ylabel("Average Reward")
    plt.title("Cross-Scenario Performance Comparison")
    plt.legend(ncol=3)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# =========================
# 1) 困难度 -> Eve 距离分布映射
# =========================

@dataclass
class DifficultyConfig:
    near_range: Tuple[float, float]
    far_range: Tuple[float, float]
    near_prob: float


def difficulty_to_config(level: int) -> DifficultyConfig:
    """
    level: 1(最简单) -> 5(最困难)
    越困难 -> Eve 越可能靠近 Alice（near_prob 更大，near_range 更小）
    """
    level = int(level)
    if level < 1 or level > 5:
        raise ValueError("difficulty level must be in {1,2,3,4,5}")

    configs = {
        1: DifficultyConfig(near_range=(100.0, 120.0),  far_range=(200.0, 250.0), near_prob=0.10),
        2: DifficultyConfig(near_range=(80.0, 100.0),  far_range=(180.0, 230.0),  near_prob=0.25),
        3: DifficultyConfig(near_range=(60.0, 80.0),   far_range=(160.0, 210.0),  near_prob=0.50),
        4: DifficultyConfig(near_range=(5.0, 25.0),   far_range=(90.0, 110.0),  near_prob=0.70),
        5: DifficultyConfig(near_range=(0.0, 20.0),   far_range=(80.0, 100.0),   near_prob=0.85),
    }
    return configs[level]


# =========================
# 2) 纯随机 Eve 生成（复用你的语义信道模型参数）
# =========================

def build_random_eve_models(
    alice_pos: Tuple[float, float],
    k_to_params: Dict[int, Dict[str, float]],
    legit_gen: LegitLinkGenerator,
    cfg: DifficultyConfig,
) -> Tuple[Dict[int, SemanticChannelModel], Tuple[float, float], float, str]:
    """
    返回：
      eve_models: {k: SemanticChannelModel}
      eve_pos: (x_E, y_E)
      d_eve: Alice-Eve 距离
      regime: "NEAR" or "FAR"
    """
    u = np.random.rand()
    if u < cfg.near_prob:
        d_min, d_max = cfg.near_range
        regime = "NEAR"
    else:
        d_min, d_max = cfg.far_range
        regime = "FAR"

    d_eve = float(np.random.uniform(d_min, d_max))
    theta = float(np.random.uniform(0.0, 2.0 * math.pi))

    ax, ay = float(alice_pos[0]), float(alice_pos[1])
    ex = ax + d_eve * math.cos(theta)
    ey = ay + d_eve * math.sin(theta)
    eve_pos = (ex, ey)

    # 从 legit_gen 里拿信道参数（拿不到就默认）
    K0 = getattr(legit_gen, "K0", 1.0)
    d0 = getattr(legit_gen, "d0", 1.0)
    alpha = getattr(legit_gen, "alpha", 3.5)
    shadow_std_dB = getattr(legit_gen, "shadow_std_dB", 4.0)
    Gt_dB = getattr(legit_gen, "Gt_dB", 0.0)
    Gr_dB = getattr(legit_gen, "Gr_dB", 0.0)
    mimo_tx = getattr(legit_gen, "mimo_tx", 1)
    mimo_rx = getattr(legit_gen, "mimo_rx", 1)
    normalize_xi = getattr(legit_gen, "normalize_xi", True)

    eve_models: Dict[int, SemanticChannelModel] = {}
    for k, p in k_to_params.items():
        eve_models[k] = SemanticChannelModel(
            A1=p["A1"], A2=p["A2"], C1=p["C1"], C2=p["C2"],
            K0=K0, d0=d0, alpha=alpha, shadow_std_dB=shadow_std_dB,
            Gt_dB=Gt_dB, Gr_dB=Gr_dB, mimo_tx=mimo_tx, mimo_rx=mimo_rx,
            normalize_xi=normalize_xi,
        )

    return eve_models, eve_pos, d_eve, regime


# =========================
# 3) 构造单次评估 env
# =========================

def build_env_once_by_difficulty(
    level: int,
    legit_gen: LegitLinkGenerator,
) -> Tuple[AlicePLSEnv, Dict]:
    """
    每次调用 -> 新场景：
      - Bob 链路：用 legit_gen.build_bob_semantic_models()
      - Eve 链路：按 difficulty 生成距离分布
    """
    cfg = difficulty_to_config(level)

    k_to_params = legit_gen.k_to_params
    bob_models, alice_pos, bob_pos, d_bob = legit_gen.build_bob_semantic_models()

    eve_models, eve_pos, d_eve, regime = build_random_eve_models(
        alice_pos=alice_pos,
        k_to_params=k_to_params,
        legit_gen=legit_gen,
        cfg=cfg,
    )

    env = AlicePLSEnv(
        bob_models=bob_models,
        eve_models=eve_models,
        k_values=K_VALUES,
        P_dB_list=P_DB_LIST,
        distance_bob=float(d_bob),
        distance_eve=float(d_eve),
        sigma2_bob=SIGMA2_BOB,
        sigma2_eve=SIGMA2_EVE,
        n_mc=N_MC,
        lambda_e=LAMBDA_E,
        bob_xi_min=BOB_XI_MIN,
        d_norm=D_NORM,
    )

    meta = {
        "level": int(level),
        "alice_pos": alice_pos,
        "bob_pos": bob_pos,
        "eve_pos": eve_pos,
        "d_bob": float(d_bob),
        "d_eve": float(d_eve),
        "regime": regime,
        "cfg": {
            "near_range": cfg.near_range,
            "far_range": cfg.far_range,
            "near_prob": cfg.near_prob,
        }
    }
    return env, meta


# =========================
# 4) Baselines：random / heuristic_threshold
# =========================

def eval_random_policy(env: AlicePLSEnv, rand_baseline: RandomBaseline) -> Dict:
    P_dB, k = rand_baseline.select_action(env)
    xi_B, xi_E = eval_xi_pair_from_env(env, P_dB=float(P_dB), k=int(k))
    reward = xi_B - float(env.lambda_e) * xi_E
    return {
        "reward": float(reward),
        "xi_B": float(xi_B),
        "xi_E": float(xi_E),
        "diff": float(xi_B - xi_E),
        "ratio": float(xi_B / (xi_E + 1e-12)),
        "P_dB": float(P_dB),
        "k": int(k),
    }


def eval_threshold_heuristic(env: AlicePLSEnv, heur: ThresholdHeuristicBaseline) -> Dict:
    P_dB, k = heur.select_action(env)
    xi_B, xi_E = eval_xi_pair_from_env(env, P_dB=float(P_dB), k=int(k))
    reward = xi_B - float(env.lambda_e) * xi_E
    return {
        "reward": float(reward),
        "xi_B": float(xi_B),
        "xi_E": float(xi_E),
        "diff": float(xi_B - xi_E),
        "ratio": float(xi_B / (xi_E + 1e-12)),
        "P_dB": float(P_dB),
        "k": int(k),
    }


# =========================
# 5) PPO：加载并做确定性评估（argmax）
# =========================

def load_ppo_agent(policy_path: str, env: AlicePLSEnv) -> PPOAgent:
    agent = PPOAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        lr=3e-4,
        gamma=0.0,     # eval 用不到
        K_epochs=4,    # eval 用不到
        eps_clip=0.1,  # eval 用不到
    )
    ckpt = torch.load(policy_path, map_location=agent.device)

    # 兼容两种保存方式：
    # 1) 直接 state_dict
    # 2) {"policy_state_dict": ..., ...}
    if isinstance(ckpt, dict) and "policy_state_dict" in ckpt:
        agent.policy.load_state_dict(ckpt["policy_state_dict"])
    else:
        agent.policy.load_state_dict(ckpt)

    agent.policy.eval()
    return agent


@torch.no_grad()
def eval_ppo_deterministic(agent: PPOAgent, env: AlicePLSEnv) -> Dict:
    state = env.reset()
    st = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
    logits = agent.policy.actor(st)  # [1, action_dim]
    action = int(torch.argmax(logits, dim=-1).item())
    _, reward, _, info = env.step(action)
    return {
        "reward": float(reward),
        "xi_B": float(info["xi_B"]),
        "xi_E": float(info["xi_E"]),
        "diff": float(info["diff"]),
        "ratio": float(info["ratio"]),
        "P_dB": float(info["P_dB"]),
        "k": int(info["k"]),
        "action": int(action),
    }


# =========================
# 6) 统计 + 画图
# =========================

def mean_std(xs: List[float]) -> Tuple[float, float]:
    if not xs:
        return 0.0, 0.0
    arr = np.asarray(xs, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=1) if len(arr) > 1 else 0.0)


def plot_bar_for_level(level: int, summary: Dict[str, Dict[str, float]], out_png: str):
    """
    summary 形如：
      {
        "random": {"mean_reward":..., "std_reward":...},
        "heuristic_threshold": {...},
        "ppo_llm": {...},
        ...
      }
    """
    methods = ["random", "heuristic_threshold", "ppo_nollm", "ppo_llm", "ppo_llm_teacher"]
    labels = ["Random", "Heuristic", "PPO(NoLLM)", "PPO(LLM)", "PPO(LLM+Teacher)"]

    means = [summary[m]["mean_reward"] for m in methods]
    stds = [summary[m]["std_reward"] for m in methods]

    x = np.arange(len(methods))

    plt.figure()
    plt.bar(x, means, capsize=4)
    plt.xticks(x, labels, rotation=15)
    plt.ylabel("Average reward (xi_B - lambda * xi_E)")
    # plt.title(f"Cross-scene evaluation (Difficulty Level {level})")
    plt.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main1():
    os.makedirs(100 * " ", exist_ok=True) if False else None  # no-op
    os.makedirs(OUT_DIR, exist_ok=True)

    legit_gen = LegitLinkGenerator(K_VALUES)

    # 临时 env 用来初始化 PPO agent（state_dim/action_dim 依赖 env）
    tmp_env, _ = build_env_once_by_difficulty(level=3, legit_gen=legit_gen)

    # 加载 3 个 PPO
    ppo_llm = load_ppo_agent(PPO_PATH_LLM, tmp_env)
    ppo_nollm = load_ppo_agent(PPO_PATH_NOLLM, tmp_env)
    ppo_llm_teacher = load_ppo_agent(PPO_PATH_LLM_TEACHER, tmp_env)

    # baselines
    # rand_baseline = RandomBaseline()
    # heur_baseline = ThresholdHeuristicBaseline(neighbor_steps=1, max_eval_per_k=2)

    rand_baseline = RandomBaseline()
    heur_baseline = WeakHeuristicBaseline(prob_guided=0.2, seed=0)
    
    results = {
        "config": {
            "N_TRIALS_PER_LEVEL": N_TRIALS_PER_LEVEL,
            "K_VALUES": K_VALUES,
            "P_DB_LIST": P_DB_LIST,
            "SIGMA2_BOB": SIGMA2_BOB,
            "SIGMA2_EVE": SIGMA2_EVE,
            "N_MC": N_MC,
            "LAMBDA_E": LAMBDA_E,
            "BOB_XI_MIN": BOB_XI_MIN,
            "D_NORM": D_NORM,
            "ppo_paths": {
                "PPO(LLM)": PPO_PATH_LLM,
                "PPO(NoLLM)": PPO_PATH_NOLLM,
                "PPO(LLM+Teacher)": PPO_PATH_LLM_TEACHER,
            },
            "heuristic_params": {
                "neighbor_steps": getattr(heur_baseline, "neighbor_steps", None),
                "max_eval_per_k": getattr(heur_baseline, "max_eval_per_k", None),
            }
        },
        "levels": {}
    }

    for level in [1, 2, 3, 4, 5]:
        random_rewards = []
        heur_rewards = []
        ppo_llm_rewards = []
        ppo_nollm_rewards = []
        ppo_teacher_rewards = []

        level_trials = []

        for t in range(N_TRIALS_PER_LEVEL):
            env, meta = build_env_once_by_difficulty(level=level, legit_gen=legit_gen)


            

            rec_random = eval_random_policy(env, rand_baseline)
            rec_heur = eval_threshold_heuristic(env, heur_baseline)
            rec_ppo_llm = eval_ppo_deterministic(ppo_llm, env)
            rec_ppo_nollm = eval_ppo_deterministic(ppo_nollm, env)
            rec_ppo_teacher = eval_ppo_deterministic(ppo_llm_teacher, env)

            random_rewards.append(rec_random["reward"])
            heur_rewards.append(rec_heur["reward"])
            ppo_llm_rewards.append(rec_ppo_llm["reward"])
            ppo_nollm_rewards.append(rec_ppo_nollm["reward"])
            ppo_teacher_rewards.append(rec_ppo_teacher["reward"])

            level_trials.append({
                "trial": t,
                "meta": meta,
                "random": rec_random,
                "heuristic_threshold": rec_heur,
                "ppo_llm": rec_ppo_llm,
                "ppo_nollm": rec_ppo_nollm,
                "ppo_llm_teacher": rec_ppo_teacher,
            })

        m_r, s_r = mean_std(random_rewards)
        m_h, s_h = mean_std(heur_rewards)
        m_pl, s_pl = mean_std(ppo_llm_rewards)
        m_pn, s_pn = mean_std(ppo_nollm_rewards)
        m_pt, s_pt = mean_std(ppo_teacher_rewards)

        level_summary = {
            "random": {"mean_reward": m_r, "std_reward": s_r},
            "heuristic_threshold": {"mean_reward": m_h, "std_reward": s_h},
            "ppo_llm": {"mean_reward": m_pl, "std_reward": s_pl},
            "ppo_nollm": {"mean_reward": m_pn, "std_reward": s_pn},
            "ppo_llm_teacher": {"mean_reward": m_pt, "std_reward": s_pt},
        }

        results["levels"][str(level)] = {
            "difficulty_config": difficulty_to_config(level).__dict__,
            "summary": level_summary,
            "trials": level_trials,
        }

        print(f"\n[Level {level}] N={N_TRIALS_PER_LEVEL}")
        print(f"  random               mean={m_r:.4f} std={s_r:.4f}")
        print(f"  heuristic  mean={m_h:.4f} std={s_h:.4f}")
        print(f"  PPO(NoLLM)           mean={m_pn:.4f} std={s_pn:.4f}")
        print(f"  PPO(LLM)             mean={m_pl:.4f} std={s_pl:.4f}")
        print(f"  PPO(LLM+Teach)       mean={m_pt:.4f} std={s_pt:.4f}")

        # # 画并保存柱状图（每个 level 一张）
        # out_png = f"{OUT_PNG_PREFIX}_{level}.png"
        # plot_bar_for_level(level, level_summary, out_png)
        # print(f"Saved bar chart -> {out_png}")

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved evaluation json -> {OUT_JSON}")
    
    out_png = os.path.join(OUT_DIR, "bar_all_levels.png")
    plot_bar_all_levels(results["levels"], out_png)
    print(f"Saved combined bar chart -> {out_png}")

def main2():
    with open(OUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    out_png = os.path.join(OUT_DIR, "bar_all_levels.png")
    plot_bar_all_levels(data["levels"], out_png)
    print(f"Saved combined bar chart -> {out_png}")

import numpy as np
import matplotlib.pyplot as plt
import json
import os
from typing import Dict


# =========================
# 全局可调参数（只改这里）
# =========================

# 字体
FONT_FAMILY = "Times New Roman"

# 字号
FONT_SIZE_LABEL = 14
FONT_SIZE_TITLE = 15
FONT_SIZE_TICK = 12
FONT_SIZE_LEGEND = 12

# 颜色（可用 RGB 或 16 进制）
BAR_COLORS = [
    "#BBCFC3",  # Random
    "#C9DDE3",  # Greedy
    "#E18182",  # PPO(NoLLM)
    "#F9CE9C",  # PPO(LLM)
    "#83B5B5",  # PPO(LLM+Teacher)
]

BAR_COLORS1 = [
    "#4C72B0",  # Random
    "#55A868",  # Heuristic
    "#C44E52",  # PPO(NoLLM)
    "#8172B2",  # PPO(LLM)
    "#CCB974",  # PPO(LLM+Teacher)
]

import matplotlib.font_manager as fm
fm.fontManager.addfont('/models/Qwen/lib_jyz/lib_jyz/times.ttf')

def plot_bar_all_levels(level_summaries: Dict[str, Dict], out_png: str):
    """
    level_summaries:
      {
        "1": { "random": {...}, "heuristic_threshold": {...}, ... },
        "2": { ... },
        ...
        "5": { ... }
      }
    """

    # === 固定结构（不改） ===
    levels = [1, 2, 3, 4, 5]
    methods = [
        ("random", "Random"),
        ("heuristic_threshold", "Heuristic"),
        ("ppo_nollm", "PPO(NoLLM)"),
        ("ppo_llm", "PPO(LLM)"),
        ("ppo_llm_teacher", "PPO(LLM+Teacher)"),
    ]

    n_levels = len(levels)
    n_methods = len(methods)

    x = np.arange(n_levels)
    width = 0.16  # 每个柱子的宽度

    # === 字体统一 ===
    plt.rcParams["font.family"] = FONT_FAMILY

    plt.figure(figsize=(10, 5))

    for i, ((key, label), color) in enumerate(zip(methods, BAR_COLORS)):
        means = [
            level_summaries[str(l)]["summary"][key]["mean_reward"]
            for l in levels
        ]

        plt.bar(
            x + (i - (n_methods - 1) / 2) * width,
            means,
            width=width,
            label=label,
            color=color,
        )

    plt.xticks(x, levels, fontsize=FONT_SIZE_TICK)
    plt.yticks(fontsize=FONT_SIZE_TICK)

    plt.xlabel("Difficulty Level", fontsize=FONT_SIZE_LABEL)
    plt.ylabel("Average Reward", fontsize=FONT_SIZE_LABEL)

    plt.legend(ncol=3, fontsize=FONT_SIZE_LEGEND)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    with open(OUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    out_png = os.path.join(OUT_DIR, "bar_all_levels.png")
    plot_bar_all_levels(data["levels"], out_png)
    print(f"Saved combined bar chart -> {out_png}")


if __name__ == "__main__":
    main()
