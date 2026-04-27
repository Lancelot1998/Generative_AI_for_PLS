# utils/train_ppo_multi_env_with_scenes.py
# -*- coding: utf-8 -*-

import os
import json
from typing import Callable

from ppo_pls import PPOAgent


def train_ppo_multi_env_with_scenes(
    agent: PPOAgent,
    make_env: Callable,
    num_episodes: int = 100000,
    refresh_every: int = 1,
    update_interval: int = 32,
    results_dir: str = "./results/ppo_multi",
    run_name: str = "run_multi_with_scenes",
):
    """
    【增强版】多环境 PPO 训练：
    - 不修改 PPO 学习逻辑
    - 额外记录每个 env_id 对应的 scene_info
    - 在 log.json 中新增字段:
        "scenes": { "0": {...}, "1": {...}, ... }

    前提：
    - make_env() 返回的 env 必须包含 env.scene_info 字段
    """

    os.makedirs(results_dir, exist_ok=True)
    log_json_path = os.path.join(results_dir, f"{run_name}_log.json")
    summary_txt_path = os.path.join(results_dir, f"{run_name}_summary.txt")

    # ===== 原 PPO 需要的日志 =====
    episode_rewards = []
    episode_diff = []
    episode_ratio = []
    episode_xi_B = []
    episode_xi_E = []
    episode_env_id = []

    best_reward = -1e9
    best_info = None

    # ===== 新增：env_id -> scene_info =====
    scenes = {}

    env = None
    current_env_id = -1

    for ep in range(num_episodes):
        # === 1. 按 refresh_every 规则刷新环境 ===
        if ep == 0 or (refresh_every > 0 and ep % refresh_every == 0):
            env = make_env()
            current_env_id += 1

            # ★ 关键：记录场景信息
            if hasattr(env, "scene_info"):
                scenes[str(current_env_id)] = env.scene_info
            else:
                scenes[str(current_env_id)] = {}

            state = env.reset()
        else:
            state = env.reset()

        # === 2. PPO 单步交互 ===
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)

        agent.buffer.rewards.append(reward)
        agent.buffer.dones.append(float(done))

        # === 3. 记录 episode 级日志 ===
        episode_rewards.append(reward)
        episode_diff.append(info.get("diff", None))
        episode_ratio.append(info.get("ratio", None))
        episode_xi_B.append(info.get("xi_B", None))
        episode_xi_E.append(info.get("xi_E", None))
        episode_env_id.append(current_env_id)

        if reward > best_reward:
            best_reward = reward
            best_info = info.copy()
            best_info["env_id"] = current_env_id

        # === 4. PPO update ===
        if (ep + 1) % update_interval == 0:
            agent.update()

        if (ep + 1) % 500 == 0:
            print(
                f"[Episode {ep+1}] env_id={current_env_id}, "
                f"reward={reward:.4f}, "
                f"xi_B={info.get('xi_B', 0):.4f}, "
                f"xi_E={info.get('xi_E', 0):.4f}, "
                f"P_dB={info.get('P_dB', -1)}, k={info.get('k', -1)}"
            )

    # ===== 5. 写 log.json（新增 scenes）=====
    log_data = {
        "episode_rewards": episode_rewards,
        "episode_diff": episode_diff,
        "episode_ratio": episode_ratio,
        "episode_xi_B": episode_xi_B,
        "episode_xi_E": episode_xi_E,
        "episode_env_id": episode_env_id,
        "best_strategy": best_info,
        "scenes": scenes,     # ★ 核心新增字段
    }

    with open(log_json_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)

    # ===== 6. summary.txt =====
    with open(summary_txt_path, "w", encoding="utf-8") as f:
        f.write("=== PPO Multi-Env Training Summary ===\n")
        f.write(f"Num episodes: {num_episodes}\n")
        f.write(f"Refresh every: {refresh_every}\n")
        f.write(f"Total envs: {current_env_id + 1}\n")
        f.write(f"Best reward: {best_reward:.6f}\n")
        if best_info is not None:
            f.write(f"Best env_id: {best_info['env_id']}\n")
            f.write(f"Best P_dB: {best_info.get('P_dB')}\n")
            f.write(f"Best k: {best_info.get('k')}\n")
            f.write(f"Best xi_B: {best_info.get('xi_B')}\n")
            f.write(f"Best xi_E: {best_info.get('xi_E')}\n")

    print("\n[PPO] Training finished.")
    print("Log saved to:", log_json_path)
    print("Summary saved to:", summary_txt_path)
