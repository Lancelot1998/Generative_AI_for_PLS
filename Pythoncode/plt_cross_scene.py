# plt_cross_scene.py
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple


# 你日志如果是 list 形式，默认按这个顺序解释
# [episode, env_id, reward, xi_B, xi_E, P_dB, k]
POS_KEYS = ["episode", "env_id", "reward", "xi_B", "xi_E", "P_dB", "k"]


def _entry_list_to_dict(row: list) -> Dict[str, Any] | None:
    """把 [episode, env_id, reward, ...] 这种 list 记录转换成 dict。"""
    if not isinstance(row, list) or len(row) < 3:
        return None
    d = {}
    for i, key in enumerate(POS_KEYS):
        if i >= len(row):
            break
        d[key] = row[i]
    return d


def _normalize_entries(obj: Any) -> List[Dict[str, Any]]:
    """
    把各种可能格式的日志 obj 统一变成 List[dict]。
    支持：
      - list[dict]
      - dict（episode->dict）
      - list[list] （位置记录）
      - list[list[dict]]（嵌套）
      - list[dict/list混合]
    """
    out: List[Dict[str, Any]] = []

    if obj is None:
        return out

    # case 1: dict -> values
    if isinstance(obj, dict):
        # 常见：{ "0": {...}, "1": {...} }
        vals = list(obj.values())
        return _normalize_entries(vals)

    # case 2: list -> 遍历
    if isinstance(obj, list):
        for item in obj:
            # item 是 dict
            if isinstance(item, dict):
                out.append(item)
                continue

            # item 是 list：可能是 [episode, env_id, ...] 或 [dict] 或 [dict, dict]
            if isinstance(item, list):
                # [dict] 或 [dict, dict, ...]
                if len(item) > 0 and all(isinstance(x, dict) for x in item):
                    out.extend(item)
                    continue
                # [[...], [...]] 再递归
                if len(item) > 0 and all(isinstance(x, list) for x in item):
                    out.extend(_normalize_entries(item))
                    continue
                # 位置记录
                d = _entry_list_to_dict(item)
                if d is not None:
                    out.append(d)
                continue

            # 其他类型忽略
        return out

    # 其他顶层类型不支持
    return out


def load_log(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    entries = _normalize_entries(raw)
    if len(entries) == 0:
        raise ValueError(f"No valid entries parsed from {path}. "
                         f"Please print the raw json head to inspect.")
    return entries


def per_env_mean(
    entries: List[Dict[str, Any]],
    key: str = "reward",
    last_n: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    groups: Dict[int, List[float]] = {}

    for d in entries:
        # env_id 可能是 float/str，统一转 int
        env_id = int(d.get("env_id", 0))
        val = float(d.get(key, 0.0))
        groups.setdefault(env_id, []).append(val)

    env_ids = sorted(groups.keys())
    means = []
    for eid in env_ids:
        arr = np.asarray(groups[eid], dtype=float)
        if last_n is not None and last_n > 0:
            arr = arr[-last_n:]
        means.append(float(np.mean(arr)))
    return np.asarray(env_ids, dtype=int), np.asarray(means, dtype=float)


def summarize_method(
    path: str,
    key: str = "reward",
    last_n: int | None = None,
) -> Dict[str, Any]:
    entries = load_log(path)
    env_ids, env_means = per_env_mean(entries, key=key, last_n=last_n)
    return {
        "path": path,
        "n_entries": len(entries),
        "n_envs": len(env_ids),
        "env_ids": env_ids,
        "env_means": env_means,
        "mean": float(np.mean(env_means)) if len(env_means) else float("nan"),
        "std": float(np.std(env_means)) if len(env_means) else float("nan"),
    }


def plot_bar(summary: Dict[str, Dict[str, Any]], title: str, save_path: str | None = None):
    names = list(summary.keys())
    means = [summary[n]["mean"] for n in names]
    stds = [summary[n]["std"] for n in names]

    plt.figure()
    plt.bar(range(len(names)), means, yerr=stds, capsize=6)
    plt.xticks(range(len(names)), names, rotation=20)
    plt.ylabel("Cross-env mean reward")
    plt.title(title)
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200)

    plt.show()


def main():
    # ===== 你在这里填路径 =====
    # 例如：
    # PPO (无 LLM)
    ppo_path = "/models/Qwen/lyq_data/pythoncode/DeepSCmaster/results/ppo_random/Random_refresh_every_5000_1_log.json"
    # PPO (LLM)
    llm_ppo_path = "/models/Qwen/lyq_data/pythoncode/DeepSCmaster/results/ppo_Qwen/Qwen_refresh_every_5000_log.json"

    llm_ppo_t_path = "/models/Qwen/lyq_data/pythoncode/DeepSCmaster/results/ppo_Qwen/Qwen_refresh_every_5000_log.json"
    # Baselines
    random_base_path = "/models/Qwen/lyq_data/pythoncode/DeepSCmaster/results/baselines/baseline_random_log.json"
    greedy_base_path = "/models/Qwen/lyq_data/pythoncode/DeepSCmaster/results/ppo_Qwen_teacher/Qwen_teacher_refresh_every_5000_log.json"
    # heuristic_base_path = "/models/Qwen/lyq_data/pythoncode/DeepSCmaster/results/baselines/baseline_heuristic_log.json"

    # ===== 统计配置 =====
    metric_key = "reward"   # 你说“reward=语义还原度差值”就用 reward
    last_n = None           # 如果你只想看每个 env 的最后 500 轮均值：last_n=500

    summaries = {
        "PPO(RandomEve)": summarize_method(ppo_path, key=metric_key, last_n=last_n),
        "PPO(LLM-Eve)": summarize_method(llm_ppo_path, key=metric_key, last_n=last_n),
        "PPO(LLM-Teacher)": summarize_method(llm_ppo_path, key=metric_key, last_n=last_n),
        "Baseline-Random": summarize_method(random_base_path, key=metric_key, last_n=last_n),
        "Baseline-Greedy": summarize_method(greedy_base_path, key=metric_key, last_n=last_n),
    }

    for name, s in summaries.items():
        print(f"{name:18s} | envs={s['n_envs']:3d} | mean={s['mean']:.4f} | std={s['std']:.4f} | entries={s['n_entries']}")

    plot_bar(
        summaries,
        title=f"Cross-scenario performance (metric={metric_key}, last_n={last_n})",
        save_path="./figure/cross_env_bar_reward.png"
    )

if __name__ == "__main__":
    main()
