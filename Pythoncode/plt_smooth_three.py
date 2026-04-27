import os
import json
import numpy as np
import matplotlib.pyplot as plt


def load_rewards(json_path):
    """从 json 文件中读取 episode_rewards 列表"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return np.array(data["episode_rewards"], dtype=float)

def load_rewards_t(json_path):
    """从 json 文件中读取 episode_rewards 列表"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data_b = np.array(data["episode_xi_B"], dtype=float)
    data_e = np.array(data["episode_xi_E"], dtype=float)
    return data_b-data_e

def moving_average(x, window_size=200):
    """滑动平均平滑"""
    if window_size <= 1:
        return x
    kernel = np.ones(window_size) / window_size
    ma = np.convolve(x, kernel, mode="valid")
    pad = np.full(window_size - 1, np.nan)
    return np.concatenate([pad, ma])


def plot_rewards_with_smoothing_three(
    json_path_1,
    json_path_2,
    json_path_3,
    label_1="Curve 1",
    label_2="Curve 2",
    label_3="Curve 3",
    window_size=200,
    save_dir="./figures",
    save_name="episode_rewards_comparison_three.png",
):

    # ====== 颜色组（raw + smoothed 成对）======
    color_raw_1 = (0.3, 0.5, 1.0, 0.30)      # 蓝色淡 → raw
    color_smooth_1 = (0.0, 0.2, 0.8, 1.00)    # 蓝色深 → smoothed

    color_raw_2 = (1.0, 0.6, 0.2, 0.30)       # 橙色淡 → raw
    color_smooth_2 = (0.8, 0.3, 0.0, 1.00)    # 橙色深 → smoothed

    color_raw_3 = (0.4, 0.8, 0.4, 0.30)       # 绿色淡 → raw
    color_smooth_3 = (0.0, 0.6, 0.0, 1.00)    # 绿色深 → smoothed

    # ---- 读取三条数据 ----
    rewards_1 = load_rewards(json_path_1)
    rewards_2 = load_rewards(json_path_2)
    rewards_3 = load_rewards(json_path_3)

    episodes_1 = np.arange(1, len(rewards_1) + 1)
    episodes_2 = np.arange(1, len(rewards_2) + 1)
    episodes_3 = np.arange(1, len(rewards_3) + 1)

    # ---- 平滑后的曲线 ----
    smooth_1 = moving_average(rewards_1, window_size=window_size)
    smooth_2 = moving_average(rewards_2, window_size=window_size)
    smooth_3 = moving_average(rewards_3, window_size=window_size)

    # ---- 画图 ----
    plt.figure(figsize=(16, 6))

    # ---------- 原始曲线 raw ----------
    plt.plot(episodes_1, rewards_1, color=color_raw_1, linewidth=0.5, label=f"{label_1} (raw)")
    plt.plot(episodes_2, rewards_2, color=color_raw_2, linewidth=0.5, label=f"{label_2} (raw)")
    plt.plot(episodes_3, rewards_3, color=color_raw_3, linewidth=0.5, label=f"{label_3} (raw)")

    # ---------- 平滑曲线 smoothed ----------
    plt.plot(episodes_1, smooth_1, color=color_smooth_1, linewidth=2.2, label=f"{label_1} (smoothed)")
    plt.plot(episodes_2, smooth_2, color=color_smooth_2, linewidth=2.2, label=f"{label_2} (smoothed)")
    plt.plot(episodes_3, smooth_3, color=color_smooth_3, linewidth=2.2, label=f"{label_3} (smoothed)")

    # ---------- 添加水平线 y=1.2 ----------
    plt.axhline(y=1.2, color="black", linestyle="--", linewidth=1.5, label="teacher_bonus")

    plt.title("Episode Rewards Comparison (Three Curves)", fontsize=16)
    plt.xlabel("Episode", fontsize=14)
    plt.ylabel("Reward", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right", fontsize=12)

    # ---- 保存 ----
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, save_name)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Figure saved to: {save_path}")

    plt.show()


if __name__ == "__main__":
    json1 = "/models/Qwen/lyq_data/pythoncode/DeepSCmaster/results/ppo_random/Random_refresh_every_5000_gen_log.json"
    json2 = "/models/Qwen/lyq_data/pythoncode/DeepSCmaster/results/ppo_Qwen/Qwen_refresh_every_5000_log.json"
    json3 = "/models/Qwen/lyq_data/pythoncode/DeepSCmaster/results/ppo_Qwen_teacher/Qwen_teacher_refresh_every_5000_log.json"

    plot_rewards_with_smoothing_three(
        json1,
        json2,
        json3,
        label_1="Random Refresh Every 5000",
        label_2="Qwen2.5 Refresh Every 5000",
        label_3="Qwen2.5 Teacher And Refresh Every 500",
        window_size=100,
        save_dir="/models/Qwen/lyq_data/pythoncode/DeepSCmaster/figure",
        save_name="rewards_smooth_comparison_three_t.png",
    )
