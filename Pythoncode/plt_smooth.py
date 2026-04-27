import os
import json
import numpy as np
import matplotlib.pyplot as plt


def load_rewards(json_path):
    """从 json 文件中读取 episode_rewards 列表"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return np.array(data["episode_rewards"], dtype=float)


def moving_average(x, window_size=200):
    """
    简单滑动平均，用于平滑 reward 曲线
    window_size 可以根据你的需要调整
    """
    if window_size <= 1:
        return x
    # 用卷积实现滑动平均
    kernel = np.ones(window_size) / window_size
    # mode='valid' 会丢掉前面 window_size-1 个点，
    # 为了对齐长度，我们前面补上对应数量的 NaN
    ma = np.convolve(x, kernel, mode="valid")
    pad = np.full(window_size - 1, np.nan)
    return np.concatenate([pad, ma])


def plot_rewards_with_smoothing(
    json_path_1,
    json_path_2,
    label_1="Scene 1",
    label_2="Refresh Every 5000",
    window_size=200,
    save_dir="./figures",
    save_name="episode_rewards_comparison.png",
):
    # 1. 读取数据
    rewards_1 = load_rewards(json_path_1)
    rewards_2 = load_rewards(json_path_2)

    episodes_1 = np.arange(1, len(rewards_1) + 1)
    episodes_2 = np.arange(1, len(rewards_2) + 1)

    # 2. 计算平滑曲线
    smooth_1 = moving_average(rewards_1, window_size=window_size)
    smooth_2 = moving_average(rewards_2, window_size=window_size)

    # 3. 画图
    plt.figure(figsize=(16, 6))

    # 原始数据：使用较淡的颜色 + 透明度 alpha=0.3
    plt.plot(
        episodes_1,
        rewards_1,
        linewidth=0.5,
        alpha=0.3,
        label=f"{label_1} (raw)",
    )
    plt.plot(
        episodes_2,
        rewards_2,
        linewidth=0.5,
        alpha=0.3,
        label=f"{label_2} (raw)",
    )

    # 也可以改成 fill_between 的“阴影”效果（可选）：
    # plt.fill_between(episodes_1, rewards_1, alpha=0.2, step="mid", label=f"{label_1} (raw)")
    # plt.fill_between(episodes_2, rewards_2, alpha=0.2, step="mid", label=f"{label_2} (raw)")

    # 平滑后的曲线：线条更粗，alpha=1.0
    plt.plot(
        episodes_1,
        smooth_1,
        linewidth=2.0,
        label=f"{label_1} (smoothed)",
    )
    plt.plot(
        episodes_2,
        smooth_2,
        linewidth=2.0,
        label=f"{label_2} (smoothed)",
    )

    plt.title("Episode Rewards Comparison (Smoothed + Raw)", fontsize=16)
    plt.xlabel("Episode", fontsize=14)
    plt.ylabel("Reward", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right", fontsize=12)

    # 4. 保存图片到指定文件夹
    os.makedirs(save_dir, exist_ok=True)  # 如果文件夹不存在就创建
    save_path = os.path.join(save_dir, save_name)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Figure saved to: {save_path}")

    # 如果只想保存，不想弹出窗口，可以注释掉下面这行
    plt.show()


if __name__ == "__main__":
    # 修改成你自己的 json 路径
    json1 = "/models/Qwen/lyq_data/pythoncode/DeepSCmaster/results/ppo_random/Random_refresh_every_5000_gen_log.json"
    json2 = "/models/Qwen/lyq_data/pythoncode/DeepSCmaster/results/ppo_Qwen/Qwen_refresh_every_5000_log.json"

    plot_rewards_with_smoothing(
        json1,
        json2,
        label_1="Random Refresh Every 5000",
        label_2="Qwen2.5 Refresh Every 5000",
        window_size=300,          # 可以调大/调小看效果
        save_dir="/models/Qwen/lyq_data/pythoncode/DeepSCmaster/figure",     # 你想保存到的文件夹
        save_name="rewards_smooth_comparison.png",
    )
