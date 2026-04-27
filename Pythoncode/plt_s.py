import json
import matplotlib.pyplot as plt
import os

# 你的文件路径（请根据你的实际路径修改）
file1 = "/models/Qwen/lyq_data/pythoncode/DeepSCmaster/results/ppo_Qwen/Qwen_scene_1_log.json"
file2 = "/models/Qwen/lyq_data/pythoncode/DeepSCmaster/results/ppo_Qwen/Qwen_refresh_every_5000_log.json"

def load_rewards(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["episode_rewards"]

# 加载
rewards1 = load_rewards(file1)
rewards2 = load_rewards(file2)

episodes1 = list(range(len(rewards1)))
episodes2 = list(range(len(rewards2)))

# 输出文件夹
save_dir = "/models/Qwen/lyq_data/pythoncode/DeepSCmaster/figure"
os.makedirs(save_dir, exist_ok=True)  # 若文件夹不存在则创建

# ============================
# 图 1：两条奖励曲线对比
# ============================
plt.figure(figsize=(12, 6))
plt.plot(episodes1, rewards1, label="Scene 1", alpha=0.8)
plt.plot(episodes2, rewards2, label="Refresh Every 5000", alpha=0.8)

plt.title("Episode Rewards Comparison")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.legend()
plt.grid(True)
plt.tight_layout()

# 保存图像
save_path1 = os.path.join(save_dir, "comparison_reward_plot.png")
plt.savefig(save_path1, dpi=300, bbox_inches="tight")
print(f"图像 1 已保存至: {save_path1}")

plt.show()


# ============================
# 图 2：分开绘制两个曲线
# ============================

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(episodes1, rewards1, color="blue")
plt.title("Scene 1 Rewards")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(episodes2, rewards2, color="green")
plt.title("Refresh Every 5000 Rewards")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid(True)

plt.tight_layout()

# 保存第二张图
save_path2 = os.path.join(save_dir, "reward_subplots.png")
plt.savefig(save_path2, dpi=300, bbox_inches="tight")
print(f"图像 2 已保存至: {save_path2}")

# plt.show()
