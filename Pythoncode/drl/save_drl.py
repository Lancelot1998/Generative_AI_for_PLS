import torch
import os

def save_ppo_agent(
    agent,
    save_path: str,
):
    """
    保存 PPO agent（与你当前 ppo_pls.py 完全兼容）

    参数:
      agent     : PPOAgent 实例
      save_path: 保存路径，例如 "./results/ppo_Qwen/ppo_agent_final.pth"
      extra_info: 可选的额外信息（例如训练轮数、备注等）
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    checkpoint = {
        "policy_state_dict": agent.policy.state_dict(),
        "optimizer_state_dict": agent.optimizer.state_dict(),
    }

    torch.save(checkpoint, save_path)
    print(f"[PPO] Agent saved to: {save_path}")