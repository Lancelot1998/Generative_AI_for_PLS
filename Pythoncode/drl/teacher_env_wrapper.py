# teacher_env_wrapper.py
import numpy as np

class TeacherWrappedEnv:
    """
    一个简单的环境 wrapper：
    - 里面包着原始 AlicePLSEnv (base_env)
    - 多了一份 teacher 对 (P_dB, k) 的推荐动作集合 teacher_actions
    - 在 step 后，对 reward 进行 reward shaping：
        r' = r_env + beta * I[(P_dB, k) ∈ teacher_actions]
    """

    def __init__(self, base_env, teacher_actions, beta: float = 0.2):
        """
        参数:
            base_env: 原始的 AlicePLSEnv 实例
            teacher_actions: set((P_dB, k))，teacher 推荐的动作集合
            beta: 奖励增强系数（可调）
        """
        self.base_env = base_env
        self.teacher_actions = set(teacher_actions)
        self.beta = beta

        # 暴露和原 env 相同的接口
        self.state_dim = base_env.state_dim
        self.action_dim = base_env.action_dim

        # 方便解析 action -> (P_dB, k)
        self.P_dB_list = base_env.P_dB_list
        self.k_values = base_env.k_values
        self.num_P = len(self.P_dB_list)
        self.num_k = len(self.k_values)

    def reset(self):
        """
        直接调用底层 env 的 reset
        """
        return self.base_env.reset()

    def step(self, action: int):
        """
        1) 调用底层 env.step(action)，得到原始 (state, reward, done, info)
        2) 把离散 action index 还原成 (P_dB, k)
        3) 如果 (P_dB, k) 在 teacher 推荐集合内，额外加 beta 的奖励
        """
        state, reward, done, info = self.base_env.step(action)

        # ---- 解析 action -> (P_dB, k) ----
        # 假设 action ∈ [0, num_P * num_k)
        p_idx = action // self.num_k
        k_idx = action % self.num_k

        # 做个边界保护
        if 0 <= p_idx < self.num_P and 0 <= k_idx < self.num_k:
            P_dB = self.P_dB_list[p_idx]
            k = self.k_values[k_idx]
        else:
            # 异常情况（一般不会发生）
            P_dB = None
            k = None

        # ---- teacher reward shaping ----
        teacher_bonus = 0.0
        if (P_dB, k) in self.teacher_actions:
            teacher_bonus = self.beta
            reward = reward + teacher_bonus

        # 可以把 teacher_bonus 记录到 info 里，方便后续分析
        if isinstance(info, dict):
            info = dict(info)
            info["teacher_bonus"] = teacher_bonus
            info["P_dB"] = P_dB
            info["k"] = k

        return state, reward, done, info
