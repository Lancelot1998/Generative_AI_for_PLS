class Args:
    def __init__(self, n):
        self.vocab_file = "/models/Qwen/lyq_data/europarl/vocab.json"
        self.checkpoint = f"/models/Qwen/lyq_data/checkpoints/deepsc-Rayleigh_{n}/checkpoint.pth"

        self.distance = 10.0  # 几何距离 d（米）
        self.sigma2 = 0.1  # 噪声功率 σ^2（线性）

        self.P_min = 0.1  # 扫描 P 的下限
        self.P_max = 2.0  # 扫描 P 的上限
        self.P_steps = 8  # P 的分段数量（用于 linspace）

        self.max_batches = 50  # 最大 batch 数
        self.batch_size = 64  # batch 大小


