import torch
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def power_normalize(x: torch.Tensor, target_power: float = 1.0) -> torch.Tensor:
    """简单的功率归一化"""
    power = (x * x).mean().sqrt()
    if power > 0:
        x = x * (math.sqrt(target_power) / power)
    return x


class GeoShadowRayleighChannel:
    """
    完整升级版的信道模型（你可随意传参控制）：

    包含：
    1. 路径损耗 (d/d0)^(-alpha)
    2. log-normal 阴影衰落（shadow_std_dB）
    3. Rayleigh 小尺度衰落（g ~ CN(0,1)）
    4. 发射功率 P
    5. 噪声 σ²
    6. 发射天线增益 Gt（dB）
    7. 接收天线增益 Gr（dB）
    8. 简化 MIMO 阵列增益（mimo_tx * mimo_rx）
    9. 可选：Tx 功率归一化

    最终信道增益：
        |h|^2 = Gt * Gr * (mimo_tx * mimo_rx) * PL(d) * Shadow * Rayleigh
    """

    def __init__(self,
                 K0: float = 1.0,
                 d0: float = 1.0,
                 alpha: float = 3.5,
                 shadow_std_dB: float = 4.0,
                 Gt_dB: float = 0.0,
                 Gr_dB: float = 0.0,
                 mimo_tx: int = 1,
                 mimo_rx: int = 1,
                 normalize_tx: bool = True):
        self.K0 = K0
        self.d0 = d0
        self.alpha = alpha
        self.shadow_std_dB = shadow_std_dB
        self.Gt_dB = Gt_dB
        self.Gr_dB = Gr_dB
        self.mimo_tx = mimo_tx
        self.mimo_rx = mimo_rx
        self.normalize_tx = normalize_tx

        # 把天线增益 dB → 线性
        self.Gt_lin = 10 ** (Gt_dB / 10)
        self.Gr_lin = 10 ** (Gr_dB / 10)

        # 简化 MIMO：阵列增益 = M*N
        self.MIMO_gain = mimo_tx * mimo_rx

    # -------------------------------
    # 大尺度：路径损耗 + 阴影
    # -------------------------------
    def _large_scale_gain(self, distance: float):
        """返回 |G| 和 |G|^2"""
        shadow = torch.normal(
            0.0, self.shadow_std_dB, size=[1]
        ).to(device)

        # 路径损耗 (d/d0)^(-alpha)
        PL = (distance / self.d0) ** (-self.alpha)

        # 阴影：10^(χ/10)
        shadow_lin = 10 ** (shadow / 10)

        # 大尺度增益
        G_abs2 = self.K0 * PL * shadow_lin
        G_abs = torch.sqrt(G_abs2)

        return G_abs, G_abs2

    # -------------------------------
    # 小尺度：Rayleigh
    # -------------------------------
    def _small_scale_rayleigh(self):
        """返回 |g| 和 |g|^2"""
        g_abs2 = torch.distributions.Exponential(rate=1.0).sample().to(device)
        return torch.sqrt(g_abs2), g_abs2

    # -------------------------------
    # 主函数：信道作用
    # -------------------------------
    def __call__(self,
                 Tx_sig: torch.Tensor,
                 P: float,
                 distance: float,
                 sigma2: float):

        x = Tx_sig
        if self.normalize_tx:
            x = power_normalize(x)

        # 1. 大尺度 |G|
        G_abs, G_abs2 = self._large_scale_gain(distance)

        # 2. 小尺度 |g|
        g_abs, g_abs2 = self._small_scale_rayleigh()

        # 3. 总信道增益 |h|^2
        #    包含天线增益 Gt * Gr、MIMO 阵列增益
        H_gain = (self.Gt_lin * self.Gr_lin * self.MIMO_gain)
        h_abs2 = H_gain * G_abs2 * g_abs2
        h_abs = torch.sqrt(h_abs2)

        # 4. y = sqrt(P) * |h| * x
        y = math.sqrt(P) * h_abs * x

        # 5. AWGN
        noise = torch.normal(
            mean=0.0,
            std=math.sqrt(sigma2),
            size=y.shape
        ).to(device)
        y_noisy = y + noise

        # 6. SNR：gamma = P |h|^2 / sigma2
        gamma = P * h_abs2 / sigma2

        return y_noisy, gamma
