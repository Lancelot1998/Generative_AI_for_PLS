# semantic_channel.py
# -*- coding: utf-8 -*-
"""
基于 DeepSC 拟合参数 (A1, A2, C1, C2) 的“快算语义信道模型”。

使用方法示例：
    from semantic_channel import SemanticChannelModel

    # 1. 填入你拟合好的参数
    scm = SemanticChannelModel(
        A1=..., A2=..., C1=..., C2=...,
        K0=1.0, d0=1.0, alpha=3.5, shadow_std_dB=4.0,
        Gt_dB=0.0, Gr_dB=0.0,      # 天线增益（dB，可选）
        mimo_tx=1, mimo_rx=1,      # MIMO 阵列规模（可选）
        normalize_xi=False         # 是否将 xi 线性归一化到 [0,1]
    )

    # 2. 给定发射功率 P、几何距离 d、接收噪声功率 sigma2，直接算平均语义相似度
    xi_mean = scm.semantic_similarity(P=1.0, distance=50.0, sigma2=0.1, n_mc=100)

    # 或者同时要均值+标准差：
    xi_mean, xi_std = scm.semantic_similarity(
        P=1.0, distance=50.0, sigma2=0.1, n_mc=300, return_std=True
    )
"""

import numpy as np
from typing import Optional, Tuple, Union


ArrayLike = Union[np.ndarray, float]


class SemanticChannelModel:
    """
    语义信道快算模型：

    - 通过 DeepSC + 仿真得到的一组 logistic 拟合参数 (A1, A2, C1, C2)，
      建立 xi(gamma) 与 SNR 的关系；
    - 使用几何 + 阴影 + Rayleigh 小尺度模型计算瞬时 SNR gamma；
    - 对给定 (P, d, sigma2) 返回平均语义相似度 xi。

    信道数学模型（与 pdf 一致）：
        h = G(d, χ) * g
        |G|^2 = K0 * (d/d0)^(-alpha) * 10^(χ / 10)
        g ~ CN(0,1),  => |g|^2 ~ Exp(1)
        gamma = P * |h|^2 / sigma2

    本实现在此基础上**扩展**：
        |h|^2 额外乘以：Gt_lin * Gr_lin * MIMO_gain
        其中：
            Gt_lin = 10^(Gt_dB/10)
            Gr_lin = 10^(Gr_dB/10)
            MIMO_gain = mimo_tx * mimo_rx  （简化阵列增益模型）
    """

    def __init__(self,
                 A1: float,
                 A2: float,
                 C1: float,
                 C2: float,
                 K0: float = 1.0,
                 d0: float = 1.0,
                 alpha: float = 3.5,
                 shadow_std_dB: float = 4.0,
                 # === 新增：天线增益 & MIMO 参数 ===
                 Gt_dB: float = 0.0,
                 Gr_dB: float = 0.0,
                 mimo_tx: int = 1,
                 mimo_rx: int = 1,
                 # === 新增：是否把 xi 归一化到 [0,1] ===
                 normalize_xi: bool = False,
                 rng: Optional[np.random.Generator] = None):
        """
        参数：
            A1, A2, C1, C2 : 你拟合得到的 DeepSC 语义曲线参数
            K0, d0, alpha  : 大尺度路径损耗参数
            shadow_std_dB  : 阴影衰落标准差（单位 dB）
            Gt_dB, Gr_dB   : 发射 / 接收天线增益（单位 dB），默认为 0 dB
            mimo_tx, mimo_rx : 发射 / 接收天线数，简化为阵列增益 M_tx*M_rx
            normalize_xi   : 若为 True，则将 xi 做线性归一化到 [0,1]
                             公式：xi_norm = clip((xi - A1)/(A2 - A1), 0, 1)
            rng            : 可选的 numpy 随机数生成器
        """
        # 拟合参数
        self.A1 = float(A1)
        self.A2 = float(A2)
        self.C1 = float(C1)
        self.C2 = float(C2)

        # 大尺度信道参数
        self.K0 = float(K0)
        self.d0 = float(d0)
        self.alpha = float(alpha)
        self.shadow_std_dB = float(shadow_std_dB)

        # 天线 & MIMO 参数
        self.Gt_dB = float(Gt_dB)
        self.Gr_dB = float(Gr_dB)
        self.mimo_tx = int(mimo_tx)
        self.mimo_rx = int(mimo_rx)

        # dB → 线性
        self.Gt_lin = 10.0 ** (self.Gt_dB / 10.0)
        self.Gr_lin = 10.0 ** (self.Gr_dB / 10.0)
        # 简化阵列增益：MIMO_gain = M_tx * M_rx
        self.MIMO_gain = max(1, self.mimo_tx) * max(1, self.mimo_rx)

        # xi 是否做线性归一化
        self.normalize_xi = bool(normalize_xi)

        # 随机数生成器
        self.rng = rng if rng is not None else np.random.default_rng()

    # ---------- 1. logistic 语义曲线 xi(gamma) ----------

    def xi_from_gamma(self, gamma: ArrayLike) -> ArrayLike:
        """
        根据拟合好的参数计算 xi(gamma)。
        gamma 可以是标量或 numpy 数组（单位：线性 SNR，而不是 dB）。

        对应公式：
            xi(gamma) = A1 + (A2 - A1) / (1 + exp( - (C1 * log(gamma) + C2) ))

        若 normalize_xi=True，则进一步做线性归一化：
            xi_norm = clip( (xi - A1) / (A2 - A1), 0, 1 )
        """
        gamma_arr = np.asarray(gamma, dtype=float)
        # 防止 log(0)
        gamma_arr = np.clip(gamma_arr, 1e-12, None)

        xi_arr = self.A1 + (self.A2 - self.A1) / (
            1.0 + np.exp(-(self.C1 * np.log(gamma_arr) + self.C2))
        )

        if self.normalize_xi:
            # 线性归一化到 [0,1]，基于拟合到的 A1/A2
            denom = max(self.A2 - self.A1, 1e-8)
            xi_arr = (xi_arr - self.A1) / denom
            xi_arr = np.clip(xi_arr, 0.0, 1.0)

        # 如果输入是标量，就返回标量
        if np.isscalar(gamma):
            return float(xi_arr)
        return xi_arr

    # ---------- 2. 几何 + 阴影 + Rayleigh 信道：采样 gamma ----------

    def _sample_large_scale_gain(self, distance: float, n_samples: int) -> np.ndarray:
        """
        采样 n_samples 个大尺度功率增益 |G|^2。
        distance: 几何距离 d（米）
        返回形状 (n_samples,) 的数组：|G|^2
        """
        distance = float(distance)

        # 阴影 χ ~ N(0, shadow_std_dB^2) (单位 dB)
        chi = self.rng.normal(
            loc=0.0,
            scale=self.shadow_std_dB,
            size=n_samples
        )  # dB

        # 路径损耗 (d/d0)^(-alpha)
        pl = (distance / self.d0) ** (-self.alpha)  # 标量

        # |G|^2 = K0 * pl * 10^(χ/10)
        G_abs2 = self.K0 * pl * (10.0 ** (chi / 10.0))  # shape (n_samples,)
        return G_abs2

    def sample_gamma(self,
                     P: float,
                     distance: float,
                     sigma2: float,
                     n_samples: int = 1) -> np.ndarray:
        """
        在给定发射功率 P、几何距离 distance、噪声功率 sigma2 下，
        采样 n_samples 个瞬时 SNR gamma。

        基本模型：
            gamma = P * |G|^2 * |g|^2 / sigma2

        本实现加入天线增益与 MIMO 阵列增益：
            gamma = P * |G|^2 * |g|^2 * Gt_lin * Gr_lin * MIMO_gain / sigma2
        """
        P = float(P)
        sigma2 = float(sigma2)

        # 1) 大尺度增益 |G|^2
        G_abs2 = self._sample_large_scale_gain(distance, n_samples=n_samples)

        # 2) 小尺度 Rayleigh: |g|^2 ~ Exp(1)
        g_abs2 = self.rng.exponential(scale=1.0, size=n_samples)

        # 3) 总增益因子（天线 + MIMO）
        H_gain = self.Gt_lin * self.Gr_lin * self.MIMO_gain

        # 4) SNR
        gamma = P * G_abs2 * g_abs2 * H_gain / sigma2  # shape (n_samples,)
        return gamma

    # ---------- 3. 一步到位：给 (P, d, sigma2) 算平均语义相似度 ----------

    def semantic_similarity(self,
                            P: float,
                            distance: float,
                            sigma2: float,
                            n_mc: int = 100,
                            return_std: bool = False
                            ) -> Union[float, Tuple[float, float]]:
        """
        主接口：给定发射功率、几何距离、噪声功率，输出平均语义相似度 E[ xi(gamma) ]。

        参数：
            P        : 发射功率（线性值，不是 dB）
            distance : 几何距离 d（米）
            sigma2   : 噪声功率（线性值）
            n_mc     : Monte Carlo 采样次数（越大越稳定，越小越快）
            return_std : 若为 True，则同时返回 (均值, 标准差)

        返回：
            若 return_std=False：
                xi_mean : 平均语义相似度（标量）
            若 return_std=True：
                (xi_mean, xi_std)
        """
        # 1. 多次采样 gamma
        gammas = self.sample_gamma(
            P=P,
            distance=distance,
            sigma2=sigma2,
            n_samples=n_mc
        )

        # 2. 对每个 gamma 计算 xi(gamma)
        xis = self.xi_from_gamma(gammas)

        # 3. 统计
        xi_mean = float(np.mean(xis))
        if not return_std:
            return xi_mean

        xi_std = float(np.std(xis))
        return xi_mean, xi_std
