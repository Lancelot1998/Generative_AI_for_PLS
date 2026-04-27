# env_legit_link.py
import numpy as np
from typing import Dict, Tuple
from channel.semantic_channelpro import SemanticChannelModel  # 改成你的真实文件名/类名

import os
import json
from typing import List, Optional

def load_k_params_from_dir(
    k_list: Optional[List[int]] = None,
    prefix: str = "params_",
    suffix: str = ".json",
) -> Dict[int, Dict[str, float]]:

    params_dir = '/models/Qwen/lyq_data/logit_p/'
    
    k_to_params: Dict[int, Dict[str, float]] = {}

    for k in k_list:
        fname = f"{prefix}{k}{suffix}"      # 例如 "params_4.json"
        fpath = os.path.join(params_dir, fname)

        if not os.path.exists(fpath):
            raise FileNotFoundError(f"未找到 k={k} 对应的参数文件: {fpath}")

        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 这里假设 json 有 A1,A2,C1,C2 四个键
        A1 = float(data["A1"])
        A2 = float(data["A2"])
        C1 = float(data["C1"])
        C2 = float(data["C2"])

        k_to_params[k] = {"A1": A1, "A2": A2, "C1": C1, "C2": C2}

    return k_to_params



class LegitLinkGenerator:
    """
    用于随机生成 Alice–Bob 的位置和 Bob 链路的语义信道模型。
    - Alice 固定在 (0, 0)，你也可以稍后改成参数
    - Bob 在给定的矩形范围内随机采样
    - 对每个 k 构造一个 SemanticChannelModel（Bob 链路）
    """

    def __init__(self,
                 k_list,
                 x_min: float = 10.0,
                 x_max: float = 100.0,
                 y_min: float = -50.0,
                 y_max: float = 50.0,
                 # 大尺度 & 阴影参数（Bob 链路）
                 K0: float = 1.0,
                 d0: float = 1.0,
                 alpha: float = 3.5,
                 shadow_std_dB: float = 4.0,
                 # 天线 & MIMO 参数
                 Gt_dB: float = 0.0,
                 Gr_dB: float = 0.0,
                 mimo_tx: int = 1,
                 mimo_rx: int = 1,
                 normalize_xi: bool = True,
                 rng: np.random.Generator | None = None):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

        self.K0 = K0
        self.d0 = d0
        self.alpha = alpha
        self.shadow_std_dB = shadow_std_dB
        self.Gt_dB = Gt_dB
        self.Gr_dB = Gr_dB
        self.mimo_tx = mimo_tx
        self.mimo_rx = mimo_rx
        self.normalize_xi = normalize_xi

        self.rng = rng if rng is not None else np.random.default_rng()
        self.k_to_params = load_k_params_from_dir(k_list=k_list)

    def sample_positions(self) -> Tuple[Tuple[float, float], Tuple[float, float], float]:
        """
        返回:
          alice_pos = (0.0, 0.0)
          bob_pos   = (x_b, y_b)
          distance  = |Alice-Bob| 的欧氏距离
        """
        alice_pos = (0.0, 0.0)
        x_b = self.rng.uniform(self.x_min, self.x_max)
        y_b = self.rng.uniform(self.y_min, self.y_max)
        bob_pos = (x_b, y_b)

        dx = x_b - alice_pos[0]
        dy = y_b - alice_pos[1]
        distance = float(np.sqrt(dx * dx + dy * dy))
        bob_pos = (1.0, 1.0)
        distance = float(np.sqrt(2))
        return alice_pos, bob_pos, distance

    def build_bob_semantic_models(self):
        
        alice_pos, bob_pos, distance = self.sample_positions()
        
        models_dict: Dict[int, SemanticChannelModel] = {}
        for k, p in self.k_to_params.items():
            scm = SemanticChannelModel(
                A1=p["A1"],
                A2=p["A2"],
                C1=p["C1"],
                C2=p["C2"],
                K0=self.K0,
                d0=self.d0,
                alpha=self.alpha,
                shadow_std_dB=self.shadow_std_dB,
                Gt_dB=self.Gt_dB,
                Gr_dB=self.Gr_dB,
                mimo_tx=self.mimo_tx,
                mimo_rx=self.mimo_rx,
                normalize_xi=self.normalize_xi
            )
            models_dict[k] = scm

        return models_dict, alice_pos, bob_pos, distance
