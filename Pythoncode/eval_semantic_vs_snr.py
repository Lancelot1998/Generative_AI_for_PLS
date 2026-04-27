# -*- coding: utf-8 -*-
"""
eval_semantic_vs_snr.py

功能：
1. 加载你训练好的 DeepSC Rayleigh 模型 (.pth)
2. 定义一个符合 PDF 的几何 + 阴影 + Rayleigh 小尺度信道
3. 在不同的信道条件下（通过改变 P、d、sigma2），
   用 DeepSC 做推理，收集一堆样本点 (gamma, xi)
4. 用 logistic 曲线拟合 PDF 中的式 (3)：

   xi(gamma) = A1 + (A2 - A1) / ( 1 + exp( - (C1 * log(gamma) + C2) ) )

备注：
- 这里为了简单，把 K 固定住了（相当于拟合单个 K 下的参数组），
  所以函数写成 xi(gamma; A1,A2,C1,C2)。
- 语义相似度 xi 用一个简单的 token 级一致度来近似，
  你可以之后换成 BLEU / BERT 相似度。
"""

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))



import math
import json
import argparse
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.optimize import curve_fit  # pip install scipy

from models_jyz.transceiver_jyz import DeepSC               # 你的模型定义
from dataset import EurDataset, collate_data # 原 DeepSC 数据集
from channel.GeoShadowRayleighChannelPro import GeoShadowRayleighChannel
from channel.Arg import Args

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_deepsc_model(n, checkpoint_path: str,
                      vocab_path: str,
                      d_model: int = 128,
                      dff: int = 512,
                      num_layers: int = 4,
                      num_heads: int = 8) -> Tuple[nn.Module, dict]:
    """
    按照 main.py 中的构造方式，加载训练完的 DeepSC 模型和 vocab
    """
    vocab = json.load(open(vocab_path, "rb"))
    token_to_idx = vocab["token_to_idx"]
    num_vocab = len(token_to_idx)

    model = DeepSC(num_layers,
                   src_vocab_size=num_vocab,
                   trg_vocab_size=num_vocab,
                   src_max_len=num_vocab,
                   trg_max_len=num_vocab,
                   d_model=d_model,
                   num_heads=num_heads,
                   dff=dff,
                   k = n,
                   dropout=0.1).to(device)

    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, vocab


# ====================================
# 2. 信道：几何 + 阴影 + Rayleigh 小尺度
# ====================================

# def power_normalize(x: torch.Tensor, target_power: float = 1.0) -> torch.Tensor:
#     """
#     简单的功率归一化：保证平均功率 ≈ target_power
#     """
#     power = (x * x).mean().sqrt()
#     if power > 0:
#         x = x * (math.sqrt(target_power) / power)
#     return x
#
#
# class GeoShadowRayleighChannel:
#     """
#     几何感知 + 阴影 + Rayleigh 小尺度信道
#
#     h = G(d, χ) * g
#     |G|^2 = K0 * (d/d0)^(-alpha) * 10^(χ/10)
#     g ~ CN(0,1)
#     gamma = P * |h|^2 / sigma2
#     """
#
#     def __init__(self,
#                  K0: float = 1.0,
#                  d0: float = 1.0,
#                  alpha: float = 3.5,
#                  shadow_std_dB: float = 4.0,
#                  normalize_tx: bool = True):
#         self.K0 = K0
#         self.d0 = d0
#         self.alpha = alpha
#         self.shadow_std_dB = shadow_std_dB
#         self.normalize_tx = normalize_tx
#
#     def _sample_large_scale_gain(self, distance: float):
#         """生成 |G|, |G|^2"""
#         shadow = torch.normal(0.0, self.shadow_std_dB, size=[1]).to(device)
#         pl = (distance / self.d0) ** (-self.alpha)
#         G_abs2 = self.K0 * pl * (10.0 ** (shadow / 10.0))
#         G_abs = torch.sqrt(G_abs2)
#         return G_abs, G_abs2
#
#     def _sample_small_scale(self):
#         """g ~ CN(0,1)"""
#         g_real = torch.normal(0.0, math.sqrt(0.5), size=[1]).to(device)
#         g_imag = torch.normal(0.0, math.sqrt(0.5), size=[1]).to(device)
#         return g_real, g_imag
#
#     def __call__(self,
#                  Tx_sig: torch.Tensor,
#                  P: float,
#                  distance: float,
#                  sigma2: float) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         Tx_sig: [B, N, 2]  (I, Q)
#         P:     发射功率（线性）
#         distance: 几何距离 d
#         sigma2: 噪声功率（线性）
#
#         返回:
#           Rx_sig: 经过信道 + 噪声的信号 [B, N, 2]
#           gamma: 这一拍的等效 SNR (标量 Tensor)
#         """
#         x = Tx_sig
#         if self.normalize_tx:
#             x = power_normalize(x)
#
#         B, N, _ = x.shape
#
#         # 1) 大尺度增益 |G|^2 （几何 + 阴影）
#         G_abs, G_abs2 = self._sample_large_scale_gain(distance)  # 标量 tensor
#
#         # 2) 小尺度 Rayleigh: |g|^2 ~ Exp(1)
#         #   注意这里我们不关心相位，只要 |g|^2 和 SNR 就够了
#         g_abs2 = torch.distributions.Exponential(rate=torch.tensor(1.0)).sample().to(device)  # 标量
#         g_abs = torch.sqrt(g_abs2)
#
#         # 3) 总复信道幅度 |h| = |G| * |g|
#         h_abs = G_abs * g_abs  # 标量
#         h_abs2 = G_abs2 * g_abs2  # 标量
#
#         # 4) 施加信道：y = sqrt(P) * |h| * x   （逐元素标量放大）
#         y = math.sqrt(P) * h_abs * x  # 形状仍是 [B, N, 2]
#
#         # 5) 加 AWGN 噪声
#         noise = torch.normal(
#             mean=0.0,
#             std=math.sqrt(sigma2),
#             size=y.shape
#         ).to(device)
#         y_noisy = y + noise  # [B, N, 2]
#
#         # 6) 等效 SNR：gamma = P * |h|^2 / sigma2
#         gamma = P * h_abs2 / sigma2  # 标量
#
#         return y_noisy, gamma


# =======================================
# 3. 语义相似度：简单 token 匹配 (可换)
# =======================================

def decode_tokens_to_text(ids: List[int], idx_to_token: dict) -> str:
    """
    把一个 token id 序列变成字符串（简单用空格拼）
    """
    words = []
    for tid in ids:
        tok = idx_to_token.get(int(tid), "")
        if tok in ["<PAD>", "<START>", "<END>"]:
            continue
        if tok == "":
            continue
        words.append(tok)
    return " ".join(words)


def semantic_similarity_simple(ref_ids: np.ndarray,
                               pred_ids: np.ndarray,
                               idx_to_token: dict) -> float:
    """
    改造后的“语义相似度”：
    - 不再做逐位置 token 严格匹配
    - 使用 Bag-of-Words 的 Jaccard 相似度：
        ξ = |W_ref ∩ W_pred| / |W_ref ∪ W_pred|
      其中 W_ref / W_pred 是去掉特殊符号后的“词集合”（忽略顺序）

    这样更接近“语义重叠度”，不会因为一个 token 偏移就把后面全判错。
    """
    # 先把 id → token
    special_tokens = {"<PAD>", "<START>", "<END>"}

    ref_tokens = set()
    for tid in ref_ids:
        tok = idx_to_token.get(int(tid), "")
        if tok in special_tokens or tok == "":
            continue
        ref_tokens.add(tok)

    pred_tokens = set()
    for tid in pred_ids:
        tok = idx_to_token.get(int(tid), "")
        if tok in special_tokens or tok == "":
            continue
        pred_tokens.add(tok)

    # 处理空集情况
    if len(ref_tokens) == 0 and len(pred_tokens) == 0:
        return 1.0  # 两边都空，认为完全一致
    if len(ref_tokens) == 0 or len(pred_tokens) == 0:
        return 0.0  # 一边空一边不空，完全不相似

    inter = len(ref_tokens & pred_tokens)
    union = len(ref_tokens | pred_tokens)
    if union == 0:
        return 0.0
    return inter / union


def semantic_similarity_simple1(ref_ids: np.ndarray,
                               pred_ids: np.ndarray,
                               idx_to_token: dict) -> float:
    """
    一个非常简单的语义相似度：
    - 解码成 token 序列
    - 计算位置对齐下的 token 一致比例
    你之后可以换成 BLEU / BERT 相似度
    """
    ref_txt = decode_tokens_to_text(ref_ids, idx_to_token)
    pred_txt = decode_tokens_to_text(pred_ids, idx_to_token)

    ref_tokens = ref_txt.split()
    pred_tokens = pred_txt.split()
    if len(ref_tokens) == 0:
        return 0.0
    same = sum(1 for a, b in zip(ref_tokens, pred_tokens) if a == b)
    return same / len(ref_tokens)


# ====================================
# 4. DeepSC 前向：带“自定义信道”的版本
# ====================================

def forward_deepsc_with_channel(model: nn.Module,
                                sents: torch.Tensor,
                                channel: GeoShadowRayleighChannel,
                                P: float,
                                distance: float,
                                sigma2: float,
                                pad_idx: int):
    """
    使用 DeepSC + 自定义信道做一次前向
    - 为了简单，src = trg = sents，不使用 mask / look-ahead
    - 通道加在 encoder 输出到 channel_decoder 之间
    """

    src = sents
    trg = sents  # 对于自编码任务，src 和 trg 相同

    src_mask = None
    look_ahead_mask = None
    trg_padding_mask = None

    # 1. Encoder
    enc_out = model.encoder(src, src_mask)           # [B,L,d_model]

    # 2. Channel encoder: d_model -> 16
    ch_enc = model.channel_encoder(enc_out)          # [B,L,16]
    B, L, C = ch_enc.shape   # C=16
    Tx_sig = ch_enc.view(B, L * (C // 2), 2)         # [B,N,2], N = L*8

    # 3. 通过几何+阴影信道
    Rx_sig, gamma = channel(Tx_sig, P=P,
                            distance=distance,
                            sigma2=sigma2)           # [B,N,2], scalar gamma

    # 4. reshape 回 [B,L,16]
    ch_out = Rx_sig.view(B, L, C)

    # 5. Channel decoder: 16 -> d_model
    ch_dec = model.channel_decoder(ch_out)           # [B,L,d_model]

    # 6. Decoder：这里直接把 trg token 丢进去（不做 shift，简单处理）
    dec_out = model.decoder(trg, ch_dec,
                            look_ahead_mask,
                            trg_padding_mask)        # [B,L,d_model]

    # 7. 映射到词表
    logits = model.dense(dec_out)                    # [B,L,vocab]

    return logits, gamma


# =========================================
# 5. 收集 (gamma, xi) 样本，用于拟合 Logistic
# =========================================

def collect_samples(model: nn.Module,
                    vocab: dict,
                    checkpoint_name: str,
                    P_list: List[float],
                    distance: float,
                    sigma2: float,
                    max_batches: int = 100,
                    batch_size: int = 64) -> Tuple[np.ndarray, np.ndarray]:
    """
    在不同 P 值下，用 DeepSC + 几何信道推理，收集一堆 (gamma, xi) 样本
    """

    token_to_idx = vocab["token_to_idx"]
    idx_to_token = {v: k for k, v in token_to_idx.items()}
    pad_idx = token_to_idx["<PAD>"]

    test_set = EurDataset("test")
    loader = DataLoader(test_set,
                        batch_size=batch_size,
                        shuffle=False,
                        collate_fn=collate_data)

    channel = GeoShadowRayleighChannel()

    all_gammas = []
    all_xi = []

    print(f"== Collect samples for checkpoint {checkpoint_name} ==")
    for P in P_list:
        print(f"\n--- P = {P:.3f}, distance = {distance:.1f}, sigma2 = {sigma2:.4f} ---")
        batch_count = 0
        with torch.no_grad():
            for sents in loader:
                sents = sents.to(device)
                logits, gamma = forward_deepsc_with_channel(
                    model, sents, channel,
                    P=P,
                    distance=distance,
                    sigma2=sigma2,
                    pad_idx=pad_idx
                )

                # 预测 token
                pred_ids = torch.argmax(logits, dim=-1)   # [B,L]
                sents_np = sents.cpu().numpy()
                pred_np = pred_ids.cpu().numpy()

                # 这里简单地给整个 batch 用同一个 gamma
                gamma_val = float(gamma.item())

                # 逐句计算语义相似度
                for ref_ids, out_ids in zip(sents_np, pred_np):
                    xi_val = semantic_similarity_simple(
                        ref_ids, out_ids, idx_to_token
                    )
                    all_gammas.append(gamma_val)
                    all_xi.append(xi_val)

                batch_count += 1
                if batch_count >= max_batches:
                    break

        print(f"  Collected {batch_count} batches for P={P}")

    gammas = np.array(all_gammas, dtype=np.float64)
    xis = np.array(all_xi, dtype=np.float64)

    print(f"\nTotal samples: {len(gammas)}")
    return gammas, xis


# ========================
# 6. Logistic 拟合 ξ(γ)
# ========================

def xi_logistic(gamma, A1, A2, C1, C2):
    """
    对应式 (3) 的形式（省略 K，下标 K 的依赖体现在参数上）：

    xi(gamma) = A1 + (A2 - A1) / (1 + exp( - (C1 * log(gamma) + C2) ))
    """
    gamma = np.clip(gamma, 1e-8, None)
    return A1 + (A2 - A1) / (1.0 + np.exp(-(C1 * np.log(gamma) + C2)))


def fit_logistic(n, gammas: np.ndarray, xis: np.ndarray):
    """
    用 scipy.optimize.curve_fit 拟合 xi(gamma)，
    得到 A1, A2, C1, C2
    """

    # 只用 0<xi<1 的样本，防止一些极端值影响
    mask = np.logical_and(xis >= 0.0, xis <= 1.0)
    gammas_fit = gammas[mask]
    xis_fit = xis[mask]

    # 初始值 & 边界（xi 大概在 [0,1]）
    p0 = [0.0, 1.0, 1.0, 0.0]  # A1, A2, C1, C2
    bounds = ([-0.5, 0.0, -np.inf, -np.inf],
              [1.0, 1.5,  np.inf,  np.inf])

    popt, pcov = curve_fit(xi_logistic,
                           gammas_fit,
                           xis_fit,
                           p0=p0,
                           bounds=bounds,
                           maxfev=20000)

    A1, A2, C1, C2 = popt
    print("\n=== Fitted logistic parameters (for this K) ===")
    print(f"A1 = {A1:.4f}")
    print(f"A2 = {A2:.4f}")
    print(f"C1 = {C1:.4f}")
    print(f"C2 = {C2:.4f}")

    save_dir = "/models/Qwen/lyq_data/logit_p"
    save_file = f"params_{n}.json"

    save_path = os.path.join(save_dir, save_file)

    params = {
        "A1": float(A1),
        "A2": float(A2),
        "C1": float(C1),
        "C2": float(C2)
    }

    with open(save_path, "w") as f:
        json.dump(params, f, indent=4)

    print(f"参数已保存到: {save_path}")


    return popt, pcov


# ==============
# 7. main 函数
# ==============

def main_formu(n):
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--checkpoint",
    #                     default="./data/lyq/checkpoint_80.pth")
    # parser.add_argument("--vocab-file",
    #                     default="./data/lyq/vocab.json")
    # parser.add_argument("--distance", type=float, default=10.0,
    #                     help="几何距离 d（米）")
    # parser.add_argument("--sigma2", type=float, default=0.1,
    #                     help="噪声功率 σ^2")
    # parser.add_argument("--P-min", type=float, default=0.1)
    # parser.add_argument("--P-max", type=float, default=2.0)
    # parser.add_argument("--P-steps", type=int, default=8)
    # parser.add_argument("--max-batches", type=int, default=50)
    # parser.add_argument("--batch-size", type=int, default=64)
    # args = parser.parse_args()

    args = Args(n)
    # 1. 加载模型和词表
    model, vocab = load_deepsc_model(n, args.checkpoint, args.vocab_file)

    # 2. 决定要扫哪些发射功率 P（你也可以改成扫 sigma2）
    P_list = np.linspace(args.P_min, args.P_max, args.P_steps).tolist()

    # 3. 收集样本 (gamma, xi)
    gammas, xis = collect_samples(
        model=model,
        vocab=vocab,
        checkpoint_name=os.path.basename(args.checkpoint),
        P_list=P_list,
        distance=args.distance,
        sigma2=args.sigma2,
        max_batches=args.max_batches,
        batch_size=args.batch_size
    )

    # 4. 拟合 logistic 曲线
    fit_logistic(n, gammas, xis)


if __name__ == "__main__":
    nn = [4,8,16,32,64]
    for n in nn:
        main_formu(n)
