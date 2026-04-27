# qwen_eve_utils.py
from typing import Dict, Tuple
from channel.semantic_channelpro import SemanticChannelModel


def build_eve_models_from_qwen(
    qwen_eve_info: Dict,
    k_to_params: Dict[int, Dict[str, float]],
    normalize_xi: bool = True
) -> Tuple[Dict[int, SemanticChannelModel],
           Tuple[float, float, float],
           float]:
    """
    根据 Qwen 输出的 Eve 场景，构造 Eve 链路的 {k: SemanticChannelModel}

    qwen_eve_info 示例:
    {
      "Eve_position": [x_e, y_e, z_e],
      "distance_e": 80.0,
      "channel_params": {
        "K0": 1.0,
        "d0": 1.0,
        "alpha": 3.5,
        "shadow_std_dB": 4.0,
        "Gt_dB": 0.0,
        "Gr_dB": 0.0,
        "mimo_tx": 1,
        "mimo_rx": 1
      }
    }
    """
    eve_pos = tuple(qwen_eve_info["Eve_position"])
    d_eve = float(qwen_eve_info["distance_e"])
    ch = qwen_eve_info["channel_params"]

    eve_models: Dict[int, SemanticChannelModel] = {}
    for k, p in k_to_params.items():
        eve_models[k] = SemanticChannelModel(
            A1=p["A1"],
            A2=p["A2"],
            C1=p["C1"],
            C2=p["C2"],
            K0=ch.get("K0", 1.0),
            d0=ch.get("d0", 1.0),
            alpha=ch.get("alpha", 3.5),
            shadow_std_dB=ch.get("shadow_std_dB", 4.0),
            Gt_dB=ch.get("Gt_dB", 0.0),
            Gr_dB=ch.get("Gr_dB", 0.0),
            mimo_tx=ch.get("mimo_tx", 1),
            mimo_rx=ch.get("mimo_rx", 1),
            normalize_xi=normalize_xi
        )

    return eve_models, eve_pos, d_eve
