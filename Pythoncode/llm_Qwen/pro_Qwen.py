import sys, os
import re
import json

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


def extract_json(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        return text.strip()

    candidate = text[start:end].strip()

    try:
        json.loads(candidate)
        return candidate
    except Exception:
        return candidate


def gen_Qwen(model,tokenizer):

    chatbot = pipeline("text-generation", model=model, tokenizer=tokenizer)

    prompt = """
你现在是一个 JSON 数据生成器。
你的唯一任务是：**只输出 JSON，不允许输出任何解释、提示词、说明文字、额外句子**。

严格只输出一个 JSON 对象，不包含 ```json、解释、提示词、代码块、注释 等任何内容。

通信场景：Tx(0,0,0) → Rx(50,0,0)
请根据无线通信与物理层安全（PLS）场景，随机，注意是随机，生成一名窃听者 Eve 的位置和信道参数，并给出窃听风险等级。

输出格式（必须完全匹配键名，不要多也不要少）：

{
    "Eve_position": [80.0, 20.0, 0.0],
    "distance_e": 85.0,
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
现在开始生成：**只输出 JSON，不允许有额外内容**。
"""

    out = chatbot(
        prompt,
        max_new_tokens=400,
        temperature=0.1,
        pad_token_id=tokenizer.eos_token_id,
    )[0]["generated_text"]

    # HuggingFace pipeline 返回的是：prompt + completion
    # 我们只对 “补全部分” 做处理，避免把 prompt 里的内容当成 JSON
    completion = out[len(prompt):]

    # 从 completion 里强行提取 JSON 部分
    final_json = extract_json(completion)

    return final_json
