# llm_teacher_policy.py
# -*- coding: utf-8 -*-

import json
import re
from typing import List, Tuple, Sequence, Optional

import torch


class LLMActionTeacher:
    """
    使用 Qwen 直接生成动作建议 (P_dB, k) 列表的“LLM teacher”。

    用法示例（在 test_Qwen_teacher.py 中）：

        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        model_llm = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            device_map="auto",
            trust_remote_code=True
        )

        from llm_teacher_policy import LLMActionTeacher

        llm_teacher = LLMActionTeacher(
            tokenizer=tokenizer,
            model=model_llm,
            top_n_ratio=0.2,
            temperature=0.2,
            max_new_tokens=512,
        )

        teacher_actions = llm_teacher.build_teacher_actions(
            alice_pos=alice_pos,
            bob_pos=bob_pos,
            eve_pos=eve_pos,
            distance_bob=d_bob,
            distance_eve=d_eve,
            P_dB_list=P_dB_list,
            k_values=k_values,
            lambda_e=1.0,
            bob_xi_min=0.5,
            extra_desc="例如：当前为高风险窃听环境，Eve 更靠近 Alice 一侧。"
        )
    """

    def __init__(
        self,
        tokenizer,
        model,
        top_n_ratio: float = 0.2,
        temperature: float = 0.2,
        max_new_tokens: int = 512,
    ):
        """
        参数：
          tokenizer, model : 你在外面用 from_pretrained(MODEL_PATH) 加载好的 Qwen 模型
          top_n_ratio      : 希望 LLM 给出的大致动作个数比例，例如 0.2 表示选出约 20% 的动作
          temperature      : LLM 采样温度，越小越保守
          max_new_tokens   : LLM 生成的最大 token 数
        """
        self.tokenizer = tokenizer
        self.model = model
        self.top_n_ratio = float(top_n_ratio)
        self.temperature = float(temperature)
        self.max_new_tokens = int(max_new_tokens)

    # ===========================
    # 1. 对外主接口：生成动作列表
    # ===========================

    def build_teacher_actions(
        self,
        alice_pos: Tuple[float, float],
        bob_pos: Tuple[float, float],
        eve_pos: Tuple[float, float],
        distance_bob: float,
        distance_eve: float,
        P_dB_list: Sequence[float],
        k_values: Sequence[int],
        lambda_e: float = 1.0,
        bob_xi_min: float = 0.5,
        extra_desc: Optional[str] = None,
    ) -> List[Tuple[float, int]]:
        """
        向 LLM 提问，生成动作建议列表。

        返回：
          teacher_actions: List[(P_dB, k)]，例如 [(8.0, 16), (10.0, 32), ...]
                           若 LLM 解析失败，会 fallback 到简单规则生成。
        """
        num_total_actions = len(P_dB_list) * len(k_values)
        num_suggest = max(1, int(num_total_actions * self.top_n_ratio))

        prompt = self._build_prompt(
            alice_pos=alice_pos,
            bob_pos=bob_pos,
            eve_pos=eve_pos,
            distance_bob=distance_bob,
            distance_eve=distance_eve,
            P_dB_list=P_dB_list,
            k_values=k_values,
            lambda_e=lambda_e,
            bob_xi_min=bob_xi_min,
            num_suggest=num_suggest,
            extra_desc=extra_desc,
        )

        raw_text = self._call_llm(prompt)
        teacher_actions = self._parse_actions_from_llm_output(
            raw_text, P_dB_list, k_values
        )

        if not teacher_actions:
            teacher_actions = self._fallback_actions(P_dB_list, k_values)

        print(f"[LLM Teacher] 最终得到 {len(teacher_actions)} 个推荐动作")
        for (P_dB, k) in teacher_actions[:5]:
            print(f"   (P_dB={P_dB:.1f}, k={k})")

        return teacher_actions

    # ===========================
    # 2. 构造 prompt（带 BEGIN/END_JSON 标记）
    # ===========================

    def _build_prompt(
        self,
        alice_pos: Tuple[float, float],
        bob_pos: Tuple[float, float],
        eve_pos: Tuple[float, float],
        distance_bob: float,
        distance_eve: float,
        P_dB_list: Sequence[float],
        k_values: Sequence[int],
        lambda_e: float,
        bob_xi_min: float,
        num_suggest: int,
        extra_desc: Optional[str] = None,
    ) -> str:
        """
        构造发给 Qwen 的中文 prompt。要求 Qwen 把 JSON 放在 <BEGIN_JSON> 与 <END_JSON> 之间。
        """
        P_dB_str = ", ".join([f"{p:.1f}" for p in P_dB_list])
        k_str = ", ".join([str(k) for k in k_values])

        extra = extra_desc if extra_desc is not None else "无额外说明。"

        prompt = f"""
你是一个无线通信与语义通信的顶级专家，目标是帮助发送端 Alice 选择合适的发射功率 P_dB 和语义压缩粒度 k，
使得：
1. 合法接收端 Bob 接收到的语义相似度尽可能大；
2. 窃听者 Eve 接收到的语义相似度尽可能小。

下面是当前场景的信息（单位可以近似理解，不需要非常精确）：

- Alice 位置: {alice_pos}
- Bob   位置: {bob_pos}, 与 Alice 的距离 d_B ≈ {distance_bob:.2f}
- Eve   位置: {eve_pos}, 与 Alice 的距离 d_E ≈ {distance_eve:.2f}

- 候选发射功率 P_dB 列表（单位 dB）为:
  [{P_dB_str}]

- 候选语义压缩粒度 k 列表为:
  [{k_str}]

你可以认为：
- 更大的 P_dB 往往会同时提高 Bob 和 Eve 的语义相似度；
- 更大的 k 往往能提高 Bob 的语义保真度，但在窃听链路上也可能提高 Eve 的语义保真度；
- 优化目标可以粗略写成:  maximize  (xi_B - {lambda_e:.2f} * xi_E)；
- 同时希望 Bob 的语义相似度不低于 {bob_xi_min:.2f}。

请根据上面的信息和你的专业知识，从给定的 P_dB 和 k 候选中，
推荐大约 {num_suggest} 个 (P_dB, k) 组合，作为“较优动作”。

**重要格式要求（请严格遵守）：**
1. 你只能输出一个 JSON，对象的字段为 "actions"。
2. "actions" 是一个数组，每个元素是形如 {{"P_dB": 8, "k": 16}} 的对象。
3. P_dB 只能从给定列表 [{P_dB_str}] 中选取。
4. k 只能从给定列表 [{k_str}] 中选取。
5. 不要输出除 JSON 以外的任何文字说明。
6. 请把 JSON 放在下面的标记之间：

<BEGIN_JSON>
{{
  "actions": [
    {{"P_dB": 8, "k": 16}},
    {{"P_dB": 10, "k": 32}}
  ]
}}
<END_JSON>

上面的数值只是示例，不要照抄。真正输出时：
- 请删除这个示例内容，换成你推荐的动作。
- 保证整个回答中唯一的 JSON 就在 <BEGIN_JSON> 和 <END_JSON> 中间。
- 不要在 <BEGIN_JSON> 前后再加任何文字。

额外说明（仅供你理解，不需要重复输出）：
{extra}
"""
        return prompt.strip()

    # ===========================
    # 3. 调用 Qwen（用 generate）
    # ===========================

    def _call_llm(self, prompt: str) -> str:
        """
        调用 Qwen2ForCausalLM 生成文本。
        使用 transformers 标准 generate 接口，而不是 model.chat。
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # 只取新生成的部分
        gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        return text

    # ===========================
    # 4. 从 LLM 输出中解析 JSON 为动作列表
    # ===========================

    def _parse_actions_from_llm_output(
        self,
        raw_text: str,
        P_dB_list: Sequence[float],
        k_values: Sequence[int],
    ) -> List[Tuple[float, int]]:
        """
        从 LLM 输出的 raw_text 中解析 JSON，并转成 List[(P_dB, k)]。
        优先使用 <BEGIN_JSON> ... <END_JSON> 之间的内容；
        若没有标记，再退回到大括号截取。
        """
        text = raw_text.strip()
        if not text:
            print("[LLM Teacher] 输出为空，使用 fallback。")
            return []

        # 1) 优先找 BEGIN/END_JSON 之间的内容
        m = re.search(r"<BEGIN_JSON>(.*)<END_JSON>", text, re.S)
        if m:
            json_block = m.group(1).strip()
        else:
            # 2) 如果没标记，就尝试从第一个 '{' 到最后一个 '}' 的片段
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                print("[LLM Teacher] 没找到 JSON 片段，使用 fallback。")
                return []
            json_block = text[start : end + 1]

        try:
            data = json.loads(json_block)
        except json.JSONDecodeError:
            print("[LLM Teacher] JSON 解析失败，使用 fallback。")
            return []

        if "actions" not in data or not isinstance(data["actions"], list):
            print("[LLM Teacher] JSON 中没有 'actions' 字段或类型不对，使用 fallback。")
            return []

        valid_P = set(float(p) for p in P_dB_list)
        valid_k = set(int(k) for k in k_values)

        teacher_actions: List[Tuple[float, int]] = []
        for item in data["actions"]:
            if not isinstance(item, dict):
                continue
            P_val = item.get("P_dB")
            k_val = item.get("k")
            if P_val is None or k_val is None:
                continue
            try:
                P_val = float(P_val)
                k_val = int(k_val)
            except (TypeError, ValueError):
                continue
            if P_val in valid_P and k_val in valid_k:
                teacher_actions.append((P_val, k_val))

        # 去重
        teacher_actions = list(dict.fromkeys(teacher_actions))
        return teacher_actions

    # ===========================
    # 5. 解析失败时的 fallback
    # ===========================

    def _fallback_actions(
        self,
        P_dB_list: Sequence[float],
        k_values: Sequence[int],
    ) -> List[Tuple[float, int]]:
        """
        当 LLM 输出解析失败时的简易 fallback：
        - 选择中等 P_dB（例如中位数附近）
        - 选择中等偏上的 k
        - 返回若干个组合
        """
        print("[LLM Teacher] 使用 fallback 动作集合。")

        if not P_dB_list or not k_values:
            return []

        P_sorted = sorted(P_dB_list)
        k_sorted = sorted(k_values)

        mid_p = P_sorted[len(P_sorted) // 2]
        high_p = P_sorted[-1]
        low_p = P_sorted[0]

        mid_k = k_sorted[len(k_sorted) // 2]
        high_k = k_sorted[-1]
        low_k = k_sorted[0]

        actions = [
            (mid_p, mid_k),
            (high_p, mid_k),
            (mid_p, high_k),
            (low_p, high_k),
            (high_p, low_k),
        ]
        actions = list(dict.fromkeys(actions))
        return actions
