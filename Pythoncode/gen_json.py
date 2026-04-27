import json
from pathlib import Path
from typing import Dict, Any, List

# 1. 输入文件路径
json1 = Path("/models/Qwen/lyq_data/pythoncode/DeepSCmaster/results/ppo_random/Random_refresh_every_5000_2_log.json")
json2 = Path("/models/Qwen/lyq_data/pythoncode/DeepSCmaster/results/ppo_random/Random_refresh_every_5000_1_log.json")

# 2. 输出文件路径
out_path = Path("/models/Qwen/lyq_data/pythoncode/DeepSCmaster/results/ppo_random/Random_refresh_every_5000_gen_log.json")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def average_logs_dict_mode(log1: Dict[str, Any],
                           log2: Dict[str, Any]) -> Dict[str, Any]:
    """
    log1, log2 形如:
    {
      "episode": [...],
      "reward": [...],
      "xi_B":   [...],
      ...
    }

    对每个 key 下的“数值型 list”做逐元素平均。
    """
    avg_log: Dict[str, Any] = {}

    # 取 key 的并集，防止有的文件多/少几个字段
    keys = set(log1.keys()) | set(log2.keys())

    for k in keys:
        v1 = log1.get(k, None)
        v2 = log2.get(k, None)

        # 两边都有，并且都是 list
        if isinstance(v1, list) and isinstance(v2, list):
            # list 长度取最短，防止不等长
            n = min(len(v1), len(v2))
            merged_list: List[Any] = []

            for i in range(n):
                a = v1[i]
                b = v2[i]

                # 都是数字 -> 做平均
                if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                    merged_list.append((a + b) / 2.0)
                else:
                    # 否则保持第一个版本（或者你按需要改）
                    merged_list.append(a)

            avg_log[k] = merged_list

        # 如果是标量数字，也可以平均一下
        elif isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
            avg_log[k] = (v1 + v2) / 2.0

        else:
            # 其他情况（比如字符串）：优先拿第一个文件里的
            avg_log[k] = v1 if v1 is not None else v2

    return avg_log


def main():
    log1 = load_json(json1)
    log2 = load_json(json2)

    print(f"Loaded keys from {json1.name}: {list(log1.keys())}")
    print(f"Loaded keys from {json2.name}: {list(log2.keys())}")

    if not isinstance(log1, dict) or not isinstance(log2, dict):
        raise TypeError("当前脚本假设 log 是 dict 结构，如 {'episode': [...], 'reward': [...], ...}")

    avg_log = average_logs_dict_mode(log1, log2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(avg_log, f, ensure_ascii=False, indent=2)

    print(f"Saved averaged log to: {out_path}")


if __name__ == "__main__":
    main()
