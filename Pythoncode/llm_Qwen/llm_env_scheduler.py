# llm_env_scheduler.py
# -*- coding: utf-8 -*-

from typing import Callable

class LLMEnvScheduler:
    """
    控制“生成 Eve 环境”时 LLM 的介入频率：
      - llm_every = 1  -> 每次都用 LLM
      - llm_every = 2  -> 每 2 次用 1 次 LLM，其余随机
      - llm_every = 4  -> 每 4 次用 1 次 LLM，其余随机
      ...

    注意：这里的“次数”建议定义为“每次新建 env 的次数”，
    即与你的 train_ppo_multi_env 里的 refresh_every 对齐。
    """
    def __init__(self, llm_every: int = 1, start_with_llm: bool = True):
        assert llm_every >= 1
        self.llm_every = int(llm_every)
        self.start_with_llm = bool(start_with_llm)
        self._counter = 0

    def reset(self):
        self._counter = 0

    def use_llm_this_time(self) -> bool:
        """
        返回：当前这次 make_env() 是否走 LLM 分支
        """
        t = self._counter
        self._counter += 1

        if self.llm_every == 1:
            return True

        # 两种对齐方式：是否从第 0 次就用 LLM
        if self.start_with_llm:
            return (t % self.llm_every) == 0
        else:
            return (t % self.llm_every) == (self.llm_every - 1)

    def wrap_make_env(self,
                      make_env_llm: Callable[[], object],
                      make_env_random: Callable[[], object]) -> Callable[[], object]:
        """
        返回一个混合版 make_env()，供 train_ppo_multi_env 直接使用。
        """
        def mixed_make_env():
            if self.use_llm_this_time():
                env = make_env_llm()
                # 你也可以给 env 打个标记方便记录
                setattr(env, "eve_source", "llm")
                return env
            else:
                env = make_env_random()
                setattr(env, "eve_source", "random")
                return env

        return mixed_make_env
