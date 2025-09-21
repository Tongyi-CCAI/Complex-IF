from __future__ import annotations

import os
from typing import Any, Dict, Optional, List
import random
import dashscope  # pip install dashscope
from loguru import logger
import time
from src.models import RemoteLLM
from src.schema import BaseLLM, BaseLLMOutput
from src.utils.registry import REGISTRY

@REGISTRY.register_model("tongyi-llm")
class TongyiLLM(BaseLLM):
    system_prompt: str = "You are a helpful assistant."

    @classmethod
    def build(
        cls,
        name: str = "qwen-max",
        api_base: str = os.getenv("DASHSCOPE_API_BASE", "https://poc-dashscope.aliyuncs.com/api/v1"),
        api_key: str = os.getenv(
            "DASHSCOPE_API_KEY", "sk-some-super-secret-key-you-will-never-know"
        ),
        system_prompt: Optional[str] = None,
        run_args: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> TongyiLLM:
        dashscope.api_key = api_key
        dashscope.base_http_api_url = api_base
        # print(run_args)
        return cls(
            name=name,
            run_args=run_args or {},
            system_prompt=system_prompt or "You are a helpful assistant.",
            **kwargs,
        )

    def generate(
        self,
        text: str,
        messages = None,
    ) -> BaseLLMOutput:
        fail_num = 0
        while fail_num<20:
            try:
                fail_num += 1
                if messages is None:
                    messages = [{"role":"system","content":self.system_prompt},{"role":'user', "content":text}]
                # print(**self.run_args)
                completion = dashscope.Generation.call(
                    model=self.name,
                    messages=messages,
                    result_format='message',  # set the result to be "message" format.
                    **self.run_args
                )
                if completion.output:
                    generated = completion.output.choices[0].message.content
                    if '_ACTION_INPUT__' not in generated:
                        generated = generated
                    else:
                        generated = generated.split('_ACTION_INPUT__:')[1].split('__ACTION__:')[0].strip()
                elif completion.code=='InvalidParameter' and 'input length' in completion.message:
                    generated = "输入长度超过限制"
                    return BaseLLMOutput(
                        generated=generated,
                        prompt_tokens=0,
                        completion_tokens=0,
                    )
                elif completion.code=='DataInspectionFailed':
                    generated = "输入中包含不合适的内容"
                    return BaseLLMOutput(
                        generated=generated,
                        prompt_tokens=0,
                        completion_tokens=0,
                    )
                else:
                    # logger.error(f"{completion}")
                    time.sleep(1)
                    continue
                return BaseLLMOutput(
                    generated=generated,
                    prompt_tokens=completion.usage.input_tokens,
                    completion_tokens=completion.usage.output_tokens,
                )
            except Exception as e:
                time.sleep(1)
                logger.error(f"{self.name} 模型调用失败 {e}")
                continue
                # break
        # 如果走到了这里，说明没有成功返回调用结果
        logger.error(f"在尝试 {fail_num} 次之后，{self.name} 模型调用失败")