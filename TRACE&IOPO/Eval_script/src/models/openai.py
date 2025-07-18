from __future__ import annotations

import os
from typing import Any, Dict, Optional

import openai
from loguru import logger

from src.schema import BaseLLM, BaseLLMOutput
from src.utils.registry import REGISTRY


@REGISTRY.register_model("remote-llm")
class RemoteLLM(BaseLLM):
    system_prompt: str = "You are a helpful assistant."

    @classmethod
    def build(
            cls,
            name: str = "llama2-13b-chat",
            api_base: str = os.getenv("OPENAI_API_BASE", ""),
            api_key: str = os.getenv(
                "OPENAI_API_KEY", "sk-some-super-secret-key-you-will-never-know"
            ),
            system_prompt: Optional[str] = None,
            run_args: Optional[Dict[str, Any]] = None,
            **kwargs: Any,
    ) -> RemoteLLM:
        openai.api_base = api_base
        openai.api_key = api_key
        return cls(
            name=name,
            run_args=run_args or {},
            system_prompt=system_prompt or "You are a helpful assistant.",
            **kwargs,
        )

    def generate(
            self,
            text: str,
    ) -> BaseLLMOutput:
        while True:
            try:
                completion = openai.Completion.create(
                    model=self.name,
                    prompt="\n".join([self.system_prompt, text]),
                    **self.run_args,
                )
                return BaseLLMOutput(
                    generated=completion.choices[0].text,
                    prompt_tokens=completion.usage.prompt_tokens,
                    completion_tokens=completion.usage.completion_tokens,
                )
            except Exception as e:
                logger.error("ServiceUnavailableError", e)
                continue


@REGISTRY.register_model("chatgpt35")
class ChatGPT(GPT):
    def generate(
            self,
            text: str = "",
    ) -> BaseLLMOutput:
        while True:
            try:
                completion = openai.ChatCompletion.create(
                    model=self.name,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": text},
                    ],
                    **self.run_args,
                )
                return BaseLLMOutput(
                    generated=completion.choices[0].message.content,
                    prompt_tokens=completion.usage.prompt_tokens,
                    completion_tokens=completion.usage.completion_tokens,
                )
            except Exception as e:
                logger.error("ServiceUnavailableError", e)
                continue


from openai import OpenAI

@REGISTRY.register_model("vllm")
class VLLM(BaseLLM):
    system_prompt: str = "You are a helpful assistant."
    api_base: str
    api_key: str

    @classmethod
    def build(
            cls,
            name: str = "llama2-13b-chat",
            api_base: str = "http://localhost:8000/v1",
            api_key: str = "EMPTY",
            system_prompt: Optional[str] = None,
            run_args: Optional[Dict[str, Any]] = None,
            **kwargs: Any,
    ) -> VLLM:
        return cls(
            name=name,
            run_args=run_args or {},
            system_prompt=system_prompt or "You are a helpful assistant.",
            api_base=api_base,
            api_key=api_key,
            **kwargs,
        )

    def generate(
            self,
            text: str = "",
            messages=None
    ) -> BaseLLMOutput:
        client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
        )
        fail_num = 0
        while fail_num < 4: ####20 original
            try:
                fail_num += 1
                if messages is None:
                    messages = [{"role": "system", "content": self.system_prompt}, {"role": 'user', "content": text}]
                completion = client.chat.completions.create(
                    model=self.name,
                    messages=messages,
                    **self.run_args,
                    # timeout=10
                )
                # print(completion)
                return BaseLLMOutput(
                    generated=completion.choices[0].message.content,
                    prompt_tokens=completion.usage.prompt_tokens,
                    completion_tokens=completion.usage.completion_tokens,
                )
            except Exception as e:
                logger.error(f"VLLM {self.name}调用失败 {e}")
                continue


import time
import requests


def request_openai_with_alibaba(**config):
    # MAX_API_RETRY = 50
    MAX_API_RETRY = 3
    REQ_TIME_GAP = 3

    request_body = {}
    for key, value in config.items():
        # print(key)
        if key in ['model', 'functions', 'messages', 'temperature', 'max_tokens', 'stop']:
            request_body[key] = value
    if 'messages' not in request_body and 'prompt' in config:
        request_body['messages'] = [{"role": 'user', "content": config['prompt']}]
    if 'messages' not in request_body:
        raise RuntimeError(f"no messages provided.")

    if 'qwen' in request_body['model']:
        raise RuntimeError(f"no qwen model.")
    else:
        HEADERS = {
            'Authorization': 'Bearer ' + config['api_key'],
            'Content-Type': 'application/json'
        }

        # HEADERS = {
        #     'X-AK': f'{config["api_key"]}',
        #     'Content-Type': 'application/json'
        # }

        URL = config['api_base']
        for i in range(MAX_API_RETRY):
            try:
                # print(request_body)
                no_response = 'no_response'
                # request_body['ask'] = True
                response = requests.post(URL, headers=HEADERS, json=request_body)
                # print('after',request_body)
            except Exception as e:
                # print("请求失败", e)
                time.sleep(REQ_TIME_GAP)
                continue
            try:
                response_dict = response.json()
            except Exception as e:
                # print("请求结果转json失败", e, response)
                time.sleep(REQ_TIME_GAP)
                continue
            try:
                # print(response_dict)
                if response_dict["code"] == 200 or response_dict["code"] == 'success':
                    # 重构response
                    # 之竹
                    response_data = response_dict['data']
                    response_ret = {}
                    response_ret['model'] = request_body['model']
                    response_ret['messages'] = request_body['messages']
                    response_ret['usage'] = {'completion_tokens': response_data['completion_tokens'],
                                             'prompt_tokens': response_data['prompt_tokens'],
                                             'total_tokens': response_data['total_tokens']}
                    response_ret['choices'] = []
                    # print(response_data)

                    for xx in response_data['response']['choices']:
                        x = xx['message']
                        if 'response_role' in xx:
                            x['response_role'] = xx['response_role']
                        if 'role' not in x:
                            x['role'] = x['response_role']

                        if 'function_call' in xx:
                            x['function_call'] = xx['function_call']
                        if 'finish_reason' in xx:
                            x['finish_reason'] = xx['finish_reason']
                        if 'content' not in x or x["content"] is None:
                            x["content"] = ''
                        if 'function_call' in x and x['function_call']:
                            response_ret['choices'].append({'message': {'content': x['content'], 'role': x['role'],
                                                                        'function_call': x['function_call']},
                                                            'finish_reason': x['finish_reason']})
                        else:
                            response_ret['choices'].append({'message': {'content': x['content'], 'role': x['role']},
                                                            'finish_reason': x['finish_reason']})

                    return response_ret
                else:
                    # print('response code!= 200', response_dict)
                    # print(request_body)
                    if 'limit exceeded' in response_dict['message']:
                        print(response_dict)
                    if 'context_length_exceeded' in response_dict['message'] or 'repetitive patterns in your prompt' in \
                            response_dict['message']:
                        print('context_length_exceeded', response_dict)
                        response_ret = {}
                        response_ret['model'] = request_body['model']
                        response_ret['messages'] = request_body['messages']
                        response_ret['usage'] = {'completion_tokens': 0, 'prompt_tokens': 0, 'total_tokens': 0}
                        response_ret['choices'] = []
                        response_ret['choices'].append({'message': {'content': 'TERMINATE', 'finish_reason': 'stop',
                                                                    'response_role': 'assistant'}})
                        return response_ret
                    # print('请求失败，状态码为', response_dict)
                    time.sleep(REQ_TIME_GAP)
                    pass
            except Exception as e:
                logger.error(f"Response重构失败 {e}")
                time.sleep(REQ_TIME_GAP)
                continue
        if 'response_dict' in locals() and locals()['response_dict'] is not None:
            # if response_dict["code"] == 500:
            #     print('500', response_dict)
            #     print(request_body)
            #     response_ret = {}
            #     response_ret['model'] = request_body['model']
            #     response_ret['messages'] = request_body['messages']
            #     response_ret['usage'] =  {'completion_tokens':0, 'prompt_tokens':0, 'total_tokens':0}
            #     response_ret['choices'] = []
            #     response_ret['choices'].append({'message':{'content': 'TERMINATE', 'finish_reason': 'stop', 'response_role': 'assistant'}})
            #     return response_ret

            raise RuntimeError(f"Failed after {MAX_API_RETRY} retries.", response_dict)
        else:
            raise RuntimeError(f"Failed after {MAX_API_RETRY} retries.", no_response)


@REGISTRY.register_model("alibaba-remote-llm")
class AlibabaRemoteLLM(BaseLLM):
    system_prompt: str = "You are a helpful assistant."
    api_base: str
    api_key: str

    @classmethod
    def build(
            cls,
            name: str = "llama2-13b-chat",
            api_base: str = os.getenv("OPENAI_API_BASE", ""),
            api_key: str = os.getenv(
                "OPENAI_API_KEY", "sk-some-super-secret-key-you-will-never-know"
            ),
            system_prompt: Optional[str] = None,
            run_args: Optional[Dict[str, Any]] = None,
            **kwargs: Any,
    ) -> AlibabaRemoteLLM:
        api_base = os.getenv("OPENAI_API_BASE", "")
        api_key = os.getenv(
            "OPENAI_API_KEY", "sk-some-super-secret-key-you-will-never-know"
        )
        return cls(
            name=name,
            run_args=run_args or {},
            system_prompt=system_prompt or "You are a helpful assistant.",
            api_base=api_base,
            api_key=api_key,
            **kwargs,
        )

    def generate(
            self,
            text: str,
            messages=None,
    ) -> BaseLLMOutput:
        fail_num = 0
        while fail_num < 20: #### 20
            try:
                fail_num += 1
                if messages is None:
                    messages = [{"role": "system", "content": self.system_prompt}, {"role": 'user', "content": text}]
                completion = request_openai_with_alibaba(
                    model=self.name,
                    messages=messages,
                    api_base=self.api_base,
                    api_key=self.api_key,
                    **self.run_args,
                )
                return BaseLLMOutput(
                    generated=completion['choices'][0]['message']['content'],
                    prompt_tokens=completion["usage"]['prompt_tokens'],
                    completion_tokens=completion["usage"]['completion_tokens'],
                )
            except Exception as e:
                logger.error(f"AlibabaRemoteLLM {self.name}调用失败 {e}")
                continue

        # 如果走到了这里，说明没有成功返回调用结果
        logger.error(f"在尝试 {fail_num} 次之后，{self.name} 模型调用失败")