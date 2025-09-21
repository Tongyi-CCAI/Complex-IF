import json
import os
import time
import argparse
import re
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
import traceback
import multiprocessing
import sys
import requests
import dashscope  # pip install dashscope
from dashscope import Generation
from http import HTTPStatus

def get_api_response(
    prompt,
    model_name,
    api_key,
    api_url,
) -> str:
    message = prompt[1]
    i = 0
    maxtry = 30
    while i < maxtry:
        try:
            client = OpenAI(
                api_key = os.getenv("DASHSCOPE_API_KEY", api_key),
                base_url=api_url,
                )

            stream = client.chat.completions.create(
                model=model_name,
                messages=prompt,
            )
            response = stream.choices[0].message.content
            return response
        except Exception as e:
            print(f"Try {i}/{maxtry}\t message:{prompt} \tError:{e}", flush=True)
            i += 1
            time.sleep(2)
            continue
    return "Err"


def run_evaluation(save_path, datas, num_pool, evaluation_model, model, model_name, api_key, api_base):
    _input = [{"data": i, "evaluation_model": evaluation_model, "save_path":save_path, "model_name": model_name, "api_key": api_key, "api_base": api_base} 
              for i in datas if i]
    
    # 根据操作系统选择合适的启动方法
    if sys.platform != 'win32':
        multiprocessing.set_start_method('fork', force=True)
    
    with multiprocessing.Pool(processes=num_pool) as pool:
        # 使用for循环方式，更节省内存
        for _ in tqdm(pool.imap(model, _input, chunksize=1), 
                     total=len(_input), 
                     desc='Processing', 
                     ncols=100):
            pass