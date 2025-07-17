import os
import json
import time

from src.utils.config import load
from src.datasets import oss_dataset
from src.models import chatbot
from src.evaluators import *

from src.utils.factory import EvaluatorFactory
from src.generators.analyzer import LogAnalyzer
from src.utils.oss_util import OssUtil

import argparse
from loguru import logger
# 创建解析器
parser = argparse.ArgumentParser(description='This is a description of your program.')
# 添加命令行参数
parser.add_argument('--config_path', type=str, default='config/evaluator-ifeval-gpt-gpt.yaml', help='config_path')
if __name__ == '__main__':

    # 解析命令行参数
    args = parser.parse_args()
    config_path = args.config_path
    config = load(open(config_path))
    logger.info(config)


    evaluator = EvaluatorFactory.from_config(config).build()
    average_score, result = evaluator()
    # print(result[0])

    eval_task = {
        'average_score': average_score,
        'config': config,
        'result': result
    }
    # Convert eval_task dictionary to a JSON string
    json_str = json.dumps(eval_task, ensure_ascii=False)
    output_file_name = evaluator.out_file
    # OssUtil.upload(file_content=json_str, file_name=output_file_name)

    # 把output_file_name写入到本地一份
    with open(output_file_name, 'w') as f:
        f.write(json_str)

    # OssUtil.uploadfile(output_file_name)


