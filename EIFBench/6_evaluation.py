import json
import os
import time
import argparse
import re
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
import traceback
from key import SCORE_prompt
import multiprocessing
from gpt_api import get_api_response, run_evaluation
import sys
import copy
import re

def load_jsonl(file_path):
    _data = []
    with open(file_path, 'r') as f:
        for data in f:
            jline = json.loads(data)
            _data.append(jline)
    return _data


def get_payload(line):
    input = line['input']
    # instruction = line['instruction']
    question = line['question']
    sub_instruction_key = line['sub_instruction_id']
    sub_instruction = line['sub_instruction']
    if line['output'] != None:
        output = line['output']
    else:
        output = 'None'
    content =  SYS_MSG.format(input=input, output=output, sub_instruction=f"{sub_instruction_key}:{sub_instruction}", question=question)
    payload = {
        # "model": "gpt-4-1106-preview",
        "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": content},
                    ]
        # "max_tokens": 8192,
        # "temperature": 0.0,
        # "top_p": 0.95,
        # "stream": True
    }
    return payload

def extract_instructions(data):
    # 存储提取的instructions
    # instructions = []
    formatted_instructions = []
    # 遍历sub_instructions字典
    sub_instructions = data.get("sub_instructions", {})
    
    # 按照key排序，确保顺序正确
    for key in sorted(sub_instructions.keys()):
        sub_instruction = sub_instructions[key]
        instruction = sub_instruction.get("instruction", "")
        constraints = sub_instruction.get("scoring_questions", [])
        constraints = "\n".join(constraints)
        
        # 格式化输出，包含sub_instruction_x
        formatted_text = f"{key}:\n{instruction}\nconstraints:\n{constraints}"
        formatted_instructions.append(formatted_text)
    
    # 将所有指令组合成一个字符串，用换行符分隔
    final_text = "\n\n".join(formatted_instructions)
    return final_text

def save_jsonl(entry, sava_path):
    with open(sava_path, 'a', encoding='utf-8')as file:
        file.write(json.dumps(entry, ensure_ascii=False)+ "\n")

def get_answer(input_data: dict, retry=30):
    entry, save_path = input_data['data'], input_data['save_path']
    model_name, api_key, api_url = input_data['model_name'], input_data['api_key'], input_data['api_base']
    evaluate_model = input_data['evaluation_model']
    try:
        payload = get_payload(entry)
        generation = get_api_response(payload["messages"], model_name, api_key, api_url)
        if generation == None or generation == "":
            get_answer(input_data, retry=retry-1)

        re_result = re.findall(r'答案：是|答案：否', generation)
        if len(re_result) == 1:
            if "是" in re_result[0]:
                entry['point_judge'] = True
            else:
                entry['point_judge'] = False
        else: 
            if "是" in generation and "否" not in generation:
                entry['point_judge'] = True
            else:
                entry['point_judge'] = False
        if generation == "Err":
            save_jsonl(entry, save_path.replace(".jsonl", "_error.jsonl"))
        else:
            entry['point_explanation'] = generation
            # entry['payload'] = payload
            save_jsonl(entry, save_path)
        return entry
    except Exception as e:
        time.sleep(1.2)
        retry -= 1
        if retry < 0:
            # entry['point_judge'] = False
            # entry['point_explanation'] = "None"
            # entry['payload'] = payload
            save_jsonl(entry, save_path.replace(".jsonl", "_error.jsonl"))
        print(f"retry:剩余{retry}次")
        print(e)
        print(traceback.format_exc())
        get_answer(input_data, retry=retry)

def get_data(data_path, llm_output_path, exist_output_path=""):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    with open(llm_output_path, 'r', encoding='utf-8') as f:
        outputs = [json.loads(line) for line in f.readlines()]

    if exist_output_path != "":
        exist_data = load_jsonl(exist_output_path)
    else:
        exist_data = []
    print("existing length is:{}, our output:{}".format(len(exist_data), len(outputs)))

    datas = []
    total_original = 0
    exist_count = 0
    for i, o in tqdm(enumerate(outputs)):
        d = None
        for i_d in data:
            if i_d['main_id'] == o['main_id']:
                d = i_d
        if d == None:
            continue
        sub_instructions = d['sub_instructions']
        sub_instruction_lists = extract_instructions(d)
        count = 0
        for instruction_key, instruction_item in sub_instructions.items():
            for j, q in enumerate(instruction_item['scoring_questions']):
                count += 1
                total_original += 1
                flag = False
                for exist_dt in exist_data:
                    if o['main_id'] == exist_dt['main_id'] and exist_dt['sub_instruction_id'] == instruction_key and exist_dt['question'] == q and exist_dt["point_explanation"] not in ["None", "Err"]:
                        flag = True
                        break
                if flag:
                    exist_count += 1
                    # print("yes")
                    continue
                inputs = d['input']
                datas.append({
                    "main_id" : o['main_id'],
                    "point_id" : count,
                    "rule" : False,
                    'input': inputs,
                    'sub_instruction_id': instruction_key, # q['sub_instruction_id'],
                    'sub_instruction': instruction_item['instruction'],
                    "question" : q,# q['question'],
                    "output" : o['generated']
                })
    print("original:{}, skip data:{}, total data:{}".format(total_original, exist_count, len(datas)))
    # sys.exist(0)
    return datas

def aggregation(data_path, llm_output_path, input_file, output_file):
    with open(data_path, 'r', encoding='utf-8') as f:
        raw_datas = json.load(f)
    with open(llm_output_path, 'r', encoding='utf-8') as f:
        outputs = [json.loads(line) for line in f.readlines()]
    llm_evaluations = load_jsonl(input_file)
    exist_ids = list(set([l['main_id'] for l in llm_evaluations]))
    new_datas = []
    for i, raw_data in enumerate(tqdm(raw_datas)):
        if raw_data['main_id'] not in exist_ids:
            continue
        new_data = copy.deepcopy(raw_data)
        for sub_instruction_key, sub_instructions in new_data['sub_instructions'].items():
            new_data['sub_instructions'][sub_instruction_key]['point_judge'] = []
            new_data['sub_instructions'][sub_instruction_key]['point_explanation'] = []
            
            for constraint in sub_instructions['scoring_questions']:
                flag = False
                for l in llm_evaluations:
                    if l['main_id'] == new_data['main_id'] and sub_instruction_key == l['sub_instruction_id'] and constraint== l['question']:
                        flag = True
                        new_data['sub_instructions'][sub_instruction_key]['point_judge'].append(l['point_judge'])
                        new_data['sub_instructions'][sub_instruction_key]['point_explanation'].append(l['point_explanation'])
                        break
                if not flag:
                    new_data['sub_instructions'][sub_instruction_key]['point_judge'].append("not")
                    new_data['sub_instructions'][sub_instruction_key]['point_explanation'].append("not")

        for output in outputs:
            if output['main_id'] == new_data['main_id']:
                new_data['output'] = output['generated']
        new_datas.append(new_data)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(new_datas, f, ensure_ascii=False, indent=2)            

def main_run(args):
    datas = get_data(data_path=args.data_path, llm_output_path=args.llm_output_path, exist_output_path=args.exist_output_path)
    run_evaluation(args.output_path, datas, args.num_pool, args.evaluation_model, get_answer, args.evaluation_model, args.api_key, args.api_base)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--llm_output_path", type=str, default="")
    parser.add_argument("--exist_output_path", type=str, default="")
    parser.add_argument("--evaluation_model", type=str, default="")
    parser.add_argument("--num_pool", type=int, default=2)
    parser.add_argument("--output_path", type=str, default="")
    parser.add_argument("--api_key", type=str, default="")
    parser.add_argument("--api_base", type=str, default="")
    args = parser.parse_args()
    SYS_MSG = SCORE_prompt
    main_run(args)
    aggregate_path = args.output_path.replace(".jsonl", "_aggregation.json")
    aggregation(args.data_path, args.llm_output_path, args.output_path, aggregate_path)
