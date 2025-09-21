import json
import argparse
from generate import api_generate
from key import ANSWER_prompt
import re
import uuid
import os
import numpy as np
from tqdm import tqdm
from pathlib import Path
import shutil 
import yaml
# from llm_predictor import LLMPredictor
import sys
from pathlib import Path

def extract_number(key):
    match = re.search(r'\d+', key)
    return int(match.group()) if match else float('inf')  # Provide a default value if no number is found

def read_file(update_path):
    lines = open(update_path).readlines()
    inputs = []
    update_datas = []
    for idx, line in enumerate(lines):
        print(idx)
        data = json.loads(line)
        update_datas.append(data)
    return update_datas

def extract_instructions(data):
    formatted_instructions = []
    sub_instructions = data.get("sub_instructions", {})
    keys = sub_instructions.keys()
    sorted_keys = sorted(keys, key=extract_number)
    for key in sorted_keys:
        sub_instruction = sub_instructions[key]
        instruction = sub_instruction.get("instruction", "")
        constraints = sub_instruction.get("scoring_questions", [])
        constraints = "\n".join(constraints)
        
        formatted_text = f"{key}:\n{instruction}\nconstraints:\n{constraints}"
        formatted_instructions.append(formatted_text)
    
    return formatted_instructions

def sort_output(input_path, output_path):
    lines = open(input_path, encoding='utf-8').readlines()
    inputs = []
    datas = []
    for idx, line in enumerate(lines):
        data = json.loads(line)
        datas.append(data)
    datas.sort(key=lambda x: (x['main_id'], x['chunk_id']))

    main_id_map = {}

    for data in datas:        
        main_id = data['main_id']
        
        if main_id in main_id_map:
            main_id_map[main_id]['generated'] + data['generated']
        else:
            main_id_map[main_id] = data
            main_id_map[main_id]['generated'] = data['generated']

    datas = list(main_id_map.values())

    with open(output_path, 'w', encoding='utf-8') as fw:
        for line in datas:
            fw.write(json.dumps(line, ensure_ascii=False) + '\n')

def evaluate_data(input_path, output_path, exist_data_path=None, config=None, tmp_output_path=None):
    truth_labels = []
    if exist_data_path is not None:
        exist_data = read_file(exist_data_path)
        print("total data from existing path:{}".format(len(exist_data)))
    else:
        exist_data = []
    if True:
        with open(input_path, 'r', encoding='utf-8') as f:
            datas = json.load(f)
            print("the size of dataset is:{}".format(len(datas)))
            inputs = []
            for data in tqdm(datas):
                overlaps = []
                main_id = data.get("main_id", "")
                flag = False
                for exist_dt in exist_data:
                    if exist_dt['main_id'] == main_id:
                        overlaps.append(exist_dt)

                instruction = data.get("input", "") 
                formatted_text = f"Instruction:\n{instruction}\n\n"
                sub_instructions = extract_instructions(data)

                all_sub_instructions = "\n\n".join(sub_instructions)
                # sub_instruction_list = [all_sub_instructions]
                if len(formatted_text+all_sub_instructions) > 8192:
                    sub_1_instruction = "\n\n".join(sub_instructions[:len(sub_instructions)//2])
                    sub_2_instruction = "\n\n".join(sub_instructions[len(sub_instructions)//2:])
                    sub_instruction_list = [sub_1_instruction, sub_2_instruction]
                else:
                    sub_instruction_list = [all_sub_instructions]

                chunck_id = 0
                flag_record = False
                for sub_instruct in sub_instruction_list:
                    chunck_id += 1
                    input_text=formatted_text + sub_instruct
                    for overlap in overlaps:
                        if overlap['input'] == input_text:
                            flag_record = True
                    if flag_record:
                        continue
                    item = ANSWER_prompt.format(input_text = input_text)
                    line = dict()
                    messages={"system": "You are a helpful assistant.", "user": item}
                    # line['instruction'] = instruction
                    # line['input'] = item
                    # line ['scoring_questions'] = data['scoring_questions']
                    line['prompt'] = messages
                    line['main_id'] = data['main_id']
                    line['input'] = input_text
                    line['chunk_id'] = chunck_id
                    inputs.append(line)
                    
        print("begin generate the answer for dataset, the size of dataset is:{}".format(len(inputs)))
        sort_results = api_generate(inputs, config, config['process_num'], config['port'], tmp_output_path)
        sort_output(tmp_output_path, output_path)
        # with open(output_path, 'w', encoding='utf-8') as fw:
        #     for input_dt, output_dt in zip(inputs, sort_results):
        #         input_dt['generated'] = output_dt
        #         del input_dt['prompt']
        #         fw.write(json.dumps(input_dt, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    data_path = sys.argv[1]  # read data path
    config_path = Path(sys.argv[2])         # read config path
    tmp_output_path = sys.argv[3]
    output_path = sys.argv[4] # read save path
    if len(sys.argv) == 6:
        exist_data_path = sys.argv[5]
    else:
        exist_data_path = None
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)    # 
    
    # Extract directory from output_path
    output_dir = os.path.dirname(output_path)
    
    # Create directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    evaluate_data(data_path, output_path, exist_data_path, config, tmp_output_path)
    # sort_output(tmp_path, sort_path)
