#!/bin/bash

# bash eval.sh grpo0419_320step party qwen
# 检查是否有正确数量的参数传入
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <step_identifier> <input_type> <config>"
    echo "input_type options: party, dialogue, single"
    echo "config options: gpt4o, r1, hiaku, sonnet, qwen, train"
    exit 1
fi

# 分配命令行参数
STEP_IDENTIFIER=$1
INPUT_TYPE=$2

# 根据输入类型选择输入路径
case "$INPUT_TYPE" in
    party)
        DIALOGUE_INPUT_PATH="./data/sorted_party100.json"
        ;;
    dialogue)
        DIALOGUE_INPUT_PATH="./updated_data/sorted_dialogue450.json"
        ;;
    single)
        DIALOGUE_INPUT_PATH="./data/sorted_single450.json"
        ;;
    *)
        echo "Invalid input_type. Choose from: party, dialogue, single"
        exit 1
        ;;
esac

# 固定路径和动态路径组成
TMP_OUTPUT_PATH="./evaluation_results/tmp_${STEP_IDENTIFIER}_output_${INPUT_TYPE}_7B.json"
SORTED_OUTPUT_DIALOGUE_PATH="./evaluation_results/sort_${STEP_IDENTIFIER}_output_${INPUT_TYPE}_7B.json"
EVAL_MODEL="gpt4o"
CONFIG_FILE="./configs/config.yaml"
EVAL_OUTPUT_PATH="./evaluation_results/evaluate_${STEP_IDENTIFIER}_${INPUT_TYPE}.jsonl"

# 设置可见的 GPU 设备
# export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES=0

# 运行第一个 Python 脚本
# python 5_output.py "$DIALOGUE_INPUT_PATH" "$CONFIG_FILE" "$TMP_OUTPUT_PATH" "$SORTED_OUTPUT_DIALOGUE_PATH"

# 检查第一个脚本是否成功执行
if [ $? -ne 0 ]; then
    echo "Error running 5_output.py"
    exit 1
fi

# 运行第二个 Python 脚本
python 6_evaluation.py --data_path "$DIALOGUE_INPUT_PATH" --llm_output_path "$SORTED_OUTPUT_DIALOGUE_PATH" --output_path "$EVAL_OUTPUT_PATH" 
# 检查第二个脚本是否成功执行
if [ $? -ne 0 ]; then
    echo "Error running 6_evaluation.py"
    exit 1
fi

# 运行第三个 Python 脚本
python 8_metric.py --data_path "$DIALOGUE_INPUT_PATH" --llm_evaluation_path "$EVAL_OUTPUT_PATH"

echo "All tasks completed successfully with input type: $INPUT_TYPE"
