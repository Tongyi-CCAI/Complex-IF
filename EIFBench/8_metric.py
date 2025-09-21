import json
import argparse
import os
from copy import deepcopy
# import matplotlib.pyplot as plt
from collections import defaultdict

def load_jsonl(file_path):
    _data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for data in f:
            jline = json.loads(data)
            _data.append(jline)
    return _data

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--llm_evaluation_path", type=str, default="")
    args = parser.parse_args()

    raw_data = read_file(args.data_path)
    raw_data_ids = [item['main_id'] for item in raw_data]
    outputs = {}
    point_judges = {}
    point_explanation = {}
    evaluate = {}
    evaluate_subtask = {}
    
    llm_evaluations = load_jsonl(args.llm_evaluation_path)
    print("length of llm_evaluations:", len(llm_evaluations))
    for i, l in enumerate(llm_evaluations):
        main_id = l['main_id']
        if main_id not in raw_data_ids:  # 只处理300个ID的数据
            print("none")
            continue
        if l['point_explanation'] in ['None', 'Err']:
            print("nothing")
            continue
        sub_task = l['sub_instruction_id']
        judge_ans = l['point_judge']
        point_id = l['point_id']
        question = l['question']
        outputs[main_id] = l['output']
        point_judges[(main_id, sub_task, question)] = judge_ans
        point_explanation[(main_id, sub_task, question)] = l['point_explanation']
    print("length of points are:{}".format(len(point_explanation)))

    for item_key, item_score in point_judges.items():
        main_id, sub_task, point_id = item_key
        if main_id not in evaluate:
            evaluate[main_id] = []
        if main_id not in evaluate_subtask:
            evaluate_subtask[main_id] = {}
        if sub_task not in evaluate_subtask[main_id]:
            evaluate_subtask[main_id][sub_task] = []

        evaluate[main_id].append(item_score)
        evaluate_subtask[main_id][sub_task].append(item_score)

    # 计算 sub_task 数量和对应的平均分
    subtask_count_to_scores = defaultdict(list)
    for main_id, scores in evaluate.items():
        avg_score = []
        for sub_task, scores in evaluate_subtask[main_id].items():
            avg_score.append(all(scores))
        subtask_count = len(evaluate_subtask[main_id])
        average_score = sum(avg_score) / len(avg_score)
        subtask_count_to_scores[subtask_count].append(average_score)
    
    # 计算各 sub_task 数量的平均评分
    subtask_count_average_scores = {count: sum(scores)/len(scores) for count, scores in subtask_count_to_scores.items()}

    # 初始化 main_id 数量的平均评分统计
    main_id_group_count_to_scores = defaultdict(list)

    for main_id, scores in evaluate.items():
        avg_score = []
        for sub_task, scores in evaluate_subtask[main_id].items():
            main_id_group_count_to_scores[len(scores)].append(all(scores))

    # 计算各 main_id 数量的平均评分
    main_id_count_average_scores = {count: sum(scores)/len(scores) for count, scores in main_id_group_count_to_scores.items()}

    # 对 subtask_count_average_scores 进行排序
    sorted_subtask_count = sorted(subtask_count_average_scores.items())
    subtask_counts, subtask_avg_scores = zip(*sorted_subtask_count)

    # 对 main_id_count_average_scores 进行排序
    sorted_main_id_count = sorted(main_id_count_average_scores.items())
    main_id_counts, main_id_avg_scores = zip(*sorted_main_id_count)

    result_dict = {}
    count = 0
    true_count, false_count = 0, 0
    constraints = []
    for id, id_list in evaluate.items():
        constraints.append(sum(id_list) / len(id_list))
        flag = all(id_list)
        if flag:
            true_count += 1
        else:
            false_count += 1
        count += 1

    print("total tasks are:{}, hard accuracy for task level are:{}".format(count, true_count / count))
    result_dict["task level"] = true_count / count
    result_dict["constraint level"] = sum(constraints) / len(constraints)

    count = 0
    true_count, false_count = 0, 0
    # total_cons = []
    soft_subtask_cons = []
    for id, sub_tasks in evaluate_subtask.items():
        sub_values = []
        for j, sub_task in sub_tasks.items():
            flag = all(sub_task)
            # total_cons.extend(sub_task)
            if flag:
                true_count += 1
                sub_values.append(1)
            else:
                false_count += 1
                sub_values.append(0)
            count += 1
        # print(sum(sub_task), len(sub_task))
        soft_subtask_cons.append(sum(sub_values)/len(sub_values))
    result_dict["sub-task level"] = sum(soft_subtask_cons) / len(soft_subtask_cons)

    print("total sub-task are:{}, hard accuracy for sub-task level are:{}".format(len(soft_subtask_cons), sum(soft_subtask_cons) / len(soft_subtask_cons)))
    # print("{:.4f}".format(sum(soft_subtask_cons) / len(soft_subtask_cons)),end=" ")
    print("total constraints are:{}, hard accuracy for constraint level are:{}".format(len(constraints), result_dict["constraint level"]))
    # print("{:.4f}".format(result_dict["constraint level"]),end=" \n")



