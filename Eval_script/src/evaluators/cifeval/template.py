import json
class CifevalTemplate:
    @staticmethod
    def generate_verdicts(question, gold_answer, pred_answer, constraints=None, description=None):

        format_dict = [{"密码至少8个字符": 10, "包含1个大写字母": 0, "包含1个小写字母": 0, "包含2个数字": 10}]
        format_str = json.dumps(format_dict, ensure_ascii=False)
        PROMPT_TEMPLATE_EVAL_TRACE = '''[System]
You are a fair judge, and please evaluate the quality of an AI assistant’s response to user query. You need to assess the response based on the following constraints. 
We will provide you with the user’s query, some constraints, and the AI assistant’s response that needs your evaluation. When you commence your evaluation, you should follow the following process:
1. Evaluate the AI assistant’s response on different constraints, and after each constraint evaluation, assign a score from 0 to 10.
2. Finally, aggregate the assessments from each  constraint to give an overall score for the AI assistant’s response, ranging from 0 to 10.
3. Your scoring should be as strict as possible, overall, the higher the quality of the model’s response, the higher the score.
4. When the model’s response is irrelevant to the question, contains significant factual errors, or generates harmful content, the dimension total score must be 0 points.
5. It is necessary to strictly follow the format in the / *Example * / for generation, the formats for Fine Grained Score is Json, the formats Constraints Overall Score is [[d]].
Please remember to provide evaluations and explanations before your scoring. After your explanation of each constraint, include a score for that constraint. Then, output Fine Grained Score. Finally, output overall score in format like "Constraints Overall Score: [[5]]"

/ *Example * /
—INPUT—
< query >:
为该账户创建一个密码
< constraints >:
1. 密码至少8个字符；
2. 包含1个大写字母；
3. 包含1个小写字母；
4. 包含2个数字；
< response >:
Ax7y4gTf

—OUTPUT—
Explanation: 
1. 密码长度：密码长度为8个字符，符合第1条约束，得分10分。
2. 包含一个大写字母：密码“Ax7y4gTf”里包含“A”和“T”两个大写字母，不符合第2条约束，得分0分。
3. 包含一个小写字母：密码“Ax7y4gTf”里包含“x”、“y”和“g”三个小写字母，不符合第3条约束，得分0分。
4. 包含两个数字：密码“Ax7y4gTf”里包含“7”和“4”两个数字，符合第4条约束，得分10分。
Fine Grained Score: {format_str}
Constraints Overall Score: [[5]]

/ *Input * /
—INPUT—
< query >:
{description}
< constraints >:
{constraints}
< response >:
{ans}

—OUTPUT—'''
        if "NULL" not in question:
            # description = description + '\n' + question
            description = question

        ans = pred_answer
        prompt = PROMPT_TEMPLATE_EVAL_TRACE.format(format_str=format_str, description=description, constraints=constraints, ans=ans)
        return prompt


