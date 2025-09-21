from __future__ import annotations

from typing import Callable, Tuple, Union, List
import re

from src.schema import BaseEvaluator, QARecord
from src.utils.registry import REGISTRY
from src.evaluators.trace.template import TraceTemplate
from src.schema.model import BaseLLM
from src.schema import (
    QAPrediction,
MultiturnQAPrediction,
    QARecord,
    BaseLLM,
    BaseLLMOutput
)

def trace_score(pred: QAPrediction, evaluator_llm: BaseLLM) -> QAPrediction:

    gold_answer = pred.answer
    question = pred.question
    pred_answer = pred.generated
    constraints = pred.constraints
    description = pred.description
    evaluator_prompt = TraceTemplate.generate_verdicts(
            question=question,
            gold_answer=gold_answer,
            pred_answer=pred_answer,
            constraints=constraints,
            description=description
    )
    # print(evaluator_prompt)
    evaluator_llm_output = evaluator_llm(evaluator_prompt)
    # print(evaluator_llm_output)

    matches = re.search(r"Constraints Overall Score:\s*\[\[([0-9.]+)\]\]", evaluator_llm_output.generated, re.DOTALL)

    try:
        rating = matches.group(1).strip()
        rating = float(rating)
        pred.matched = rating
        pred.evaluator_llm_output = evaluator_llm_output
        return pred
    except:
        print('error when extracting score')
        pred.matched = None
        pred.evaluator_llm_output = evaluator_llm_output
        # print(evaluator_llm_output.generated)
        return pred

@REGISTRY.register_evaluator("trace")
class TraceEvaluator(BaseEvaluator):
    matcher: Callable[[QAPrediction, BaseLLM], QAPrediction] = trace_score

