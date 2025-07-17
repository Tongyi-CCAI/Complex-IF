<div align="center">

<h2>Complex IF for Deep Analysis built by Tongyi Lab, Alibaba Group <img src="./figs/tongyi.png" width="30px" style="display:inline;"></h2>

</div>

### IOPO: Empowering LLMs with Complex Instruction Following via Input-Output Preference Optimization
 
### Abstract
In the realm of large language models (LLMs), the ability of models to accurately follow instructions is paramount as more agents and applications leverage LLMs for construction, where the complexity of instructions are rapidly increasing. However, on the one hand, there is only a certain amount of complex instruction evaluation data; on the other hand, there are no dedicated algorithms to improve the ability to follow complex instructions. To this end, this paper introduces TRACE, a benchmark for improving and evaluating the complex instructionfollowing ability, which consists of 120K training data and 1K evaluation data. Furthermore, we propose IOPO (Input-Output Preference Optimization) alignment method which takes both input and output preference pairs into consideration, where LLMs not only rapidly align with response preferences but also meticulously explore the instruction preferences. Extensive experiments on both in-domain and outof-domain datasets confirm the effectiveness of IOPO, showing 8.15%, 2.18% improvements on in-domain data and 6.29%, 3.13% on outof-domain data compared to SFT and DPO respectively.

### Comparison

<img src="figs/intro.png" width="500">

### TRACE Benchmark
TRACE consists of 119,345 instructions for model training, and 1,042 instructions for evaluation, where the minimum and maximum number of constraints per instruction are 1 and 15, with average numbers of 4.36 and 4.89, respectively.

<img src="figs/trace_test_constraint_type.png" width="300">

### How to Run
```
cd Method-IOPO/
pip install -e ".[torch,metrics]"
llamafactory-cli train examples/qwen2_lora_iopo.yaml
```

### How to evaluate
First, launch vllm for deploying the trained model:
```
python -m vllm.entrypoints.openai.api_server --served-model-name qwen2_7b_trace_iopo --model <trained_model_saved_path> --tensor-parallel-size 4
```
Then, run the evaluation script:
```
cd Eval_script/
bash evaluate_all_task_for_model.sh <eval_results_output_path> "models/vllm_qwen2_7b_trace_iopo.yaml" "config/evaluator-trace-gpt-gpt.yaml"
```

### License

The content of this project itself is licensed under [LICENSE](LICENSE).

### Citation

If this work is helpful, please kindly cite as:

```bigquery
@inproceedings{zhang-etal-2025-iopo,
    title = "IOPO: Empowering LLMs with Complex Instruction Following via Input-Output Preference Optimization",
    author = "Xinghua Zhang, Haiyang Yu, Cheng Fu, Fei Huang, Yongbin Li",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (ACL 2025)",
    month = July,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics"
}
```

### ✨Star History

<div align="center">

[![Star History Chart](https://api.star-history.com/svg?repos=Tongyi-CCAI/Complex-IF&type=Date)](https://www.star-history.com/#Tongyi-CCAI/Complex-IF&Date)

</div>
