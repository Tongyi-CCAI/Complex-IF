<div align="center">

## **Complex Instruction Following (IF) & Reasoning for Deep Analysis**  

### Built by Tongyi Lab, Alibaba Group <img src="../figs/tongyi.png" width="25px" style="margin-top:10px;">

</div>

## 📊 TRACE Benchmark

- **Training Instructions**: 119,345
- **Evaluation Instructions**: 1,042

**Constraints per Instruction:**  
- Minimum: **1**, Maximum: **15**  
- Average: **4.36** (training), **4.89** (evaluation)  

<img src="../figs/trace_test_constraint_type.png" width="300" alt="TRACE Benchmark Statistics">

## 🧪 How to Evaluate

### ➡ Step 1: Launch `vllm` to Deploy the Trained Model

```bash
python -m vllm.entrypoints.openai.api_server \
    --served-model-name qwen2_7b_trace_iopo \
    --model <trained_model_saved_path> \
    --tensor-parallel-size 4
```

### ➡ Step 2: Run Evaluation Script

```bash
cd TRACE&IOPO/Eval_script/
bash evaluate_all_task_for_model.sh <eval_results_output_path> \
    "models/vllm_qwen2_7b_trace_iopo.yaml" \
    "config/evaluator-trace-gpt-gpt.yaml"
```

## 💬 Citation

If this work is helpful, please cite as:

```bibtex
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
