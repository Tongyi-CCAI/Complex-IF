感谢您的提醒！下面是经过进一步精简和优化的版本，去掉了一些多余的图标，同时确保仍然保持视觉吸引力和简洁性。

---

<div align="center">

# **Complex Instruction Following (IF) for Deep Analysis**  
### Built by Tongyi Lab, Alibaba Group  

<img src="./figs/tongyi.png" width="50px" style="margin-top:10px;">

---

**IOPO: Empowering LLMs with Complex Instruction Following via Input-Output Preference Optimization**

</div>

---

## 📜 Abstract
In the realm of large language models (LLMs), the ability of models to accurately follow instructions is paramount as more agents and applications leverage LLMs, where the complexity of instructions is rapidly increasing. 

However:  
- There’s limited complex instruction evaluation data.  
- Few algorithms are tailored to improve these abilities.

This paper introduces:  
1. **TRACE**, a benchmark for evaluating and improving complex instruction-following, containing **120K training data** and **1K evaluation data**.  
2. **IOPO** (**Input-Output Preference Optimization**), a method that:  
   - Rapidly aligns responses with user preferences.  
   - Explores instructional requirements thoroughly.

### **Key Results**
- In **in-domain datasets**:  
  **+8.15%** (vs SFT), **+2.18%** (vs DPO).  
- In **out-of-domain datasets**:  
  **+6.29%** (vs SFT), **+3.13%** (vs DPO).  

---

<div align="center">

## 🔬 Comparison Results  
<img src="figs/intro.png" width="500" alt="Comparison Chart">

</div>

---

## 📊 TRACE Benchmark
- **Training Instructions**: 119,345  
- **Evaluation Instructions**: 1,042  

**Constraints per Instruction:**  
- Minimum: **1**, Maximum: **15**  
- Average: **4.36** (training), **4.89** (evaluation)  

<div align="center">

<img src="figs/trace_test_constraint_type.png" width="400" alt="TRACE Benchmark Statistics">

</div>

---

## ⚙ How to Run

### ➡ Step 1: Install Dependencies
```bash
cd Method-IOPO/
pip install -e ".[torch,metrics]"
```

### ➡ Step 2: Train the Model
```bash
llamafactory-cli train examples/qwen2_lora_iopo.yaml
```

---

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
cd Eval_script/
bash evaluate_all_task_for_model.sh <eval_results_output_path> \
    "models/vllm_qwen2_7b_trace_iopo.yaml" \
    "config/evaluator-trace-gpt-gpt.yaml"
```

---

## 📄 License

The content of this project is licensed under the [LICENSE](LICENSE).  

---

<div align="center">

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

</div>

---

<div align="center">

## ⭐ Star History

![Star History Chart](https://api.star-history.com/svg?repos=Tongyi-CCAI/Complex-IF&type=Date)

</div>

---

### 🎉 Thank you for your interest in **Complex IF**!  
