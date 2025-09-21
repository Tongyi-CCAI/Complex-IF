<div align="center">

## **Complex Instruction Following (IF) & Reasoning for Deep Analysis**  

### Built by Tongyi Lab, Alibaba Group <img src="./figs/tongyi.png" width="25px" style="margin-top:10px;">

</div>

## ⚙ How to Run

### ➡ Step 1: Install Dependencies

```bash
pip install -e ".[torch,metrics]"
```

### ➡ Step 2: Train the Model

```bash
llamafactory-cli train examples/qwen2_lora_iopo.yaml
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
