# NextMem: Towards Latent Factual Memory of LLM-based Agents

## 📝 Introduction

Memory is critical for LLM-based agents to preserve previous observations for future decision-making, while factual memory serves as its foundation. However, existing methods face several limitations. Textual memory imposes heavy context and indexing burdens, while parametric memory suffers from catastrophic forgetting and high costs. To address these challenges, we introduce NextMem, a latent factual memory framework that utilizes an autoregressive autoencoder to efficiently construct latent memory while ensuring accurate reconstruction. We propose a two-stage training process, including autoregressive reconstruction alignment and progressive latent substitution. In addition, we incorporate quantization to reduce storage overhead. Extensive experiments demonstrate that NextMem achieves superior performance, and excels in retrieval, robustness, and extensibility properties.

## 📌 Major Contributions

![methods](assets\methods.png)

- We introduce a simple yet effective framework for latent factual memory, with autoregressive reconstruction alignment and progressive latent substitution.
- We integrate quantization methods into the latent memory of our framework, which reduces storage overhead while maintaining competitive performance.
- We validate our approach through extensive experiments and provide further insights. We provide the research community with open-source code and model checkpoints.

## 🚀 Run Experiments

### 💻1 Prepare for the environment.

Create an environment.

```
conda create -n nextmem_env python=3.11.13
```

Activate the environment.

```
conda activate nextmem_env
```

Install the packages.

```
pip install -r requirements.txt
```

Download datasets and checkpoints from [Google Drive](https://drive.google.com/drive/folders/1QK0MM4ySe30oAAJct7LDJcFy5yZjO6Wr?usp=sharing), and put them under the root dictionary `nextmem`. If you want to try other baselines, please download as follows:

```
DeepSeek-OCR: https://huggingface.co/deepseek-ai/DeepSeek-OCR
Mistral-7B-v0.2: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2
Llama3-8B: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
Llama3.1-8B: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
```

Please note that we modify the model of Deepseek-OCR to split its encoding and decoding process (see `models/modified_model_filesmodeling_deepseekocr.py`). 

Download Qwen3-8B backbones from [Hugging face](https://huggingface.co/Qwen/Qwen3-8B).

### 📝 2 Config the experiments.

Fill all the paths required in `config.py` accordingly.

### 📋 3 Run Experiments.

#### 3.1 Run the encoding-decoding examples.

Run the following commands.

```
python example.py
```

#### 3.2 Run the factual reconstruction task.

Run the following commands.

```
cd experiments/task1
nohup python -u main.py NextMemQwen >> NextMemQwen.log 2>&1 &
```

Then, the results will be stored under `task1/results`. You can also change `NextMemQwen` as other models shown in `config.py`.

#### 3.3 Run the contextual generation task.

Run the following commands.

```
cd experiments/task2
nohup python -u main.py NextMemQwen >> NextMemQwen.log 2>&1 &
```

Then, the results will be stored under `task2/results`. You can also change `NextMemQwen` as other models shown in `config.py`.

#### 3.4 Run the dense passage retrieval task.

Run the following commands.

```
cd experiments/task3
nohup python -u main.py NextMemQwen >> NextMemQwen.log 2>&1 &
```

Then, the results will be stored under `task3/results`. You can also change `NextMemQwen` as other models shown in `config.py`.

## 🔗 Citation

```
[To Be Updated]
```
