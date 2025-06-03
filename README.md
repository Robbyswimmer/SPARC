# SPARC

SPARC: Streaming Pruning And Retention Controller

SPARC is a lightweight reinforcement learning (RL) framework for dynamic, token-budgeted context management in frozen large language models (LLMs). Instead of using static heuristics like sliding windows or TF-IDF filtering, SPARC learns to decide in real time which incoming document chunks to KEEP, COMPRESS, or DROP, optimizing for downstream task accuracy under a fixed context window constraint.

🚀 Motivation

LLMs are increasingly used in streaming applications—code assistants, document QA systems, live meeting agents—but are limited by their context window (e.g., 2K-8K tokens). Most current systems use simple heuristics that fail to adapt based on content relevance or task reward. SPARC introduces a trained policy that performs real-time memory management based on context utility and budget trade-offs.

🎯 Features

RL environment StreamingQAGym based on Gymnasium

Support for KEEP, DROP, and (soon) COMPRESS actions

PPO-based training with Stable-Baselines3

Pluggable reward function combining QA accuracy (EM + F1) and token cost

Frozen LLM inference via llama-cpp-python with 4-bit GGUF models

Fully streamable HF datasets (e.g., NarrativeQA, HotpotQA)

Lightweight summariser for COMPRESS action (WIP)

End-to-end train/evaluate scripts and WandB logging

📁 Project Structure

SPARC/
├── src/
│   ├── envs/              # streaming_qagym.py
│   ├── agents/            # PPO policy + summariser
│   ├── utils/             # reward, tokenizer, LLM interface
│   └── scripts/           # train.py, evaluate.py, demo
├── tests/                 # unit tests
├── data/                  # streamed HF datasets
├── environment.yml        # conda environment file
├── requirements.txt       # pip fallback
└── README.md

⚙️ Setup

# Clone and activate conda environment
- conda create -n streamqa python=3.11 -y
- conda activate streamqa
- pip install -r requirements.txt

Optional: install llama-cpp-python for quantized LLM inference
- brew install cmake
- CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python

🧪 Quickstart

python src/scripts/example_demo.py   # random agent demo
python src/scripts/train.py          # PPO training loop
python src/scripts/evaluate.py       # accuracy, EM, token usage

🧠 Key Concepts

Observation: CLS embeddings + remaining budget + chunk index

Actions: {0: DROP, 1: KEEP, 2: COMPRESS (WIP)}

Reward: QA_EM + QA_F1 − token_penalty − step_penalty − compress_penalty

Policy: 3-layer Transformer (<12M params)

🤝 Contributing

We welcome PRs that improve training stability, extend evaluation tasks, or add summarisation capabilities. Please lint and test your code and follow the contribution guidelines in .windsurfrules.

🧭 Roadmap
