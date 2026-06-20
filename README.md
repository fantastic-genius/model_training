# Model Training & AI Agents

A collection of Python scripts for fine-tuning Large Language Models (LLMs) using LoRA and building AI agents with various capabilities including RAG, memory management, and tool integration.

## Overview

This project demonstrates:
- **LoRA Fine-tuning**: Efficient fine-tuning of LLaMA models on Apple Silicon (MPS) using PEFT (Parameter-Efficient Fine-Tuning)
- **AI Agents**: Multiple agent implementations with different capabilities (Ollama-based, RAG, memory persistence)
- **Model Inference**: Deployment and testing of fine-tuned models
- **Training Datasets**: AI/ML question-answer pairs for domain-specific fine-tuning

## Project Structure

```
model_training/
├── test_training/
│   ├── fine_tunning.py          # Basic LoRA fine-tuning script
│   ├── fine_tunning_2.py        # Advanced LoRA fine-tuning with MPS support
│   ├── ft_inference.py          # Inference script for fine-tuned models
│   ├── agent_ollama.py          # Simple agent with tool integration
│   ├── memory_agent.py          # Agent with persistent memory
│   ├── rag_agent.py             # RAG-enabled agent
│   ├── production_agent.py      # Production-ready agent implementation
│   ├── test_local.py            # Local testing utilities
│   ├── data.jsonl               # Training dataset (AI/ML Q&A)
│   ├── ft-lora-model/           # Fine-tuned LoRA adapters
│   └── .gitignore               # Git ignore patterns
└── README.md                    # This file
```

## Features

### LoRA Fine-Tuning
- **Efficient Training**: Uses LoRA (Low-Rank Adaptation) to train only ~1% of model parameters
- **Apple Silicon Support**: Optimized for M1/M2/M3 Macs with MPS (Metal Performance Shaders)
- **Model Support**: Works with LLaMA 3.2 models (3B parameters)
- **Flexible Configuration**: Customizable LoRA rank, alpha, dropout, and target modules

### AI Agents
- **Tool Integration**: Agents can execute Python code, run bash commands, and perform file operations
- **Memory Persistence**: Save and restore conversation context across sessions
- **RAG Support**: Retrieval-Augmented Generation for knowledge-enhanced responses
- **Ollama Integration**: Compatible with locally-hosted Ollama models

## Requirements

### System Requirements
- **Hardware**: Apple Silicon Mac (M1/M2/M3) or NVIDIA GPU
- **RAM**: 16GB+ recommended for 3B models
- **Storage**: 10GB+ for models and datasets

### Software Requirements
```bash
# Python packages
torch>=2.0.0
transformers>=4.35.0
peft>=0.7.0
datasets>=2.14.0
accelerate>=0.24.0
bitsandbytes>=0.41.0  # Optional, for quantization
ollama-python         # For Ollama-based agents

# For Ollama agents
# Install Ollama from https://ollama.ai
```

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/fantastic-genius/model_training.git
cd model_training/test_training
```

2. **Create a virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install torch transformers peft datasets accelerate
pip install ollama-python  # For agent scripts
```

4. **Set up Hugging Face authentication**
```bash
export HF_TOKEN="your_hugging_face_token_here"
```
Get your token from [Hugging Face Settings](https://huggingface.co/settings/tokens)

5. **Install Ollama (for agent scripts)**
```bash
# macOS
brew install ollama

# Or download from https://ollama.ai
ollama pull llama3.2:3b
```

## Usage

### Fine-Tuning a Model

**Basic fine-tuning:**
```bash
python fine_tunning_2.py
```

This script will:
1. Load the LLaMA 3.2-3B base model
2. Apply LoRA adapters to attention layers
3. Train on the AI/ML Q&A dataset
4. Save the fine-tuned adapters to `lora-output/`

**Configuration options** (edit in `fine_tunning_2.py`):
- `r`: LoRA rank (default: 16)
- `lora_alpha`: LoRA scaling factor (default: 32)
- `lora_dropout`: Dropout rate (default: 0.05)
- `learning_rate`: Training learning rate
- `num_train_epochs`: Number of training epochs

### Running Inference

**Test the fine-tuned model:**
```bash
python ft_inference.py
```

This loads the base model + LoRA adapters and generates responses.

### Using AI Agents

**Simple agent with tools:**
```bash
python agent_ollama.py
```

**Agent with memory:**
```bash
python memory_agent.py
```

**RAG agent:**
```bash
python rag_agent.py
```

**Production agent:**
```bash
python production_agent.py
```

## Training Data

The project includes `data.jsonl` with AI/ML question-answer pairs covering:
- Artificial Intelligence fundamentals
- Machine Learning concepts
- Deep Learning techniques
- Natural Language Processing (NLP)
- Computer Vision
- Neural networks
- Model training concepts (overfitting, underfitting)

**Format:**
```json
{"question": "What is Machine Learning?", "answer": "Machine Learning (ML) is..."}
```

## Fine-Tuned Model

The `ft-lora-model/` directory contains:
- `adapter_config.json`: LoRA configuration
- `adapter_model.safetensors`: Trained LoRA weights
- `tokenizer.json`: Model tokenizer
- `README.md`: Model card

**To use the fine-tuned model:**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B")
model = PeftModel.from_pretrained(base_model, "./ft-lora-model")
tokenizer = AutoTokenizer.from_pretrained("./ft-lora-model")
```

## Performance Tips

### Apple Silicon (MPS)
```python
# Enable MPS acceleration
device = "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)
```

### Memory Optimization
- Use `torch.float16` for reduced memory usage
- Adjust `per_device_train_batch_size` based on available RAM
- Use gradient accumulation for larger effective batch sizes

### Training Speed
- Use `fp16=True` in TrainingArguments
- Enable gradient checkpointing for larger models
- Adjust `num_train_epochs` and `max_steps`

## Common Issues

### Issue: "MPS backend not available"
**Solution:** Ensure you're on macOS 12.3+ with Apple Silicon (M1/M2/M3)

### Issue: Out of memory during training
**Solutions:**
- Reduce `per_device_train_batch_size`
- Use `gradient_accumulation_steps`
- Use smaller model (1B instead of 3B)
- Enable gradient checkpointing

### Issue: Hugging Face authentication failed
**Solution:** Set your token:
```bash
export HF_TOKEN="hf_..."
# Or login via CLI
huggingface-cli login
```

### Issue: Slow training on CPU
**Solution:** Training on CPU is very slow. Use MPS (Apple Silicon) or CUDA (NVIDIA GPU) for faster training.

## Architecture Details

### LoRA Configuration
```python
LoraConfig(
    r=16,                    # Rank: controls adapter size
    lora_alpha=32,           # Scaling factor
    target_modules=[         # Which layers to adapt
        "q_proj", "v_proj",  # Query/Value projections
        "k_proj", "o_proj"   # Key/Output projections
    ],
    lora_dropout=0.05,       # Regularization
    bias="none",             # No bias training
    task_type="CAUSAL_LM"    # Language modeling task
)
```

### Agent Capabilities
- **Tool Execution**: Python interpreter, bash commands, file I/O
- **Memory Management**: JSON-based conversation persistence
- **RAG Integration**: Vector-based knowledge retrieval
- **Streaming**: Real-time response generation

## Contributing

Contributions are welcome! Areas for improvement:
- Additional training datasets
- Support for more model architectures
- Enhanced agent capabilities
- Evaluation benchmarks
- Deployment examples

## License

This project is provided as-is for educational and research purposes.

## Acknowledgments

- **Hugging Face**: Transformers, PEFT, and Datasets libraries
- **Meta AI**: LLaMA model architecture
- **Ollama**: Local LLM hosting
- Built with Apple Silicon MPS acceleration

## Resources

- [Hugging Face PEFT Documentation](https://huggingface.co/docs/peft)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [LLaMA Models](https://huggingface.co/meta-llama)
- [Ollama Documentation](https://ollama.ai)

---

**Note:** This project requires a Hugging Face account and access to LLaMA models. Some models may require acceptance of terms on Hugging Face.
