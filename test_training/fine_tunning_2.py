
# Use native PyTorch with MPS backend (Metal Performance Shaders - Apple's GPU acceleration):
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import json
import os

HF_TOKEN = os.getenv("HF_TOKEN")  # Set your token as environment variable

# Check MPS is available
print(f"MPS available: {torch.backends.mps.is_available()}")
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Load base model
model_name = "meta-llama/Llama-3.2-3B"  # 2.7B - good for M2
# OR "meta-llama/Llama-3.2-3B" if you have access

tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # Use float16 for MPS
    device_map="auto",
    token=HF_TOKEN
)

# Configure LoRA
lora_config = LoraConfig(
    r=16,  # Rank
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Attention layers
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # Shows ~1% of params are trainable

# Prepare your dataset
def format_data(example):
    """Format your data as needed"""
    text = f"### Question: {example['question']}\n### Answer: {example['answer']}"
    return {"text": text}

# Load your data (create a sample JSONL file)
"""
Create data.jsonl with format:
{"question": "What is AI?", "answer": "AI is..."}
{"question": "How does ML work?", "answer": "ML works by..."}
"""

dataset = load_dataset("json", data_files="data.jsonl", split="train")
dataset = dataset.map(format_data)

# Tokenize
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length"
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

# Training arguments optimized for M2
training_args = TrainingArguments(
    output_dir="./lora-output",
    per_device_train_batch_size=1,  # Start with 1 on M2
    gradient_accumulation_steps=4,   # Effective batch size of 4
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=False,  # MPS doesn't support fp16 training well yet
    logging_steps=10,
    save_steps=50,
    save_total_limit=2,
    report_to="none",  # Disable wandb/tensorboard
    use_mps_device=True,  # Use Apple Silicon GPU
)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # We're doing causal LM, not masked LM
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# Train (this will take 20-60 minutes depending on dataset size)
trainer.train()

# Save the LoRA adapters
model.save_pretrained("./ft-lora-model")
tokenizer.save_pretrained("./ft-lora-model")

print("Training complete! Model saved to ./ft-lora-model")