from unsloth import FastLanguageModel
import torch

# Load model - 4bit quantization for M2
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3.2-3b-bnb-4bit",
    max_seq_length = 2048,
    dtype = None,  # Auto-detect for M2
    load_in_4bit = True,
)

# Add LoRA adapters (only trains 1-2% of parameters)
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
)

# Your training data format
from datasets import load_dataset

# dataset = load_dataset("json", data_files="your_data.jsonl")
dataset = load_dataset("vicgalle/alpaca-gpt4", split="train")

# Train
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset["train"],
    dataset_text_field = "text",
    max_seq_length = 2048,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_available(),
        bf16 = torch.cuda.is_available(),
        logging_steps = 1,
        output_dir = "outputs",
    ),
)

trainer.train()

# Save
model.save_pretrained("lora_model")