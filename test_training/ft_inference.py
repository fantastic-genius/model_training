from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import os

device = "mps" if torch.backends.mps.is_available() else "cpu"


HF_TOKEN = os.getenv("HF_TOKEN")  # Set your token as environment variable
model_name = "meta-llama/Llama-3.2-3B"

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    token=HF_TOKEN
)

# Load LoRA adapters
model = PeftModel.from_pretrained(base_model, "./ft-lora-model")
tokenizer = AutoTokenizer.from_pretrained("./ft-lora-model")

# Generate
# prompt = "### Question: What is machine learning?\n### Answer:"
prompt = "### Question: Explain neural network and its application in computer vision"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

outputs = model.generate(
    **inputs,
    max_new_tokens=200,
    temperature=0.7,
    do_sample=True
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))