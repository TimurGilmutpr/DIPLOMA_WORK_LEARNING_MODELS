import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from trl import SFTTrainer, SFTConfig
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import SFTTrainer
from transformers import TrainingArguments
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
from huggingface_hub import login
HF_TOKEN = "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"
login(HF_TOKEN, add_to_git_credential=False)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)


model_name = "meta-llama/Llama-3.1-8B-Instruct" 

