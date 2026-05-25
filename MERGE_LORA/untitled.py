import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import gc


adapter_path = "meta-llama_Llama-3.1-8B-Instruct_lora_epochs_10_batch_3_optim_adamw_torch_fused_r_16_alpha_32_q_proj_v_proj_k_proj_o_proj_gate_proj_up_proj_down_proj_embed_tokens_lm_head"
adapter_path= "../meta-llama_Llama-3.1-8B-Instruct_lora_epochs_10_batch_4_optim_adamw_torch_fused_r_64_alpha_128_lr_0.0002_drop_out_0.05_q_proj_v_proj_k_proj_o_proj_gate_proj_up_proj_down_proj_embed_tokens_lm_head"
print(os.listdir(adapter_path))


base_model_name = "meta-llama/Llama-3.1-8B-Instruct"
print("Загрузка базовой модели...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token


# Загрузка адаптера поверх базовой модели
print("Загрузка адаптера LoRA...")
model = PeftModel.from_pretrained(base_model, adapter_path)


# Слияние адаптера с базовой моделью
print("Слияние адаптера...")
merged_model = model.merge_and_unload()


# Сохранение объединённой модели
output_dir = "LORA_THE_BEST"
print(f"Сохранение объединённой модели в {output_dir}...")
merged_model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)


