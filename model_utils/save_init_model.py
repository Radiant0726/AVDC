import os
from pathlib import Path
curr_dir = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(Path(curr_dir))) 
sys.path.insert(0, str(Path("/home/kaiyingyan/qwen_omni_finetune/transformers/src")))
# sys.path.insert(0, curr_dir) 
# sys.path.insert(0, f"/home/kaiyingyan/qwen_omni_finetune/transformers/src")
import argparse
import random
import torch
import torch.nn.functional as F

from datasets import Dataset, load_dataset
# from modelscope import snapshot_download, AutoTokenizer
import transformers
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLProcessor,
    Qwen_omniProcessor,
    Qwen_omni_ForConditionalGeneration,
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
)

def init_weights(module,std=0.02):
    for name, param in module.named_parameters():
        if 'weight' in name:
            torch.nn.init.normal_(param, mean=0.0, std=std)  # 均值为 0，标准差为 0.02
        elif 'bias' in name:
            torch.nn.init.constant_(param, 0.0)  # 偏置初始化为 0

def save_init(model, processor, tokenizer, save_path):
    processor.save_pretrained(save_path)  # 保存 processor
    tokenizer.save_pretrained(save_path)  # 保存 tokenizer
    model.save_pretrained(
        save_path,
        safe_serialization=True,
        max_shard_size="4GB"
    )

model_path = "/remote-home/kaiyingyan/models/Qwen2.5-VL-7B-Instruct" # model path 
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
tokenizer.padding_side = "left" # 自回归 pad left
processor = Qwen_omniProcessor.from_pretrained(model_path)

model = Qwen_omni_ForConditionalGeneration.from_pretrained(
    model_path, 
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    device_map="auto", 
    max_memory={0: "80GiB",1: "80GiB",2: "80GiB",3: "80GiB"} #,4: "15GiB",5: "15GiB",6: "15GiB",7: "15GiB"
)#.cuda()

audio_encoder_path = "/remote-home/kaiyingyan/models/audio_encoder.pth"
pretrained_dict = torch.load(audio_encoder_path)['audio_tower']

# # 过滤掉形状不匹配的参数
# model_dict = model.audio_tower.state_dict()
# matched_pretrained_dict = { 
#     k: v for k, v in pretrained_dict.items()
#     if k in model_dict and v.shape == model_dict[k].shape
# }
# model_dict.update(matched_pretrained_dict)
# model.audio_tower.load_state_dict(model_dict, strict=True)

model.audio_tower.load_state_dict(pretrained_dict, strict=True)
init_weights(model.audio_connector)

INIT_DIR = f"/remote-home/kaiyingyan/qwen_omni/outputs/init_omni_model"
save_init(model, processor, tokenizer, INIT_DIR)

print("saved:",INIT_DIR)