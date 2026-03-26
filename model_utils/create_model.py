import os
import json
from pathlib import Path
curr_dir = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(Path(curr_dir))) 
# sys.path.insert(0, str(Path("/home/kaiyingyan/qwen2.5_omni_ft/transformers/src")))
# sys.path.insert(0, curr_dir) 
# sys.path.insert(0, f"/home/kaiyingyan/qwen_omni_finetune/transformers/src")
import argparse
import random
import torch
import torch.nn.functional as F
from accelerate import Accelerator

from datasets import Dataset, load_dataset
# from modelscope import snapshot_download, AutoTokenizer
import transformers
from transformers import (
    Trainer,
    Qwen2_5OmniProcessor_FT,
    Qwen2_5OmniThinkerForConditionalGeneration_FT,
    AutoTokenizer,
    PretrainedConfig
)
from peft import LoraConfig, TaskType, get_peft_model, PeftModel

def set_model(model_args, model):
    # tune_all
    if model_args.tune_all:
        for n, p in model.named_parameters():
            p.requires_grad = True
        return model
    
    # not tune_all
    for n, p in model.named_parameters():
        p.requires_grad = False

    # tune_vision
    if model_args.tune_vision:
        for n, p in model.visual.named_parameters():
            p.requires_grad = True

    # tune_vision_conn
    if model_args.tune_vision_conn:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = True

    # tune_audio
    if model_args.tune_audio:
        for n, p in model.audio_tower.named_parameters():
            p.requires_grad = True

    # tune_audio_conn
    if model_args.tune_audio_conn:
        for n, p in model.audio_tower.proj.named_parameters():
            p.requires_grad = True

    if model_args.tune_llm:
        for n, p in model.model.named_parameters():
            p.requires_grad = True
        for n, p in model.lm_head.named_parameters():
            p.requires_grad = True

    print("training:")
    for n, p in model.named_parameters():
            if p.requires_grad == True:
                print(n)
                
    return model

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        num_params = param.numel()
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            # print(name)
            trainable_params += num_params
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")


# 定义递归函数来设置 dropout 参数
def set_dropout_recursively(config, value):
    for attr_name in dir(config):
        if not attr_name.startswith("__"):  # 跳过内置属性
            attr = getattr(config, attr_name)
            if isinstance(attr, PretrainedConfig):  # 如果是嵌套的配置对象
                # print(attr)
                set_dropout_recursively(attr, value)
            elif 'dropout' in attr_name:  # 如果属性名包含 'dropout'
                setattr(config, attr_name, value)
                print(f"Set {attr_name} to {value}")

def preprocess_model(model, model_args):
    
    train_phase = model_args.train_phase

    if train_phase=="ft_audio":
        # model_args.tune_vision_conn = True
        model_args.tune_audio = True
        model_args.tune_audio_conn = True
    elif train_phase=="ft_omni":
        # model_args.tune_all=True
        model_args.tune_vision = True
        model_args.tune_vision_conn = True
        model_args.tune_audio = True
        model_args.tune_audio_conn = True
        # model_args.tune_llm = False
    elif train_phase=="ft_instruct" or "qa" in train_phase:
        model_args.tune_all = True
        # model_args.tune_vision_conn = True
        # model_args.tune_audio_conn = True
        # model_args.tune_llm = True

    model = set_model(model_args, model)
    print_trainable_parameters(model)

    # for name, module in model.named_modules():
    #     if isinstance(module, torch.nn.Linear):
    #         print(name)
    # 配置LoRA
    train_mode = model_args.train_mode
    if train_mode=='lora':
        lora_rank = model_args.lora_rank
        lora_alpha = model_args.lora_alpha
        lora_dropout = model_args.lora_dropout
        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=["qkv", "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            inference_mode=False,  # 训练模式
            r=lora_rank,  # Lora 秩 (越大表示可学习参数越多，但计算越重；常见值：8~128)
            lora_alpha=lora_alpha,  # Lora alaph，调节 LoRA 输出的强度 8 ~ 128（常见：16、32、64）
            lora_dropout=lora_dropout,  # Dropout 比例
            bias="none", # 是否训练 bias 参数；可选："none"、"all"、"lora_only" 一般设为 "none"，只训练 LoRA 层而非原始 bias
        )
        # 获取LoRA模型
        model = get_peft_model(model, config)

    else:
        # set dropout
        dropout = model_args.dropout
        set_dropout_recursively(model.config, dropout)
    return model

# 缩放因子 = lora_alpha / r ，调节 LoRA 注入更新量的强弱
# 常用组合示例
    # r	 lora_alpha	缩放因子 alpha/r
    # 8	   16	2.0
    # 16   32	2.0
    # 64   16	0.25
    # 64   64	1.0
    # 64   128	2.0