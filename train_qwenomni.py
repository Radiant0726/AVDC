# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 33931))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

import os
os.environ["http_proxy"] = "http://10.20.112.35:3143"
os.environ["https_proxy"] = "http://10.20.112.35:3143"

import warnings
warnings.filterwarnings("ignore")
# warnings.filterwarnings("ignore", category=UserWarning)
# warnings.filterwarnings("ignore", category=FutureWarning)
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import AdamW
import math
# import deepspeed.comm as dist
import torch.distributed as dist
# print("PyTorch sees {} GPU(s)".format(torch.cuda.device_count()))
# for i in range(torch.cuda.device_count()):
#     print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

import logging
import pathlib
import sys
import json
from typing import Dict
import shutil
from pathlib import Path

base_dir = Path(__file__).resolve().parent
sys.path.insert(0, base_dir) 
sys.path.insert(0, str((base_dir / "transformers/src").resolve()))

import transformers
from transformers.trainer import *

import swanlab
from swanlab.integration.transformers import SwanLabCallback

# from qwenvl.data.data_qwen import make_supervised_data_module
# from qwenvl.data.data_qwen_packed import make_supervised_data_module_packed

from transformers import (
    Trainer,
    Qwen2_5OmniProcessor_FT,
    Qwen2_5OmniThinkerForConditionalGeneration_FT,
    Qwen2_5OmniThinkerForConditionalGeneration_VCen_FT,
    AutoTokenizer,
)
from torch.utils.data import IterableDataset
from transformers.trainer_pt_utils import IterableDatasetShard
from transformers.trainer_utils import get_last_checkpoint

from data_utils.data_load import *
from model_utils.create_model import *
from eval_utils.metrics import *

from train_utils.trainer import *
from train_utils.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)

local_rank = None

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

class ShuffleSwanLabCallback(SwanLabCallback):
    def on_epoch_begin(self, args, state, control, train_dataloader, **kwargs):
        if isinstance(train_dataloader.dataset, IterableDatasetShard):
            pass  # set_epoch() is handled by the Trainer
        elif isinstance(train_dataloader.dataset, IterableDataset):
            train_dataloader.dataset.set_epoch(train_dataloader.dataset.epoch + 1)   

def main():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # print(model_args, data_args, training_args)

    train_phase = model_args.train_phase
    train_mode = model_args.train_mode
    lr = training_args.learning_rate
    
    # print(model_args.model_name_or_path)

    run_name = train_phase
    run_name = f"{run_name}_{lr}"

    data_dir_name = train_phase #  if train_phase!="ft_instruct" else f"{train_phase}_clean"
    # print("data_dir_name:", data_dir_name)

    meta_dir = data_args.meta_dir
    data_args.dataset_use = {
            "train": f"{meta_dir}/{data_dir_name}/train.json",
            "val": f"{meta_dir}/{data_dir_name}/val.json",
            "test": f"{meta_dir}/{data_dir_name}/test.json",
        }
    data_args.data_cache_dir = data_args.data_cache_dir + f"_{data_dir_name}"
    os.makedirs(data_args.data_cache_dir, exist_ok=True)

    print("Setup Processor")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Omni-7B", use_fast=False, trust_remote_code=True)
    tokenizer.padding_side = "left" 

    max_pixels, min_pixels = data_args.max_pixels, data_args.min_pixels
    max_n_samples = data_args.max_n_samples
    processor = Qwen2_5OmniProcessor_FT.from_pretrained(processor_path)
    processor.video_processor.max_pixels = max_pixels
    processor.video_processor.min_pixels = min_pixels
    processor.feature_extractor.n_samples = max_n_samples
    # processor.save_pretrained(training_args.output_dir) # save

    print("Setup Data")
    # training_args.do_eval = False
    do_eval = training_args.do_eval
    # print("do_eval:", do_eval)
    dataset = create_dataset(data_args, processor, tokenizer, do_eval)
    train_dataset, val_dataset, test_dataset = dataset["train"], dataset["val"], dataset["test"]
    train_num = dataset["train_num"]
    run_name = f"{run_name}_data-{train_num}"
    local_rank = training_args.local_rank

    # save pre epoch 
    save_steps = int(train_num / (training_args.gradient_accumulation_steps * 4))
    training_args.save_steps = save_steps if save_steps < training_args.save_steps else training_args.save_steps
    print("save_steps:", training_args.save_steps)

    # adjust max_steps
    max_steps = int(save_steps * training_args.num_train_epochs)
    training_args.max_steps = max_steps
    print("max_steps:", max_steps)

    # training_args update
    training_args.run_name = run_name
    root_output_dir = training_args.output_dir
    training_args.output_dir = f"{training_args.output_dir}/{run_name}/model"

    print("Setup Model")
    model_path = model_args.model_name_or_path
    if isinstance(training_args.resume_training, bool) and training_args.resume_training:
        if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
            model_path = get_last_checkpoint(training_args.output_dir)
            warmup_steps  = training_args.warmup_steps
            checkpoint_name = model_path.split("/")[-1]
            resume_step = int(checkpoint_name.split("-")[-1])
            training_args.output_dir = training_args.output_dir.replace(run_name, run_name + f"_resume-{resume_step}")
            run_name = run_name + f"_resume-{resume_step}"
            training_args.run_name = run_name

            # adjust lr
            progress = float(resume_step - warmup_steps + 1) / float(max(1, training_args.max_steps - warmup_steps))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            scaled_init_lr = training_args.min_lr + (training_args.learning_rate - training_args.min_lr) * cosine_decay
            training_args.learning_rate = scaled_init_lr
            training_args.max_steps = training_args.max_steps - resume_step 
            training_args.warmup_steps = 0

            print(f"checkpoint {model_path} found, resume training!")
        else:
            print("train from scratch!")
    else:
        print("train from scratch!")

    os.makedirs(f"{training_args.output_dir}",exist_ok=True)

    # tokenizer.save_pretrained(training_args.output_dir) # save
    # processor.save_pretrained(training_args.output_dir) # save

    model = Qwen2_5OmniThinkerForConditionalGeneration_VCen_FT.from_pretrained(
        model_path, 
        attn_implementation="flash_attention_2",
        torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
    )
    model.enable_input_require_grads() 
    model.config.use_cache = False
            
    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    model = preprocess_model(model, model_args)
    # set_model(model_args, model)

    if torch.distributed.get_rank() == 0:
        print_trainable_parameters(model)

    swanlab_callback = ShuffleSwanLabCallback( # SwanLabCallback
        project="qwen2.5_omni_ft",
        experiment_name=run_name,
        config={
            "model_path": model_path,
            "train_num": train_num,
            # "lora_rank": 64,
            # "lora_alpha": 16,
            # "lora_dropout": 0.1,
        },
    )

    trainer = CustomTrainer(
        model=model, 
        processor=processor,
        processing_class=tokenizer, 
        args=training_args, 
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForSeq2Seq_videos(tokenizer=tokenizer, padding=True),
        compute_metrics=compute_metrics_save(run_name, root_output_dir),
        # preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[swanlab_callback],
        min_lr=training_args.min_lr
    )

    # if training_args.resume_training:
    #     try:
    #         if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
    #             print("checkpoint found, resume training!")
    #             trainer.train(resume_from_checkpoint=True)
    #     except:
    #         trainer.train()
    #         print("checkpoint not found, train from scratch!")
    # else:
    #     trainer.train()

    trainer.train()
    trainer.save_state()
    # trainer.predict(test_dataset)
    
    model.config.use_cache = True

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    main()
