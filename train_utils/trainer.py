import os
# os.environ["http_proxy"] = "http://10.20.112.35:3143"
# os.environ["https_proxy"] = "http://10.20.112.35:3143"

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
sys.path.insert(0, "/mnt/vision_user/kaiyinyan/code/qwen2.5_omni_ft") 
sys.path.insert(0, "/mnt/vision_user/kaiyinyan/code/qwen2.5_omni_ft/transformers/src")


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

from .contra_av_loss import *
local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)

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

def get_cosine_schedule_with_warmup_and_min_lr(optimizer, num_warmup_steps, num_training_steps, min_lr, last_epoch=-1):
    initial_lr = optimizer.defaults['lr']
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # 线性 warmup，从 0 到 1
            return float(current_step + 1) / float(max(1, num_warmup_steps))
        else:
            # 余弦衰减从 1 到 min_lr / initial_lr
            progress = float(current_step - num_warmup_steps + 1) / float(max(1, num_training_steps - num_warmup_steps))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            # 缩放到 min_lr 的比例
            scaled_lr = cosine_decay * (1.0 - min_lr / initial_lr) + (min_lr / initial_lr)
            return scaled_lr
    return LambdaLR(optimizer, lr_lambda, last_epoch)


class DistMemoryBank:
    def __init__(self, feature_dim, bank_size=32, device="cuda"):
        self.bank_size = bank_size
        self.feature_dim = feature_dim
        self.device = device
        self.ptr = 0
        self.filled = 0  # 当前 bank 已填充的数量
        self.bank = torch.zeros(2, bank_size, feature_dim, device=device)

    @torch.no_grad()
    def update(self, features):
        """
        features: [2, B, D]
        """
        B = features.size(1)
        if B > self.bank_size:
            features = features[:, :self.bank_size, :]
            B = self.bank_size

        end = self.ptr + B
        if end <= self.bank_size:
            self.bank[:, self.ptr:end, :] = features
        else:
            first_len = self.bank_size - self.ptr
            self.bank[:, self.ptr:, :] = features[:, :first_len, :]
            self.bank[:, :end % self.bank_size, :] = features[:, first_len:, :]
        self.ptr = (self.ptr + B) % self.bank_size
        self.filled = min(self.bank_size, self.filled + B)  # 更新 filled

    @torch.no_grad()
    def get_memory(self):
        """分布式同步 memory bank"""
        if self.filled == 0:
            return None
        elif self.filled < self.bank_size:
            return self.bank[:, :self.filled, :]
            # return None
        return self.bank
    
    # @torch.no_grad()
    # def get_global(self):
    #     """分布式同步 memory bank"""
    #     if self.filled < self.bank_size:
    #         return None
    #     if not dist.is_initialized():
    #         return self.bank.clone().detach()
    #     world_size = dist.get_world_size()
    #     local_bank = self.bank.clone()
    #     global_bank_list = [torch.zeros_like(local_bank) for _ in range(world_size)]
    #     dist.all_gather(global_bank_list, local_bank)
    #     return torch.cat(global_bank_list, dim=1).detach()  # [2, bank_size*world_size, D]
    
class CustomTrainer(Trainer):
    def __init__(self, *args, min_lr=0, processor=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_lr = min_lr 
        print("min_lr:",min_lr)
        if processor: self.processor = processor

    def create_optimizer_and_scheduler(self, num_training_steps):
        self.optimizer = AdamW(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)

        # 用你自定义的调度器，并传入 self.min_lr
        self.lr_scheduler = get_cosine_schedule_with_warmup_and_min_lr(
            optimizer=self.optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=num_training_steps,
            min_lr=self.min_lr
        )
    
    def prediction_step(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """

        real_labels = inputs.pop("labels")
        has_labels = False if len(self.label_names) == 0 else all(inputs.get(k) is not None for k in self.label_names)
        has_labels = False        
        # For CLIP-like models capable of returning loss values.
        # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
        # is `True` in `model.forward`.
        return_loss = inputs.get("return_loss", None)
        if return_loss is None:
            return_loss = self.can_return_loss
        loss_without_labels = True if len(self.label_names) == 0 and return_loss else False

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", ["past_key_values"])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels or loss_without_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if is_sagemaker_mp_enabled():
                raw_outputs = smp_forward_only(model, inputs)
                if has_labels or loss_without_labels:
                    if isinstance(raw_outputs, dict):
                        loss_mb = raw_outputs["loss"]
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        loss_mb = raw_outputs[0]
                        logits_mb = raw_outputs[1:]

                    loss = loss_mb.reduce_mean().detach().cpu()
                    logits = smp_nested_concat(logits_mb)
                else:
                    loss = None
                    if isinstance(raw_outputs, dict):
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys)
                    else:
                        logits_mb = raw_outputs
                    logits = smp_nested_concat(logits_mb)
            else:
                if has_labels or loss_without_labels:
                    with self.compute_loss_context_manager():
                        loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                    loss = loss.detach().mean()

                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        logits = outputs[1:]
                else:
                    loss = None
                    with self.compute_loss_context_manager():
                        # outputs = model(**inputs)
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=1536,  # Limit generation length
                            do_sample=True, # sample decoding
                            temperature = 0.5,
                            top_p = 0.95,
                            top_k = 50,
                            repetition_penalty=1.0,
                            # do_sample=False, # Use greedy decoding
                            # num_beams=1,       
                            eos_token_id=self.processor.tokenizer.eos_token_id, # Stop early if possible
                            pad_token_id=self.processor.tokenizer.eos_token_id,
                            synced_gpus=True
                        )
                        input_len = inputs['input_ids'].shape[1]
                        outputs = outputs[:, input_len:]
                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                    else:
                        logits = outputs
                    # TODO: this needs to be fixed and made cleaner later.
                    if self.args.past_index >= 0:
                        self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        # if len(logits) == 1:
        #     logits = logits[0]
        labels = real_labels

        return (loss, logits, labels)