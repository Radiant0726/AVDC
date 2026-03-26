#!/bin/bash

# Distributed training configuration
export CUDA_VISIBLE_DEVICES=0,1,2,3
NPROC_PER_NODE=4  # 使用 GPUs
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}

# DeepSpeed configuration
deepspeed=configs/zero3.json

# Training entry point
entry_file=train_qwenomni.py

# Training hyperparameters
save_steps=200

dropout=0.1

train_phase="ft_omni" # ft_omni / ft_instruct
train_mode="None" 
resume_training=False

if [ "$train_phase" = "ft_omni" ]; then # ft_omni
max_steps=3000
warmup_steps=500
metric_for_best_model="cider_meteor" 
lr=1e-5
min_lr=1e-6
batch_size=1
grad_accum_steps=4
llm="Qwen/Qwen2.5-Omni-7B"

else # ft_instruct
max_steps=10000
warmup_steps=1000
metric_for_best_model="accuracy"  # loss / accuracy 
lr=5e-6
min_lr=5e-7
batch_size=1
grad_accum_steps=4
llm=""
fi

meta_dir=data
output_dir=models/qwen2.5_omni_ft

# Training arguments
args="--deepspeed ${deepspeed} 
--model_name_or_path ${llm} 
--train_phase ${train_phase} 
--train_mode ${train_mode} 
--dropout ${dropout}
--meta_dir ${meta_dir}
--load_batch_size 32
--streaming True
--dataloader_drop_last True
--reflesh_data False
--resume_training ${resume_training}
--do_train 
--bf16 True
--output_dir ${output_dir} 
--max_steps ${max_steps} 
--num_train_epochs 3
--logging_steps 1 
--gradient_checkpointing True     
--per_device_train_batch_size ${batch_size} 
--gradient_accumulation_steps ${grad_accum_steps} 
--max_pixels 100000 
--min_pixels 3136 
--max_n_samples 1000000
--do_eval 
--per_device_eval_batch_size ${batch_size} 
--eval_strategy steps 
--eval_steps ${save_steps} 
--eval_accumulation_steps ${save_steps}  
--eval_on_start False
--metric_for_best_model ${metric_for_best_model}
--load_best_model_at_end True
--save_strategy steps 
--save_steps ${save_steps} 
--save_total_limit 3 
--learning_rate ${lr} 
--min_lr ${min_lr}
--weight_decay 0.005
--warmup_steps ${warmup_steps}
--max_grad_norm 1 
--lr_scheduler_type cosine 
--dataloader_num_workers 1
--run_name ${train_phase} 
--report_to swanlab"

# Launch training
torchrun --nproc_per_node=${NPROC_PER_NODE} \
        --master_addr=${MASTER_ADDR} \
        --master_port=${MASTER_PORT} \
        ${entry_file} ${args}