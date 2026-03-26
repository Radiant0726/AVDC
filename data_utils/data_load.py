# -*- coding: utf-8 -*
import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com" 
# os.environ["http_proxy"] = "http://172.16.6.134:12607"
# os.environ["https_proxy"] = "http://172.16.6.134:12607"
from pathlib import Path
curr_dir = Path(__file__).resolve().parent
root_dir = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(Path(curr_dir))) 
sys.path.insert(0, str(Path(root_dir))) 
# sys.path.insert(0, str(Path("/home/kaiyingyan/qwen2.5_omni_ft/transformers/src")))

import argparse
import random
import copy
import shutil
import torch
import torch.nn.functional as F
import json
import pandas as pd
from datasets import Dataset, load_dataset
# from modelscope import snapshot_download, AutoTokenizer
import transformers
from transformers import DataCollatorForSeq2Seq
from func_timeout import func_set_timeout, FunctionTimedOut
from qwen_omni_utils import process_mm_info

from prompt_types import *
from data_load_contra import *


USE_AUDIO_IN_VIDEO = True
TOTAL_MAX_LENGTH = 3000
def max_pad_seq(videos, masks=None, value=0): # 
    
    # 计算最大feadim & seqlen
    max_feadim = max(tensor.shape[0] for tensor in videos)
    max_seqlen = max(tensor.shape[1] if tensor.ndim > 1 else 1 for tensor in videos)

    padded_data = []
    for i, tensor in enumerate(videos):
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(-1)
        feadim, seqlen = tensor.shape
        pad_bottom = max_feadim - feadim
        pad_right = max_seqlen - seqlen

        padded_tensor = F.pad(tensor, (0, pad_right, 0, pad_bottom), mode='constant', value=value)
        if masks!=None and pad_right>0:
            mask_seqlen = masks[i].shape[-1]
            pad_right_mask = max_seqlen - mask_seqlen
            assert pad_right_mask>=0
            masks[i] = torch.tensor(masks[i].tolist()+[value]*pad_right_mask)
        padded_data.append(padded_tensor)
    padded_data = torch.stack(padded_data).squeeze(-1) if padded_data is not None else None
    masks = torch.stack(masks) if masks is not None else None
    return padded_data, masks

class DataCollatorForSeq2Seq_videos(DataCollatorForSeq2Seq):
    def __init__(
        self,
        tokenizer,
        model= None,
        padding = True,
        max_length= None,
        pad_to_multiple_of = None,
        label_pad_token_id: int = -100,
        return_tensors: str = "pt",
    ):
        # 调用父类的构造函数
        super().__init__(
            tokenizer=tokenizer,
            model=model,
            padding=padding,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            label_pad_token_id=label_pad_token_id,
            return_tensors=return_tensors,
        )

    def __call__(self, features, return_tensors=None):

        # images = []
        videos = []
        # video_token_indices = []
        # audio_token_indices = []
        multi_keys = ["labels", "input_ids", "attention_mask"] #
        # single_keys = ["input_features", "feature_attention_mask", "pixel_values", "pixel_values_videos"]
        features_lst = []  
        num_qa_lst = []
        for item in features:
            # images.append(torch.tensor(item.pop("pixel_values"))) # list -> tensor
            videos.append(torch.tensor(item.pop("pixel_values_videos")).squeeze(0)) # list -> tensor
            # video_token_indices.append(torch.tensor(item.pop("video_token_indices")).squeeze(0))
            # audio_token_indices.append(torch.tensor(item.pop("audio_token_indices")).squeeze(0))

            # assert item["attention_mask"].shape[0] == item["labels"].shape[0]
            num_qa = item.pop("num_qa")
            num_qa_lst.append(num_qa)
            for idx in range(num_qa):
                item_one = copy.deepcopy(item)
                for key in multi_keys:
                    item_key_one = torch.tensor(item[key][idx])
                    item_one[key] = item_key_one[item_key_one != -101].tolist()
                for key, value in item_one.items():
                    if isinstance(value, torch.Tensor):
                        if value.dim()>0: item_one[key] = value.tolist()
                        else: item_one[key] = value.item()
                features_lst.append(item_one)

        # images = max_pad_seq(images,value=0)[0]
        videos = max_pad_seq(videos, value=0)[0]
        # video_token_indices = max_pad_seq(video_token_indices, value=-1)[0]
        # audio_token_indices = max_pad_seq(audio_token_indices, value=-1)[0]

        # 调用父类的 __call__ 方法
        # batch = super().__call__(features, return_tensors)
        batch = super().__call__(features_lst, return_tensors)

        for key, value in batch.items():
            split_values = torch.split(value, num_qa_lst, dim=0)
            if key in multi_keys:
                pass
                # # 先按num_qa拆分，再拼接
                # batch[key] = torch.stack(split_values, dim=0)
            else:
                batch[key] = torch.stack([split_value[0] for split_value in split_values], dim=0)

        # batch["pixel_values"] = images
        batch["pixel_values_videos"] = videos
        # batch["video_token_indices"] = video_token_indices
        # batch["audio_token_indices"] = audio_token_indices
        batch["num_qa"] = torch.tensor(num_qa_lst)

        return batch

# def check_av_len(inputs):
#     second_per_grid_ts = inputs["second_per_grid_ts"][0]
#     audio_timestamps = inputs["audio_timestamps"][0]
#     video_grid_thw = inputs["video_grid_thw"][0]
#     video_t = video_grid_thw[0]
#     audio_t = torch.max(audio_timestamps) // second_per_grid_ts if second_per_grid_ts!=0 else 0

#     # if audio_t-video_t>=0 and audio_t-video_t<=1:
#     if abs(audio_t-video_t)<=1:
#         flag = True
#     else:
#         flag = False
    
#     return flag, video_t, audio_t

def left_pad_tensor(input_lst, target_len, pad_value = 0):
    
    # 找最大长度
    max_len = max(len(x) for x in input_lst)

    # 左填充
    padded = []
    for x in input_lst:
        padding = [pad_value] * (max_len - len(x))
        new_x = padding + x
        padded.append(new_x)

    while len(padded) < target_len:
        padded.append([pad_value] * max_len)

    # 现在 padded 是一个 list of lists，长度都是 max_len
    tensor = torch.tensor(padded)  # shape [batch_size, max_len]
    return tensor

@func_set_timeout(180)  # 设置超时
def prepare_inputs(messages, has_audio, text, processor):
    # audios, images, videos = process_mm_info(messages, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    # audios, images, videos, video_sample_args = process_mm_info(messages, use_audio_in_video=USE_AUDIO_IN_VIDEO, return_video_kwargs=True)
    audios, images, videos, video_sample_args = process_mm_info(messages, use_audio_in_video=has_audio, return_video_kwargs=True)
    inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=has_audio, **video_sample_args)
    return inputs

# NUM_TIMESTAMPS_SAMPLE = 3
# NUM_TIMESTAMPS_SAMPLE = 1
def process_func(data_path, data_type, timestamps, 
                prompts, output_contents, a_type, duration, has_audio, crop_range,
                processor, tokenizer, mode, **kwargs):
    """
    将数据集进行预处理
    """

    system_content = []
    content = []

    # if duration < 20:
    #     CURR_NUM_TIMESTAMPS_SAMPLE = 2
    # else:
    #     CURR_NUM_TIMESTAMPS_SAMPLE = 1

    # fps = 1.0
    if duration < 30:
        fps = 2.0
    else:
        fps = 1.0

    if data_type=="video":

        if crop_range is None:
            content.append(
                {
                    "type": "video",
                    "video":data_path,
                    "fps":fps,
                }
            )
        else:
            content.append(
                {
                    "type": "video",
                    "video":data_path,
                    "fps":fps,
                    "video_start":crop_range[0],
                    "video_end":crop_range[1],
                }
            )
        pre_prompt = "Carefully watch the video, listen to the audio and pay attention to every detail. Address the question accurately based on your observations."
    elif data_type=="audio":
        if crop_range is None:
            content.append(
                {
                    "type": "audio",
                    "audio":data_path,
                }
            )
        else:
            content.append(
                {
                    "type": "audio",
                    "audio":data_path,
                    "audio_start":crop_range[0],
                    "audio_end":crop_range[1],
                }
            )
        pre_prompt = "Carefully Listen to the audio and pay attention to every detail. Address the question accurately based on your observations."
    elif data_type=="image":
        content.append(
            {
                "type": "image",
                "image":data_path,
            }
        )
        pre_prompt = "Carefully watch the image and pay attention to every detail. Address the question accurately based on your observations."
    system_content.append({"type": "text", "text": pre_prompt})

    if "mc" in a_type.lower() or "open_end" in a_type.lower(): # instruct
        
        if "MC" in a_type:
            if "cot" in a_type.lower():
                a_prompt = "Select the only one correct option for the following multiple-choice question."
            elif "only" in a_type.lower():
                a_prompt = "Select the only one correct option for the following multiple-choice question. Only respond with the letter corresponding to the correct answer."
            else:
                raise ValueError
            system_content.append({"type": "text", "text": a_prompt})
        
        if "cot" in a_type.lower():
            if "av-cot" in a_type.lower():
                cot_prompt = """Think deeply, explain your reasoning thoroughly, and then provide the final answer.
Your reasoning should explicitly follow these steps: (1) question decomposition, (2) temporal grounding, (3) visual and audio perception (4) multimodal reasoning and answer synthesis.
Enclose your reasoning process within <think>...</think> and enclose your final answer within <answer>...</answer>.
"""
            else:
                cot_prompt = """Think deeply, explain your reasoning thoroughly, and give the final answer. 
Enclose your reasoning process within <think>...</think> and enclose your final answer within <answer>...</answer>.
"""

            system_content.append({"type": "text", "text": cot_prompt})

    messages = [
        {
            "role": "system",
            "content": system_content,
        },
        {
            "role": "user",
            "content": content,
        }
    ]

    assert len(prompts) == len(output_contents), f"prompts and output_contents length mismatch: {len(prompts)} vs {len(output_contents)}"
    
    qa_contents = []
    for prompt, output_content in zip(prompts, output_contents):
        qa_content = [
            {
                "role": "user",
                "content": [
                    {   
                        "type": "text", 
                        "text": prompt
                    }
                ]
            },
            {
                "role": "assistant",
                "content": [{   
                        "type": "text", 
                        "text": output_content
                    }
                ]
            }
        ]
        qa_contents.append(qa_content)
    
    qa_messages = []
    if timestamps[0]==[]:
        qa_messages.append(qa_contents[0]) # 全局
        qa_contents = qa_contents[1:] # 局部


    if len(qa_contents) > 0: # 局部
        random.shuffle(qa_contents)
        qa_contents_time = qa_contents[0]
        qa_messages.append(qa_contents_time)
    # CURR_NUM_TIMESTAMPS_SAMPLE = CURR_NUM_TIMESTAMPS_SAMPLE - len(qa_messages) 
    # if CURR_NUM_TIMESTAMPS_SAMPLE > 0:
    #     sample_num = min(CURR_NUM_TIMESTAMPS_SAMPLE, len(qa_contents)) # 2 if duration < 30 else 1

    #     if sample_num > 0: # 局部
    #         if len(qa_contents) > sample_num:
    #             qa_contents_idx = random.sample(list(range(len(qa_contents))), sample_num) # sample some qa
    #             qa_contents = [qa for i, qa in enumerate(qa_contents) if i in qa_contents_idx]
            
    #         for qa_item in qa_contents:
    #             qa_messages.append(qa_item)

    # num_qa = len(qa_messages)
    # for qa in qa_messages:
    #     messages.extend(qa)

    num_qa = 1
    random.shuffle(qa_messages)
    messages.extend(qa_messages[0])
    
    # if timestamps is not None:
    #     label_timestamps = [t for i, t in enumerate(timestamps) if i in qa_contents_idx]
    # else:
    #     label_timestamps = [[-1,-1] for i in range(len(qa_contents_idx))]

    # target = [
    #     {
    #         "role": "assistant",
    #         "content": output_content
    #     }
    # ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    ) 

    # tokenizer = copy.deepcopy(tokenizer)
    # chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    # tokenizer.chat_template = chat_template

    # response = tokenizer.apply_chat_template(target)
    # response_label = copy.deepcopy(response)
    # response_label[:3] = [-100] * 3  # 去除模板前缀
    # response = tokenizer(f"{output_content}", add_special_tokens=False)

    # for test
    # inputs = prepare_inputs(messages, has_audio, text, processor)
    try:
        inputs = prepare_inputs(messages, has_audio, text, processor)
        inputs["use_audio_in_video"] = has_audio
    except FunctionTimedOut as e:
        print(f"[Timeout]: {e}, skip sample ({data_path}, {data_type})")
        return None
    except Exception as e:
        print(f"[Error] prepare_inputs failed with exception: {e}, skip sample ({data_path}, {data_type})")
        return None
    # image_inputs, video_inputs = process_vision_info(messages) 
    # inputs = processor(text=[text],images=image_inputs,videos=video_inputs,padding=True,return_tensors="pt")

    # if video_path:
    #     flag_avlen, video_t, audio_t = check_av_len(inputs)
    #     if not flag_avlen:
    #         print(f"[Error] video(len={video_t}) and audio(len={audio_t}) are not same len, skip sample ({video_path}, {audio_path})")
    #         return None

    instruction = {key: value.tolist() for key, value in inputs.items() if isinstance(value,torch.Tensor)} #tensor -> list,为了方便拼接

    # "151644": "<|im_start|>"
    # "151645": "<|im_end|>"
    input_ids = instruction["input_ids"][0] 
    im_start_idx = [index for index, value in enumerate(input_ids) if value == 151644]
    # im_end_idx = [index for index, value in enumerate(input_ids) if value == 151645]
    
    omni_im_start_idx = im_start_idx[-1]
    answer_perdix_len = 3 # <|im_start|>assistant\n
    input_ids = input_ids[:omni_im_start_idx]

    im_start_idx = im_start_idx[2:]
    # im_end_idx = im_end_idx[2:]
    qa_start_idx = im_start_idx[0]

    all_labels = []
    all_input_ids = []
    all_attention_masks = []
    input_ids_av = input_ids[:qa_start_idx]
    attention_mask = instruction["attention_mask"][0][:qa_start_idx]
    # qa_attention_mask = [0] * (len(input_ids) - qa_start_idx)
    # attention_mask = attention_mask + qa_attention_mask
    labels = [-100] * len(attention_mask) 
    for num in range(num_qa):
        q_start_idx = im_start_idx[2 * num]
        a_start_idx = im_start_idx[2 * num + 1] + answer_perdix_len
        next_q_start_idx = im_start_idx[2 * num + 2] # 加入 "\n"

        curr_attention_mask = copy.deepcopy(attention_mask)
        curr_labels = copy.deepcopy(labels)
        curr_input_ids = copy.deepcopy(input_ids_av)
        if mode == "train":
            curr_attention_mask = curr_attention_mask + [1] * (next_q_start_idx - q_start_idx)
            all_attention_masks.append(curr_attention_mask)

            curr_input_ids = curr_input_ids + input_ids[q_start_idx: next_q_start_idx]
            all_input_ids.append(curr_input_ids)

            curr_labels = curr_labels + [-100] * (a_start_idx - q_start_idx) + input_ids[a_start_idx: next_q_start_idx] 
            all_labels.append(curr_labels)

        else:
            curr_attention_mask = curr_attention_mask + [1] * (a_start_idx - q_start_idx)
            all_attention_masks.append(curr_attention_mask)

            curr_input_ids = curr_input_ids + input_ids[q_start_idx: a_start_idx]
            all_input_ids.append(curr_input_ids)

            curr_labels = [-100] * (len(curr_input_ids) - (next_q_start_idx-a_start_idx)) + input_ids[a_start_idx: next_q_start_idx]  # 保持长度
            all_labels.append(curr_labels)

        assert len(curr_labels) == len(curr_attention_mask) == len(curr_input_ids), f"label, attention_mask and input_ids length mismatch: {len(labels)}, {len(curr_attention_mask)}, {len(input_ids)}"


    # max length限制 太长了舍弃
    # if mode=="train" and len(input_ids) > TOTAL_MAX_LENGTH:  
    #     return None
        # input_ids = input_ids[:MAX_LENGTH] # 做一个截断
        # attention_mask = attention_mask[:MAX_LENGTH]
        # labels = labels[:MAX_LENGTH]

    # input_ids = torch.tensor(all_input_ids)
    # attention_mask = torch.tensor(all_attention_masks)
    # labels = torch.tensor(all_labels)

    input_ids = left_pad_tensor(all_input_ids, NUM_TIMESTAMPS_SAMPLE, -101)
    attention_mask = left_pad_tensor(all_attention_masks, NUM_TIMESTAMPS_SAMPLE, -101)
    labels = left_pad_tensor(all_labels, NUM_TIMESTAMPS_SAMPLE, -101)
    # label_timestamps = torch.tensor(label_timestamps)

    print(data_path, data_type, duration, len(input_ids), len(input_ids[0]))
    assert input_ids.shape == attention_mask.shape == labels.shape

    for key,value in inputs.items():
        if isinstance(value,torch.Tensor):
            inputs[key] = value.squeeze(0)
    
    inputs.update({"input_ids": input_ids, "attention_mask": attention_mask, 
                   "labels": labels, # "label_timestamps":label_timestamps,
                   "num_qa": num_qa})
    
    return inputs

def process_func_batch(batch_item, processor, tokenizer, mode):

    batch_inputs = {}
    batch_size = len(batch_item['data_path'])
    for data_path, data_type, timestamps, prompts, output_contents, a_type, duration, has_audio, crop_range in zip(batch_item['data_path'], batch_item['data_type'], batch_item["timestamps"], batch_item["prompts"], batch_item["output_contents"],  batch_item["a_type"], batch_item["duration"], batch_item["has_audio"], batch_item["crop_range"]):
        inputs = process_func(data_path, data_type, timestamps, prompts, output_contents, a_type, duration, has_audio, crop_range, processor, tokenizer, mode)
        if inputs == None: continue
        for key, value in inputs.items():
            if key not in batch_inputs.keys(): batch_inputs[key] = []
            batch_inputs[key].append(value)
    curr_batch_size = len(next(iter(batch_inputs.values())))
    if curr_batch_size < batch_size:
        remain_size = batch_size - curr_batch_size
        remain_sample_idx = random.sample(list(range(curr_batch_size)), k=remain_size)
        for key, value in batch_inputs.items():
            batch_inputs_remain_samples = [batch_inputs[key][idx] for idx in remain_sample_idx]
            batch_inputs[key].extend(batch_inputs_remain_samples)
    return batch_inputs

def create_dataset(data_args, processor, tokenizer,do_eval=False):
    dataset_split_paths = data_args.dataset_use
    load_batch_size = data_args.load_batch_size
    streaming = data_args.streaming
    data_cache_dir = data_args.data_cache_dir
    reflesh_data = data_args.reflesh_data

    if reflesh_data:
        try:
            shutil.rmtree(data_cache_dir)
        except:
            pass
        os.makedirs(data_cache_dir, exist_ok=True)

    if not do_eval:
        dataset_split_paths.pop("val",None)
        dataset_split_paths.pop("test",None)

    with open(dataset_split_paths["train"], 'r', encoding='utf-8') as f:
        data = json.load(f)
    columns = data[0].keys()
    train_num = len(data)
    del data

    train_ds = load_dataset("json", data_files={"train": dataset_split_paths["train"]}, 
                            streaming=streaming, cache_dir=f"{data_cache_dir}/train")["train"]
    train_ds = train_ds.shuffle(seed=42, buffer_size=1000)
    train_dataset = train_ds.map(process_func_batch, 
                    batched=True, batch_size=load_batch_size, remove_columns=columns,
                    fn_kwargs={"processor": processor, "tokenizer": tokenizer, "mode":"train"}) 
    
    val_dataset, test_dataset = None, None
    if "val" in dataset_split_paths.keys(): 
        val_ds = load_dataset("json", data_files={"val": dataset_split_paths["val"]}, 
                                streaming=False, cache_dir=f"{data_cache_dir}/val")["val"]
        val_dataset = val_ds.map(process_func_batch, 
                        batched=True, batch_size=load_batch_size, remove_columns=columns,
                        fn_kwargs={"processor": processor, "tokenizer": tokenizer, "mode":"val"})     
    # if "test" in dataset_split_paths.keys():            
    #     test_ds = load_dataset("json", data_files={"test": dataset_split_paths["test"]}, 
    #                             streaming=False, cache_dir=f"{data_cache_dir}/test")["test"]           

    #     test_dataset = test_ds.map(process_func_batch, 
    #                     batched=True, batch_size=load_batch_size, remove_columns=columns,
    #                     fn_kwargs={"processor": processor, "tokenizer": tokenizer})    

    dataset = {
        "train":train_dataset,
        "val":val_dataset,
        "test":test_dataset,
        "train_num":train_num,
    }
    return dataset
