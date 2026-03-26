import transformers
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-VL-7B-Instruct")
    train_phase: Optional[str] = field(default="ft_audio")
    train_mode: Optional[str] = field(default=None)
    dropout: Optional[float] = field(default=0.1)
    lora_rank: Optional[int] = field(default=8)
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.05)
    tune_vision: bool = field(default=False)
    tune_vision_conn: bool = field(default=False)
    tune_audio: bool = field(default=False)
    tune_audio_conn: bool = field(default=False)
    tune_llm: bool = field(default=False)
    tune_all: bool = field(default=False)

@dataclass
class DataArguments:
    load_batch_size: int=field(default=16)
    streaming: bool=field(default=True)
    meta_dir: str=field(default="")
    data_cache_dir: str=field(default="/mnt/vision_user/kaiyinyan/data/qwen_omni_ft/data_cache")
    reflesh_data: bool=field(default=False)
    video_max_frames: Optional[int] = field(default=8)
    video_min_frames: Optional[int] = field(default=4)
    data_flatten: bool = field(default=False)
    data_packing: bool = field(default=False)
    base_interval: int = field(default=2)
    max_pixels: int = field(default=28 * 28 * 576)
    min_pixels: int = field(default=28 * 28 * 16)
    max_n_samples: int = field(default=3 * 60 * 16000)

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    resume_training: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to resume training from the latest checkpoint."}
    )
    min_lr: Optional[float] = 0
    vision_connector_lr: Optional[float] = None
    vision_tower_lr: Optional[float] = None
    audio_connector_lr: Optional[float] = None
    audio_tower_lr: Optional[float] = None
    dataloader_drop_last: bool = field(default=True) 
