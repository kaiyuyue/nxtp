from typing import List
from dataclasses import dataclass
import os


@dataclass
class Args:
    """
    This is the base config class that contains
    all the hyperparameters with default values.
    """

    # hparams for directories
    data_root: str = "none"

    ckpt_root: str = f"{data_root}/checkpoints"
    cache_root: str = f"{ckpt_root}/cache"

    ckpt_dir: str = f"{ckpt_root}/x"
    hf_cache_dir: str = f"{ckpt_root}/huggingface"
    results_dir: str = f"{ckpt_root}/results"

    # hparams for caching results
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)

    # hparams for experiment
    exp_code: str = None
    data_name: tuple[str] = ("coco", "cc3m", "sbu")

    # hparams for resuming
    resume: bool = bool(0)
    resume_ckpt_path: str = None
    from_scratch: bool = bool(1)  # if True, training from scratch with resumed weights

    # hparams for training
    seed: int = 42
    input_size: int = 224
    batch_size: int = 1
    num_workers: int = 11
    pin_mem: bool = True
    parallel_mode: str = "ddp"  # ddp or fsdp
    lr: float = 6.25e-5
    lr_min: float = 0.0
    wd: float = 1e-1
    epochs: int = 10
    warmup_ratio: float = 0.2  # w/ one hundred percent
    warmup_steps: int = 2000
    dtype: str = "float16"  # float16 or float32
    force_to_use_fp16: bool = bool(
        1
    )  # avoid switching to bf16 automatically when supported
    compile: bool = bool(0)  # with pytorch 2.0 and above
    gradient_accumulation_steps: int = 8
    grad_clip: float = 0.0
    log_interval: int = 100
    ckpt_save_num: int = 10  # how many checkpoints to save in total
    ckpt_save_interval: int = -1  # how many steps to save a checkpoint

    # hparams for prompt and tokenizer
    enable_prompt_augmentation: bool = bool(1)
    special_tokens: tuple = ()
    special_tokens_to_train: tuple = ()
    rm_eos_token_for_objs: bool = bool(1)

    # hparams for input text
    label_contains_number: bool = bool(0)

    # hparams for model
    shave_language_decoder_at: int = -1  # -1 means not shaving

    # hparams for partial training
    partial_train_lang_output: bool = bool(0)
    partial_train_lang_tok_embeddings: bool = bool(0)

    # hparams for loss
    weight_loss_cap: float = 0.0
    weight_loss_obj: float = 1.0

    # hparams for tokens
    prefix_image_tok_embeds: bool = bool(0)
    decouple_label_tok_embeds: bool = bool(0)

    # hparams for CLIP (https://github.com/openai/CLIP)
    """
    python -c "import clip; print(clip.available_models())"
    ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
    """
    clip_model: str = "ViT-L/14"
    clip_dir: str = f"{ckpt_root}/clip"

    # hparams for LLaMA (https://github.com/facebookresearch/llama)
    llama_version: str = "2"
    llama_model: str = "7B"
    llama_dir: str = None
    tokenizer_path: str = f"{ckpt_root}/llama-{llama_version}/tokenizer.model"
    max_seq_len: int = 512
    max_gen_len: int = 128

    # hparams for sampling
    text_decoder_strategy: str = "greedy"  # or beam
    greedy_func: str = "top_p"  # top_k or top_p

    beam_size: int = 3
    beam_patience: float = 1.0

    temperature: float = 0.6
    penalty: float = 0.0
    top_p: float = 0.9
    top_k: int = 3

    # extra hparams for inference and evaluation
    inference_mode: str = bool(0)
    eval_ckpt_path: str = None
    input_image_path: str = None
    jupyter_mode: bool = bool(0)


args = Args()


def optim_filter(model, decay_params: List = [], other_params: List = []):
    return decay_params, other_params
