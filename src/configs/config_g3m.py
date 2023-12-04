"""
Train 1.78B model on G3M.
"""

from typing import List
import copy
import os
from .config_base import args


cfg = copy.deepcopy(args)  # unwrap to override
# ------------------------------

"""
training part
"""

# experiment tag
cfg.exp_code: str = "0.0.1.g3m.slm1b.llama2"
cfg.data_name: tuple[str] = ("cc3m", "sbu", "coco")

# language decoder
cfg.llama_version: str = "2"
cfg.llama_model: str = "7B"
cfg.llama_dir: str = (
    f"{cfg.ckpt_root}/llama-1/{cfg.llama_model.upper()}"
    if cfg.llama_version == "1"
    else f"{cfg.ckpt_root}/llama-2/llama-2-{cfg.llama_model.lower()}"
)
cfg.tokenizer_path: str = f"{cfg.ckpt_root}/llama-{cfg.llama_version}/tokenizer.model"

# truncate the decoder
cfg.shave_language_decoder_at: int = 6

# vision encoder
cfg.clip_model: str = "ViT-L/14"

# hyper-params for training
cfg.max_seq_len: int = 512
cfg.input_size: int = 224
cfg.ckpt_save_interval: int = 20000

cfg.batch_size: int = 16
cfg.gradient_accumulation_steps: int = 1
cfg.log_interval: int = 100

cfg.epochs: int = 3
cfg.lr = 1e-5
cfg.grad_clip: float = 0.0
cfg.dtype: str = "float16"
cfg.force_to_use_fp16: bool = bool(1)

# token embedding
cfg.special_tokens = ("<|image|>",)
cfg.enable_prompt_augmentation: bool = bool(1)
cfg.partial_train_lang_tok_embeddings: bool = bool(1)
cfg.partial_train_lang_output: bool = bool(0)

# for checkpoint
cfg.resume: bool = bool(0)

_resume_ckpt_root = f"{cfg.ckpt_root}/x/{cfg.exp_code}"
if cfg.resume:
    if os.path.exists(_resume_ckpt_root):
        ckpts = os.listdir(_resume_ckpt_root)
        ckpts.sort()
        ckpts = [ckpt for ckpt in ckpts if ckpt.endswith(".pth")]
        cfg.resume_ckpt_path = f"{_resume_ckpt_root}/{ckpts[-1]}"  # last one
        cfg.from_scratch: bool = bool(0)
    else:
        cfg.from_scratch: bool = bool(1)

# special settings
cfg.prefix_image_tok_embeds: bool = bool(1)
cfg.decouple_label_tok_embeds: bool = bool(1)
cfg.label_contains_number: bool = bool(0)

"""
inference part
"""
ckpt_name = "ckpt_epoch_03_iter_0021360.pth"  # which checkpoint to load
cfg.eval_ckpt_path: str = f"{cfg.ckpt_root}/x/{cfg.exp_code}/{ckpt_name}"

cfg.text_decoder_strategy = "one_shot"  # greedy | beam | one_shot
cfg.max_gen_len: int = 64
cfg.beam_size: int = 3
cfg.beam_patience: float = 1.0

cfg.greedy_func = "top_k"  # top_k | top_p
cfg.top_k: int = 1
cfg.penalty: float = 1.2

cfg.temperature: float = 1.0
cfg.top_p: float = 0.9

# ------------------------------
for k, v in cfg.__dict__.items():
    if k not in args.__dict__:
        raise ValueError(f"the setting of '{k}' not in the base config")
args = cfg  # wrap back


def optim_filter(model, decay_params: List = [], other_params: List = []):
    """optimization part

    custom optimizer for different layers
    """
    max_n_len = max([len(n) for n, _ in model.named_parameters()])

    module_names = []
    module_names += ["attention." + s for s in ["wq", "wk", "wv", "wo"]]
    module_names += ["feed_forward." + s for s in ["w1", "w2", "w3"]]
    module_names += [s + "_norm" for s in ["attention", "ffn"]]

    for n, p in model.named_parameters():
        p.requires_grad = False

        # --- vision encoder ---
        if "vision_encoder.proj" in n:
            p.requires_grad = True
            if p.dim() >= 2:
                decay_params.append(p)
            else:
                other_params.append(p)

        if model.module.vision_encoder.use_default_input_resolution is False:
            if "vision_encoder.conv1" in n:
                p.requires_grad = True
                if p.dim() >= 2:
                    decay_params.append(p)
                else:
                    other_params.append(p)
            if "vision_encoder.positional_embedding" in n:
                p.requires_grad = True
                if p.dim() >= 2:
                    decay_params.append(p)
                else:
                    other_params.append(p)
            if "vision_encoder.class_embedding" in n:
                p.requires_grad = True
                if p.dim() >= 2:
                    decay_params.append(p)
                else:
                    other_params.append(p)
            if "vision_encoder.ln_pre" in n:
                p.requires_grad = True
                if p.dim() >= 2:
                    decay_params.append(p)
                else:
                    other_params.append(p)

        # vision encoder
        for _layer_id in [str(i) for i in list(range(18, 24))]:
            _layer_name = f"vision_encoder.transformer.resblocks.{_layer_id}"
            if _layer_name in n:
                p.requires_grad = True
                if p.dim() >= 2:
                    decay_params.append(p)
                else:
                    other_params.append(p)
        if "module.vision_encoder.ln_post" in n:
            p.requires_grad = True
            if p.dim() >= 2:
                decay_params.append(p)
            else:
                other_params.append(p)

        # language decoder
        if "language_decoder.tok_embeddings" in n:
            p.requires_grad = True
            other_params.append(p)  # w/o decay

        if "language_decoder.output" in n or "language_decoder.norm" in n:
            p.requires_grad = True
            decay_params.append(p)

        for _layer_id in [str(i) for i in list(range(0, 6))]:
            for _module_name in module_names:
                _layer_name = f"language_decoder.layers.{_layer_id}.{_module_name}"
                if _layer_name in n:
                    p.requires_grad = True
                    if p.dim() >= 2:
                        decay_params.append(p)
                    else:
                        other_params.append(p)

        print(
            f"p.requires_grad: {str(p.requires_grad):<5}, param: {n:<{max_n_len}}, {p.shape}"
        )

    return decay_params, other_params
