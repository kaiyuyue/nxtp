from pathlib import Path

import os
import json
import math

import torch
import torch.nn as nn

from tokenizer import Tokenizer
from models.clip import build_clip_model
from models.lang import LLaMATransformer

"""
Helper functions for loading models
"""


def load_clip(args, device=None):
    from clip.clip import _download, _MODELS

    _download(_MODELS[args.clip_model], args.clip_dir)

    model_name = args.clip_model
    if "/" in model_name:
        model_name = model_name.replace("/", "-")
        model_name = model_name.replace("@", "-")

    model_path = os.path.join(args.clip_dir, model_name + ".pt")
    with open(model_path, "rb") as opened_file:
        model = torch.jit.load(opened_file, map_location="cpu")

    # load with fp16 or bf16
    torch.set_default_tensor_type(args.fpdtype)

    model = build_clip_model(args, model.state_dict(), embed_dim=args.embed_dim)
    model.visual.logit_scale = model.logit_scale
    clip_visual_encoder = model.visual
    del model

    # switch back to fp32
    torch.set_default_tensor_type(torch.FloatTensor)

    clip_visual_encoder = clip_visual_encoder.to(device)
    return clip_visual_encoder


def load_llama(args, device=None):
    from models.lang import ModelArgs

    with open(Path(args.llama_dir) / "params.json", "r") as f:
        params = json.loads(f.read())
    model_args: ModelArgs = ModelArgs(max_seq_len=args.max_seq_len, **params)
    args.embed_dim = model_args.dim
    print(f"LLaMA model params: {vars(model_args)}")

    tokenizer = Tokenizer(model_path=args.tokenizer_path)

    # pad a special token <|empty|> to the end of the original embedding matrix
    # for the negative index of tokenizer.pad_id, i.e., -1
    special_tokens = list(args.special_tokens) + ["<|empty|>"]
    if len(special_tokens) > 0:
        tokenizer.add_special_tokens(special_tokens)
    model_args.vocab_size = tokenizer.n_words
    model_args.n_special_tokens = tokenizer.n_special_tokens
    model_args.shave_language_decoder_at = args.shave_language_decoder_at

    # load with fp16 or bf16
    torch.set_default_tensor_type(args.fpdtype)

    model = LLaMATransformer(model_args)

    # switch back to fp32
    torch.set_default_tensor_type(torch.FloatTensor)

    if not args.inference_mode:
        checkpoints = sorted(Path(args.llama_dir).glob("*.pth"))

        # TODO: support to load multiple checkpoints for LLaMA-13B / 70B
        if len(checkpoints) == 1:
            ckpt_path = checkpoints[0]
        else:
            raise ValueError(
                f"currently only support one checkpoint, got {len(checkpoints)}"
            )

        print(
            f"loading pre-trained checkpoints of LLaMA {ckpt_path} on device {device}"
        )
        checkpoint = torch.load(ckpt_path, map_location="cpu")

        if len(special_tokens) > 0:
            # pad the special tokens to the end of the original embedding matrix
            k = "tok_embeddings.weight"
            n_dim = checkpoint[k].shape[-1]
            v = torch.empty(tokenizer.n_words, n_dim).normal_(mean=0.0, std=1)
            v[: -tokenizer.n_special_tokens].copy_(checkpoint[k])
            checkpoint[k] = v

            k = "output.weight"
            n_dim = checkpoint[k].shape[-1]
            v = torch.zeros(tokenizer.n_words, n_dim)
            if tokenizer.n_special_tokens > 1:
                # largely have special tokens other than <|empty|>
                nn.init.kaiming_uniform_(v[:-1], a=math.sqrt(5))
            v[: -tokenizer.n_special_tokens].copy_(checkpoint[k])
            checkpoint[k] = v

            del v

        msgs = model.load_state_dict(checkpoint, strict=False)
        if len(msgs.missing_keys) > 0:
            print("-", msgs.missing_keys)

        del checkpoint
    else:
        print("inference mode is on, skip loading pre-trained checkpoints of LLaMA")

    model = model.to(device)

    return model, tokenizer, model_args
