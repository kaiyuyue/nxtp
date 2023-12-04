from typing import List

import builtins
import datetime
import importlib
import os
import sys
import re

import numpy as np
import nltk
import torch
import torch.distributed as dist

from random import shuffle
from nltk.corpus import words
from nltk.stem import WordNetLemmatizer
from fairscale.nn.model_parallel.initialize import initialize_model_parallel

"""
Functions for training
"""


def load_config(argv):
    argv = argv[1:]  # rm the script self name
    if len(argv) == 0:
        raise ValueError("please pass the config file")
    if len(argv) > 1:
        raise ValueError("only accept one argument for the config file")
    cfg_path = str(argv[0]).replace("/", ".").replace(".py", "")
    cfg = importlib.import_module(cfg_path)
    return cfg


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def set_dtype(args):
    cuda_is_available = torch.cuda.is_available()
    if not args.force_to_use_fp16 and args.dtype == "float16":
        if cuda_is_available and torch.cuda.is_bf16_supported():
            args.dtype = "bfloat16"
            print("bfloat16 is supported: float16 -> bfloat16")

    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.dtype]

    # floating-point dtype for torch.tensor()
    fpdtype = {
        "float32": torch.cuda.FloatTensor if cuda_is_available else torch.FloatTensor,
        "float16": torch.cuda.HalfTensor if cuda_is_available else torch.HalfTensor,
        "bfloat16": torch.cuda.BFloat16Tensor
        if cuda_is_available
        else torch.BFloat16Tensor,
    }[args.dtype]

    args.ptdtype = ptdtype
    args.fpdtype = fpdtype
    return args


def setup_model_parallel(seed=1, mute_non_master_ranks=False):
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))
    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    global_rank = dist.get_rank()
    set_seed(seed + global_rank)  # seed

    print(
        f"> local_rank: {local_rank}, world_size: {world_size}, global_rank: {global_rank}"
    )
    if mute_non_master_ranks and global_rank > 0:
        sys.stdout = open(os.devnull, "w")
    return local_rank, global_rank, world_size, device


def setup_for_distributed(is_master):
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        force = force or (dist.get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time().strftime("%H:%M:%S.%f")[:-3]
            builtin_print("[{}] ".format(now), end="")  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print
    pass


def save_checkpoint(
    args,
    model,
    optimizer,
    scheduler,
    epoch,
    global_step,
    only_save_trainable_params=True,
):
    dir = os.path.join(args.ckpt_dir, args.exp_code)
    os.makedirs(dir, exist_ok=True)
    ckpt_path = os.path.join(
        dir,
        f"ckpt_{str(epoch).zfill(2)}_{str(global_step).zfill(7)}.pth",
    )
    print(f"saving checkpoint to {ckpt_path}")

    if only_save_trainable_params:
        # only save the trained model state dict for reducing the file size
        model_state_dict = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                model_state_dict[n] = p
    else:
        model_state_dict = model.state_dict()

    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "model": model_state_dict,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        ckpt_path,
    )


def load_checkpoint(
    args, model, optimizer=None, scheduler=None, strict=True, verbose=False
):
    print(f"loading checkpoint from {args.resume_ckpt_path}")
    ckpt = torch.load(args.resume_ckpt_path, map_location="cpu")

    if verbose:
        print("loading keys:")
    _state_dict = {}
    for n, v in ckpt["model"].items():
        if "module" in n and not hasattr(model, "module"):
            n = n.replace("module.", "")

        _state_dict[n] = v
        if verbose:
            print(f"- {n}: {v.shape}")
    msgs = model.load_state_dict(_state_dict, strict=strict)
    del _state_dict

    if verbose and len(msgs.missing_keys) > 0:
        print(msgs)

    if optimizer is not None:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except Exception as e:
            print(f"failed to load optimizer state dict: {e}")

    if scheduler is not None:
        try:
            scheduler.load_state_dict(ckpt["scheduler"])
        except Exception as e:
            print(f"failed to load scheduler state dict: {e}")

    epoch = ckpt["epoch"] if "epoch" in ckpt else 1
    global_step = ckpt["global_step"] if "global_step" in ckpt else 1
    del ckpt

    return epoch, global_step


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)

    _, pred = output.topk(5, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].flatten().float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / target.shape[0]))
    return res


"""
Functions for NLTK
"""


def init_nltk(download_dir=None, force=False):
    if force or not os.path.exists(download_dir):
        nltk.download("punkt", download_dir=download_dir, force=force)
        nltk.download("words", download_dir=download_dir, force=force)
        nltk.download(
            "averaged_perceptron_tagger", download_dir=download_dir, force=force
        )
        nltk.download("wordnet", download_dir=download_dir, force=force)
    if download_dir is not None:
        nltk.data.path.append(download_dir)

    nltk_words = set(words.words())
    return nltk_words


def filter_input_text(text):
    text = text.lower()

    # rm massive noise words
    NOISE_WORDS = [
        "person",
        "persons",
        "stock",
        "illustration",
        "background",
        "photography",
        "image",
        "images",
        "front",
        "day",
        "ounce",
    ]
    for nw in NOISE_WORDS:
        text = text.replace(nw, " ")

    # rm special characters
    text = re.sub(r"[^a-zA-Z' 0-9.,&-]", "", text)
    for c in [",", "&"]:
        text = text.replace(c, " ")
    return text


def word_contains_number(word: str) -> bool:
    return any(char.isdigit() for char in word)


def lemmatize_word(word: str) -> str:
    lemmatizer = WordNetLemmatizer()
    return lemmatizer.lemmatize(word, pos="n")  # 'n' indicates noun


def get_noun_words(
    text,
    with_adj_words=False,
    max_size=-1,
    contains_number=False,
    nltk_words=None,
    debug_mode=False,
):
    """
    Extract noun words in a sentence, e.g.,
        text: 'this is a very typical bus station'
        tkns: ['a', 'very', 'typical', 'bus', 'station']
        tags: [('a', 'DT'), ('very', 'RB'), ('typical', 'JJ'), ('bus', 'NN'), ('station', 'NN')]

    Tag list:
        1. https://www.nltk.org/book_1ed/ch05.html
        2. https://stackoverflow.com/questions/15388831/what-are-all-possible-pos-tags-of-nltk
    """
    TAGS = ["NN", "NNS"]
    if with_adj_words:
        TAGS.append("JJ")

    raw_text = text

    text = filter_input_text(text)
    tkns = nltk.word_tokenize(text)
    tags = nltk.pos_tag(tkns)
    tgts = [word for word, tag in tags if tag in TAGS]

    if not contains_number:
        tgts = [tgt for tgt in tgts if not word_contains_number(tgt)]
    if nltk_words is not None:
        tgts = [w for w in tgts if w in nltk_words]

    tgts = list(set(tgts))
    tgts = [lemmatize_word(tgt) for tgt in tgts]
    tgts = [tgt for tgt in tgts if len(tgt) > 2]

    # shuffle tokens
    shuffle(tgts)

    # shave if too long
    if max_size > 0:
        tgts = tgts[:max_size]

    if debug_mode:
        print(raw_text, ": ", tgts)

    return ",".join(tgts)  # ease batching
