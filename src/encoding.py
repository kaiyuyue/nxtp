from functools import partial

import random
import torch

from random import shuffle
from utils import get_noun_words


def construct_text_inputs(
    args,
    captions,
    tokenizer,
    offset=0,
    is_train=True,
    empty_prompt=False,
    return_strs=False,
    skip_extract_nouns=False,
):
    """
    <|image token embeddings|> text
    """
    if offset < 0:
        raise ValueError(f"offset must be non-negative, but got {offset}")

    if not is_train:
        # force not to use prompt augmentation
        args.enable_prompt_augmentation = False

    # prompt the model to do the text completion task
    # https://github.com/facebookresearch/llama/blob/llama_v1/FAQ.md#2-generations-are-bad
    _TEMPLATES_CAPS = (
        [
            "A brief description of the given image is that",
            "A succinct explanation of the picture presented is that",
            "The visual content of the image in one sentence is that",
            "A short and clear explanation of the subsequent image is that",
            "A concise interpretation of the image provided is that",
            "A compact description of the photo's key features is that",
            "A brief, clear account of the picture shown is that",
            "A clear and concise summary of the photo is that",
            "A terse but informative summary of the picture is that",
            "A compact narrative representing the image presented is that",
        ]
        if args.enable_prompt_augmentation
        else [
            "A brief description of the given image is that",
        ]
    )
    _TEMPLATES_OBJS = (
        [
            "The objects in the image are",
            "The items present in the picture are",
            "The elements depicted in the image are",
            "The objects shown in the photograph are",
            "The items visible in the image are",
            "The objects that appear in the picture are",
            "The elements featured in the image are",
            "The items captured in the photograph are",
            "The elements seen in the picture are",
            "The items represented in the image are",
        ]
        if args.enable_prompt_augmentation
        else [
            "The objects in the image are",
        ]
    )

    if empty_prompt:
        _TEMPLATES_CAPS = [""]
        _TEMPLATES_OBJS = [""]

    _prefix = "<|image|>"
    _postfix = " " if is_train else ""  # for inference, we don't need a space
    prompts = [
        _prefix + random.choice(_TEMPLATES_CAPS) + _postfix,
        _prefix + random.choice(_TEMPLATES_OBJS) + _postfix,
    ]

    if is_train:
        caps = captions

        if skip_extract_nouns:
            _func = lambda x: x.lower()
        else:
            _func = partial(
                get_noun_words,
                contains_number=args.label_contains_number,
            )

        if isinstance(caps[0], list):
            objs = []
            for sublist in caps:
                _objs = []
                for cap in sublist:
                    _objs.append(_func(cap))
                objs.append(",".join(_objs))
            caps = [" ".join(sublist) for sublist in caps]
        else:
            objs = [_func(cap) for cap in caps]

        for i, obj in enumerate(objs):
            obj = obj.split(",")  # NOTE: default delimiter ","
            shuffle(obj)
            obj = ", ".join(obj)
            objs[i] = obj
    else:
        caps = [""] * len(captions)
        objs = [""] * len(captions)

    if return_strs:
        raw_caps = caps
        raw_objs = objs

    # filter out samples that have less than 3 objects
    valid_sample_mask = torch.ones(len(objs))
    for i, obj in enumerate(objs):
        if len(obj.split(",")) < 0:  # TODO: make this a hyperparameter
            valid_sample_mask[i] = 0.0

    caps = [prompts[0] + cap for cap in caps]
    objs = [prompts[1] + obj for obj in objs]

    # where to start computing the loss
    assert (
        offset < args.max_seq_len
    ), f"num of tokens to offset ({offset}) must be less than max_seq_len ({args.max_seq_len})"
    max_seq_len = args.max_seq_len - offset

    tokens_caps = [
        tokenizer.encode(cap, bos=True, eos=True if is_train else False)[:max_seq_len]
        for cap in caps
    ]
    tokens_objs = [
        # NOTE: we won't train <eos> token for objs
        # adding it here is for localizing padding tokens
        # it will be faded out later
        tokenizer.encode(obj, bos=True, eos=True if is_train else False)[:max_seq_len]
        for obj in objs
    ]

    t_caps = torch.full(
        (len(tokens_caps), max_seq_len), tokenizer.pad_id, dtype=torch.long
    )
    t_objs = torch.full(
        (len(tokens_objs), max_seq_len), tokenizer.pad_id, dtype=torch.long
    )
    for k, t in enumerate(tokens_caps):
        t_caps[k, : len(t)] = torch.tensor(t).long()
    for k, t in enumerate(tokens_objs):
        t_objs[k, : len(t)] = torch.tensor(t).long()

    if _prefix in args.special_tokens:
        # swap <|image|> token and tokenier.bos_id
        img_token_id = tokenizer.encode(_prefix, bos=False, eos=False)[0]
        t_caps[:, 1] = tokenizer.bos_id
        t_caps[:, 0] = img_token_id
        t_objs[:, 1] = tokenizer.bos_id
        t_objs[:, 0] = img_token_id

    tokens_caps = t_caps  # [bs, max_seq_len]
    tokens_objs = t_objs  # [bs, max_seq_len]

    dummy_token_index_cap = 0
    dummy_token_index_obj = 0

    if is_train:
        # offset is applied starting from the dummy token
        target_caps = torch.full(
            (len(tokens_caps), args.max_seq_len), tokenizer.pad_id, dtype=torch.long
        )
        target_objs = torch.full(
            (len(tokens_objs), args.max_seq_len), tokenizer.pad_id, dtype=torch.long
        )

        for k, t in enumerate(tokens_caps):
            target_caps[k, dummy_token_index_cap + offset] = tokenizer.pad_id
            target_caps[k, dummy_token_index_cap + offset + 1 : offset + len(t)] = t[
                dummy_token_index_cap + 1 :
            ]
            if valid_sample_mask[k] == 0.0:
                target_caps[k, :] = tokenizer.pad_id

        for k, t in enumerate(tokens_objs):
            target_objs[k, dummy_token_index_obj + offset] = tokenizer.pad_id
            target_objs[k, dummy_token_index_obj + offset + 1 : offset + len(t)] = t[
                dummy_token_index_obj + 1 :
            ]
            if valid_sample_mask[k] == 0.0:
                target_objs[k, :] = tokenizer.pad_id

        if args.rm_eos_token_for_objs:
            # rm <EOS> token for objs
            eos_indices = torch.where(target_objs == tokenizer.eos_id)
            target_objs[eos_indices] = tokenizer.pad_id
    else:
        target_caps = None
        target_objs = None

    if return_strs:
        return (
            tokens_caps,
            tokens_objs,
            target_caps,
            target_objs,
            dummy_token_index_cap,
            dummy_token_index_obj,
            raw_caps,
            raw_objs,
        )
    else:
        return (
            tokens_caps,
            tokens_objs,
            target_caps,
            target_objs,
            dummy_token_index_cap,
            dummy_token_index_obj,
        )


def construct_embd_inputs(
    embds_imgs,
    embds_caps,
    embds_objs,
    dummy_token_index_cap,
    dummy_token_index_obj,
    tokens_caps,  # ---- for debugging ----
    tokens_objs,
    tokenizer,
):
    bs, num_patches = embds_imgs.shape[:2]

    # construct input embeddings
    input_embds_caps = []
    input_embds_objs = []
    input_tokens_objs = []
    for k in range(bs):
        _embds_img = embds_imgs[k]
        _embds_cap = embds_caps[k]
        _embds_obj = embds_objs[k]

        _input_embds_cap = torch.cat(
            [
                _embds_cap[:dummy_token_index_cap],
                _embds_img,
                _embds_cap[dummy_token_index_cap:],
            ],
            dim=0,
        )
        _input_embds_obj = torch.cat(
            [
                _embds_obj[:dummy_token_index_obj],
                _embds_img,
                _embds_obj[dummy_token_index_obj:],
            ],
            dim=0,
        )

        input_embds_caps.append(_input_embds_cap)
        input_embds_objs.append(_input_embds_obj)

        _token = tokens_objs[k]  # or tokens_caps[k]
        _input_token = torch.cat(
            [
                _token[:dummy_token_index_obj],
                torch.full(
                    (num_patches,),
                    tokenizer.pad_id,
                    dtype=torch.long,
                    device=_token.device,
                ),
                _token[dummy_token_index_obj:],
            ],
            dim=0,
        )
        input_tokens_objs.append(_input_token)

    input_tokens_objs = torch.stack(input_tokens_objs, dim=0)  # [bs, max_seq_len]
    input_embds_caps = torch.stack(input_embds_caps, dim=0)  # [bs, max_seq_len, dim]
    input_embds_objs = torch.stack(input_embds_objs, dim=0)

    return input_embds_caps, input_embds_objs, input_tokens_objs
