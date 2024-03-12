import sys
import os
import time

import torch
import torch.nn.functional as F

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import destroy_process_group
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.classifier import LangClassifier
from encoding import construct_text_inputs, construct_embd_inputs
from loader import build_dataloader
from functions import load_clip, load_llama
from utils import (
    load_config,
    init_nltk,
    set_dtype,
    setup_model_parallel,
    setup_for_distributed,
    save_checkpoint,
    load_checkpoint,
)


def main(cfg):
    """
    cfg contains two objectives:
        - cfg.args (Class): the arguments
        - cfg.param_filter (func): the filter for optimizer
    """
    args = cfg.args
    init_nltk(download_dir=f"{args.cache_root}/nltk_data")

    rank, global_rank, world_size, device = setup_model_parallel(
        seed=args.seed,
        mute_non_master_ranks=True,
    )
    master_process = global_rank == 0
    setup_for_distributed(master_process)
    print(f"training params:")
    for k, v in args.__dict__.items():
        print(f"- {k}: {v}")

    args = set_dtype(args)

    llama_model, tokenizer, model_args = load_llama(args, device)
    clip_model = load_clip(args, device)
    dataloader = build_dataloader(args, global_rank, world_size, is_train=True)

    ctx = torch.amp.autocast(device_type="cuda", dtype=args.ptdtype)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == "float16"))

    if args.compile:
        print("compiling the model...")
        clip_model = torch.compile(clip_model)

    model = LangClassifier(vision_encoder=clip_model, language_decoder=llama_model)
    model = model.to(device)
    model = DDP(model, device_ids=[rank])

    # params
    total_params = sum(param.numel() for param in model.parameters()) / 1e6
    print(f"total params: {total_params:.2f} M, {total_params / 1e3:.2f} B")

    decay_params = []
    other_params = []  # w/o weight decay
    decay_params, other_params = cfg.optim_filter(model, decay_params, other_params)

    optim_groups = [
        {"params": decay_params, "weight_decay": args.wd},
        {"params": other_params, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(
        optim_groups, lr=args.lr, betas=(0.9, 0.98), eps=1e-6, fused=True
    )
    print(optimizer)

    # handle partial training for special tokens
    special_token_ids = {
        tokenizer.encode(t, bos=False, eos=False)[0]: t for t in args.special_tokens
    }
    if (
        args.partial_train_lang_tok_embeddings
        and model.module.language_decoder.tok_embeddings.weight.requires_grad is True
    ):
        _grad_mask = torch.zeros_like(
            model.module.language_decoder.tok_embeddings.weight
        )
        for special_token_id in special_token_ids:
            _grad_mask[special_token_id, :] = 1.0
        model.module.language_decoder.tok_embeddings.weight.register_hook(
            lambda grad: grad.mul_(_grad_mask)
        )
        print(
            f"register a hook to train the delimiter token $ in 'language_decoder.tok_embeddings'"
            f"(token ids: {special_token_ids})"
        )

    if (
        args.partial_train_lang_output
        and model.module.language_decoder.output.weight.requires_grad is True
    ):
        _grad_mask = torch.zeros_like(model.module.language_decoder.output.weight)
        for special_token_id in special_token_ids:
            _grad_mask[special_token_id, :] = 1.0
        model.module.language_decoder.output.weight.register_hook(
            lambda grad: grad.mul_(_grad_mask)
        )
        print(
            f"register a hook to train the delimiter token $ in 'language_decoder.output'"
            f"(token ids: {special_token_ids})"
        )

    # batch size
    grad_accum_steps = args.gradient_accumulation_steps
    batch_size = args.batch_size * grad_accum_steps

    # steps
    steps_per_epoch = len(dataloader) // grad_accum_steps
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(int(args.warmup_ratio * 0.01 * total_steps), args.warmup_steps)

    # ckpt frequency
    ckpt_save_interval = total_steps // args.ckpt_save_num
    ckpt_save_interval = (
        args.ckpt_save_interval if args.ckpt_save_interval > 0 else ckpt_save_interval
    )

    lr_scheduler = CosineAnnealingLR(
        optimizer, T_max=total_steps - warmup_steps, eta_min=args.lr_min
    )
    grad_clip = args.grad_clip
    print(
        f"total training steps: {total_steps}\n",
        f"  epochs: {args.epochs}\n",
        f"  dataloader length: {len(dataloader)} iters or {steps_per_epoch} steps in each epoch\n",
        f"  gradient accumulation steps: {grad_accum_steps}\n",
        f"  warmup steps: {warmup_steps} w/ ratio {args.warmup_ratio:.2f}%\n",
        f"  total batch size: {batch_size * world_size}\n",
        f"  ckpt save interval: {ckpt_save_interval}\n",
        f"  grad clip: {grad_clip}",
    )

    global_step = 0
    start_epoch = 1
    pgs = -1  # previous global step for saving ckpt
    rgs = -1  # reloaded global step

    if args.resume_ckpt_path and args.resume:
        rse, rgs = load_checkpoint(
            args,
            model,
            None if args.from_scratch else optimizer,
            None if args.from_scratch else lr_scheduler,
            strict=False,
        )
        if not args.from_scratch:
            global_step = rgs
            start_epoch = rse  # resumed start epoch

            # the optimizer might have its state tensors on a different device
            # than the model, so we need to move them accordingly
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)

        print(f"resume training at epoch {start_epoch} and global step {global_step}")

    if master_process and rgs == -1:
        os.makedirs(
            os.path.join(args.ckpt_dir, args.exp_code),
            exist_ok=True,
        )
        save_checkpoint(args, model, optimizer, lr_scheduler, 0, 0)

    for epoch in range(start_epoch, args.epochs + 1):
        optimizer.zero_grad(set_to_none=True)

        t = time.perf_counter()
        for i, (imgs, caps, keys) in enumerate(dataloader, start=1):
            data_time = time.perf_counter() - t

            with ctx:
                # image embeddings
                imgs = imgs.to(device, non_blocking=True)

                h = model.module.encode_images(imgs)
                z = model.module.decode_images(h)

                embds_clss, embds_imgs = z[:, :1], z[:, 1:]
                bs, n_img_tokens = embds_imgs.shape[:2]

                # text embeddings
                text_inputs = construct_text_inputs(
                    args,
                    caps,
                    tokenizer,
                    offset=n_img_tokens,
                    is_train=True,
                    return_strs=False,
                )
                (
                    tokens_caps,
                    tokens_objs,
                    target_caps,
                    target_objs,
                    dummy_token_index_cap,
                    dummy_token_index_obj,
                ) = text_inputs[:6]

                tokens_caps = tokens_caps.to(device)  # [bs, max_seq_len + 1]
                tokens_objs = tokens_objs.to(device)
                target_caps = target_caps.to(device)  # [bs, max_seq_len + 1]
                target_objs = target_objs.to(device)

                Wte = model.module.language_decoder.tok_embeddings.weight
                embds_caps = Wte[tokens_caps]
                embds_objs = Wte[tokens_objs]

                (
                    input_embds_caps,
                    input_embds_objs,
                    input_tokens_objs,
                ) = construct_embd_inputs(
                    embds_imgs,
                    embds_caps,
                    embds_objs,
                    dummy_token_index_cap,
                    dummy_token_index_obj,
                    tokens_caps,
                    tokens_objs,
                    tokenizer,
                )
                input_embds_caps = input_embds_caps.to(device)
                input_embds_objs = input_embds_objs.to(device)

                logits_caps = model.module.language_decoder.forward(
                    input_embds_caps,
                    dummy_token_index=dummy_token_index_cap,
                    offset=n_img_tokens,
                    prefix_image_tok_embeds=args.prefix_image_tok_embeds,
                )
                logits_objs = model.module.language_decoder.forward(
                    input_embds_objs,
                    dummy_token_index=dummy_token_index_obj,
                    offset=n_img_tokens,
                    input_tokens=input_tokens_objs,
                    prefix_image_tok_embeds=args.prefix_image_tok_embeds,
                    decouple_label_tok_embeds=args.decouple_label_tok_embeds,
                    is_train=True,
                )

                # shift
                n_vocab = logits_caps.shape[-1]
                shift_logits_caps = logits_caps[..., :-1, :]
                shift_logits_objs = logits_objs[..., :-1, :]
                shift_target_caps = target_caps[..., 1:]
                shift_target_objs = target_objs[..., 1:]

                loss_caps = F.cross_entropy(
                    shift_logits_caps.reshape(
                        -1, n_vocab
                    ),  # [bs * max_seq_len, n_vocab]
                    shift_target_caps.reshape(-1),  # [bs * max_seq_len]
                    ignore_index=tokenizer.pad_id,
                )
                loss_objs = F.cross_entropy(
                    shift_logits_objs.reshape(-1, n_vocab),
                    shift_target_objs.reshape(-1),
                    ignore_index=tokenizer.pad_id,
                )

                loss_caps = loss_caps * args.weight_loss_cap
                loss_objs = loss_objs * args.weight_loss_obj

                loss = loss_caps + loss_objs
                loss = loss / grad_accum_steps

            scaler.scale(loss).backward()

            if i == 1 or i == len(dataloader) or i % grad_accum_steps == 0:
                if grad_clip > 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                if global_step > warmup_steps:
                    lr_scheduler.step()
                    last_lr = lr_scheduler.get_last_lr()[0]
                else:
                    _lr = args.lr * global_step / warmup_steps
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = _lr
                    last_lr = _lr

                global_step += 1

            # https://github.com/karpathy/nanoGPT/blob/master/train.py#L293
            model.require_backward_grad_sync = i % grad_accum_steps == 0

            batch_time = time.perf_counter() - t
            t = time.perf_counter()

            if i == 1 or i == len(dataloader) or i % args.log_interval == 0:
                warmup_tag = (
                    f"warmup step: {str(global_step).zfill(5)}"
                    if global_step <= warmup_steps
                    else ""
                )
                print(
                    f"epoch: {str(epoch).zfill(2)} ",
                    f"step: {str(global_step).zfill(8)} ",
                    f"lr: {last_lr:.7f} ",
                    f"loss: {loss.item() * grad_accum_steps:>10.7f}",
                    (
                        f"loss_caps: {loss_caps.item():>10.7f}"
                        if args.weight_loss_cap > 0.0
                        else ""
                    ),
                    f"loss_objs: {loss_objs.item():>10.7f}",
                    f"data time: {data_time:>8.5f}s ",
                    f"batch time: {batch_time:>8.5f}s ",
                    warmup_tag,
                )

            if (master_process and global_step % ckpt_save_interval == 0) or (
                master_process and global_step == total_steps
            ):
                if (global_step == pgs) or (
                    global_step == rgs and not args.from_scratch
                ):
                    continue
                save_checkpoint(
                    args, model, optimizer, lr_scheduler, epoch, global_step
                )
                pgs = global_step

    # TODO: save the last ckpt
    save_checkpoint(args, model, optimizer, lr_scheduler, epoch, global_step)

    destroy_process_group()


if __name__ == "__main__":
    cfg = load_config(sys.argv)
    main(cfg)
