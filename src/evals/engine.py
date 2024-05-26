from typing import List

import time
import torch
import clip
import torch.distributed as dist
import torch.nn.functional as F

from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from functools import partial
from random import shuffle

from evals.metrics import SemanticFScore
from utils import init_nltk, set_dtype, setup_model_parallel, setup_for_distributed
from utils import get_noun_words

"""
Vars
"""

_TEST_DATASETS = {
    0: "cc3m_valid",
    1: "coco_valid",
    2: "openimages_v7_valid",
    3: "in1k_valid",
}

_CLIP_MODELS = {
    0: "RN50",
    1: "ViT-B/32",
    2: "ViT-B/16",
    3: "ViT-L/14",
}


"""
Loader
"""


def init_loader(args, rank, world_size, transform=None):
    if "in1k" in args.test_dataset:
        from imagenet.loader import build_dataloader

        print("loading ImageNet ...")
    else:
        from loader import build_dataloader

    args.data_name = [args.test_dataset]
    loader = build_dataloader(args, rank, world_size, is_train=False)
    categories = []
    return loader, categories


"""
Functions
"""


@torch.inference_mode()
def knn_classifier(q, v, scale=1.0, topk=1, normalize=True):
    """
    k-Nearest Neighbor Classifier with CLIP feature embeddings.
    """
    if q.ndim != v.ndim:
        raise ValueError(f"q.ndim != v.ndim: {q.ndim} != {v.ndim}")

    q = q / q.norm(dim=-1, keepdim=True)  # [1, D]
    v = v / v.norm(dim=-1, keepdim=True)  # [N, D]

    x = scale * q @ v.T  # [1, N]
    if normalize:
        x = x.softmax(dim=-1)

    k = min(topk, x.shape[-1])  # in case k > probs.shape[-1]
    probs, indices = x.topk(k, dim=-1)

    return probs, indices


@torch.inference_mode()
def encode_text_with_clip(clip_model, clip_tokenizer, objs, device=None):
    """
    Encode a text into CLIP feature embeddings.

    Args:
        objs (List[str]): List of object labels.
    """
    prompts = ["{}"]
    clip_model = clip_model.module if hasattr(clip_model, "module") else clip_model

    # tokenize
    objs: List[str] = [c.replace("_", " ") for c in objs]  # [N]
    objs_w_prompts: List[List[str]] = [
        [p.format(v) for v in objs] for p in prompts
    ]  # [[N]]

    # shave
    objs_w_prompts = [
        [voc_w_prompt[:77] for voc_w_prompt in objs_w_prompts]
        for objs_w_prompts in objs_w_prompts
    ]  # [[N]]

    objs_tokens = [
        torch.cat([clip_tokenizer(voc_w_prompt) for voc_w_prompt in voc_w_prompts])
        for voc_w_prompts in objs_w_prompts
    ]  # [N, 77]
    objs_tokens = torch.stack(objs_tokens).to(device)  # [1, N, 77]

    # encode
    T, C, _ = objs_tokens.shape
    objs_embds = clip_model.encode_text(objs_tokens.view(T * C, -1))
    objs_embds = objs_embds.view(T, C, -1)

    return objs_embds


def encode_cap_to_objs(args, caps: List[str]) -> List[str]:
    if args.skip_extract_nouns:
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
        objs[i] = obj
    return objs


def name_logging_file(args, with_ranks=False, merge_ranks=False) -> str:
    k_for_topk = args.k_for_topk

    if args.ranking_method == "clip" or args.use_clip_to_rank_topk:
        ranking_method = "clip"
        if args.normalize_clip_scores:
            ranking_method += "_norm"
        else:
            ranking_method += "_raw"
    else:
        ranking_method = args.ranking_method

    if with_ranks:
        return (
            f"{args.test_dataset}"
            + f"_rank-{args.rank}"
            + f"_{args.eval_model_name}"
            + f"_prompt_{args.prompt_type}"
            + f"_{args.text_decoder_strategy}"
            + f"_gen_{args.max_gen_len}_tokens"
            + f"_{ranking_method}"
            + f"_top{k_for_topk}"
            + ".res"
        )

    if merge_ranks:
        return (
            f"{args.test_dataset}"
            + f"_rank-*"  # merge all results from different ranks
            + f"_{args.eval_model_name}"
            + f"_prompt_{args.prompt_type}"
            + f"_{args.text_decoder_strategy}"
            + f"_gen_{args.max_gen_len}_tokens"
            + f"_{ranking_method}"
            + f"_top{k_for_topk}"
            + ".res"
        )

    return (
        f"{args.test_dataset}"
        + f"_{args.eval_model_name}"
        + f"_prompt_{args.prompt_type}"
        + f"_{args.text_decoder_strategy}"
        + f"_gen_{args.max_gen_len}_tokens"
        + f"_{ranking_method}"
        + f"_top{k_for_topk}"
        + ".res"
    )


"""
CLIP
"""


def load_open_clip(args):
    import open_clip

    device = args.device
    model_name = args.clip_model_for_eval.replace("/", "-")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained="openai",
        device="cpu",
    )
    model.to(device)
    tokenizer = open_clip.get_tokenizer(model_name)
    return preprocess, model, tokenizer


def load_clip(args):
    device = args.device
    model, preprocess = clip.load(
        args.clip_model_for_eval,
        device="cpu",
        download_root=args.clip_dir,
        jit=False,
    )
    model = model.to(device)
    tokenizer = clip.tokenize
    return preprocess, model, tokenizer


"""
Engines
"""


def prepare(args):
    # nltk
    init_nltk(download_dir=f"{args.cache_root}/nltk_data")

    # ddp
    if (
        torch.cuda.is_available()
        and torch.cuda.device_count() > 0
        and (not args.jupyter_mode)  # for jupyter notebook
    ):
        args.use_ddp = True
        rank, global_rank, world_size, device = setup_model_parallel(
            seed=args.seed, mute_non_master_ranks=False
        )
        args.master_process = global_rank == 0
        args.device = device
        args.rank = rank
        setup_for_distributed(args.master_process)
    else:
        args.use_ddp = False  # found a little bit slower with ddp on a single gpu
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.master_process = True
        args.device = device
        args.rank = rank
    print(f"enable ddp: {args.use_ddp}")

    # metrics
    metrics = {
        "sscr": SemanticFScore(
            model_name=args.eval_embedding_model,
            max_seq_len=args.max_seq_len,
        ).to(device),
    }

    # dtype
    args = set_dtype(args)

    # amp
    ctx = torch.amp.autocast(device_type="cuda", dtype=args.ptdtype)
    enable_amp = args.dtype == "float16"
    print(f"enable amp: {enable_amp}")

    # clip for evaluation
    print("start loading clip...")
    if args.use_open_clip:
        clip_preprocess, clip_model, clip_tokenizer = load_open_clip(args)
    else:
        clip_preprocess, clip_model, clip_tokenizer = load_clip(args)
    if args.use_ddp:
        clip_model = DDP(clip_model, device_ids=[rank], broadcast_buffers=False)
    clip_model.eval()

    # loader
    _func_loader = init_loader
    loader, categories = _func_loader(
        args,
        global_rank,
        world_size,
        transform=clip_preprocess,
    )

    return (
        device,
        rank,
        ctx,
        metrics,
        loader,
        categories,
        clip_model,
        clip_tokenizer,
        args,
    )


@torch.inference_mode()
def evaluate(
    args,
    ctx,
    engine_func,
    loader,
    preprocess,
    batch_process,
    model,
    tokenizer,
    clip_model,
    clip_tokenizer,
    categories,
    metrics,
    k_for_topk: int = 1,
):
    args.k_for_topk = k_for_topk

    # for logging results with a long file name
    fn = name_logging_file(args, with_ranks=True)
    fio = open(f"{args.results_dir}/{fn}", "w")

    # this is for our models, which do not use clip
    model = model.module if hasattr(model, "module") else model
    model.eval()
    device = args.device

    # report params size
    total_params = sum(param.numel() for param in model.parameters()) / 1e6
    print(f"total params: {total_params:.2f} M, {total_params / 1e3:.2f} B")

    # clip logit scale factor
    clip_model = clip_model.module if hasattr(clip_model, "module") else clip_model
    logit_scale = clip_model.logit_scale.exp().to(device, non_blocking=True)

    pbar = tqdm(enumerate(loader), total=len(loader))
    for i, batch in pbar:
        imgs, caps, keys = batch  # imgs are augmented by clip preprocess
        imgs = imgs.to(device, non_blocking=True)
        bs = imgs.shape[0]
        paths = [p.replace(".tar", "") + ".jpg" for p in keys]

        with ctx:
            """
            output format:
                for each image, a list of object labels, i.e.,
                batch_preds = List[Lisrt[str]]
                batch_probs = List[Lisrt[float]]

            Those predictions are already sorted by our own method.
            If args.use_clip_to_rank_topk is True, the predictions will be further ranked by CLIP.
            """
            outputs = engine_func(args, preprocess, model, tokenizer, imgs, paths=paths)

            if isinstance(outputs, tuple) and len(outputs) == 2:  # our models
                batch_preds: List[List[str]] = outputs[0]
                batch_probs: List[List[float]] = outputs[1]
            elif isinstance(outputs, list) and len(outputs) == bs:
                batch_preds: List[List[str]] = outputs
                batch_probs: List[List[float]] = [[0.0] * len(p) for p in batch_preds]
            else:
                raise ValueError(f"unknown outputs: {outputs}")

            # each sample has multiple ground truths,
            ts: List[List[str]] = encode_cap_to_objs(args, caps)

            topk_objs = []
            raw_probs = []
            if args.use_clip_to_rank_topk:
                imgs = F.interpolate(
                    imgs,
                    size=(224, 224),
                    mode="bilinear",
                    align_corners=False,
                )
                x_imgs = clip_model.encode_image(imgs)

                for x_img, _obj in zip(x_imgs, batch_preds):
                    # rm duplicated vocs
                    _obj = list(set(_obj))
                    if len(_obj) == 0:
                        _obj = [""]  # dummy voc

                    x_obj = encode_text_with_clip(
                        clip_model, clip_tokenizer, _obj, device=device
                    )  # [1, N, D]

                    x_img = x_img.unsqueeze(0)  # [1, D]
                    x_obj = x_obj.squeeze(0)  # [N, D], where N = len(_obj)

                    probs, indices = knn_classifier(
                        x_img, x_obj, scale=logit_scale, topk=k_for_topk
                    )
                    preds = [_obj[ind] for ind in indices[0]]
                    topk_objs.append(preds)
                    raw_probs.append(probs.tolist()[0])
            else:
                for p in batch_preds:
                    p = p[:k_for_topk]
                    if len(p) == 0:
                        p = [""]  # dummy voc
                    topk_objs.append(p)

            ps: List[List[str]] = topk_objs
            print(ts, "-------", ps)

            vals = {}
            for n, metric in metrics.items():
                metric.update(ps, ts)
                r, p, f, vals_std = metric.compute()
                vals[f"{n}_R"] = r.item()
                vals[f"{n}_P"] = p.item()
                vals[f"{n}_F"] = f.item()
                for n_std, v_std in vals_std.items():
                    vals[f"{n}_{n_std}"] = v_std.item()

            pbar.set_postfix(
                **vals,
                refresh=False,
            )

            # logging results
            metric = metrics["sscr"]
            scores: List[List[float]] = metric.scores  # [references [predictions]]
            div_scores: List[List[float]] = metric.div_scores  # [predictions!]

            # write results
            for k in range(len(ps)):
                # predictions
                fio.write(f"path: {paths[k]}\n")
                fio.write(f"tgts: {ts[k]}\n")
                fio.write(f"vocs: ")
                for p, prob in zip(batch_preds[k], batch_probs[k]):
                    fio.write(f"{p}\t{round(prob, 4)},\t")
                fio.write("\n")

                # clip scores
                if args.use_clip_to_rank_topk:
                    # also record the results after CLIP ranking
                    fio.write("clip: ")
                    for p, prob in zip(ps[k], raw_probs[k]):
                        fio.write(f"{p}\t{round(prob, 4)},\t")
                    fio.write("\n")

                # metrics
                fio.write(
                    f"scores: {scores[k]}\n"
                )  # bert scores between predictions and references
                fio.write(
                    f"div_scores: {div_scores[k]}\n"
                )  # bert scores between predictions for the same image

                fio.write("\n---\n")

    print("final results:")
    vals = {}
    for n, metric in metrics.items():
        r, p, f, vals_std = metric.compute()
        print(f"- {r.item():.5f} : {n}_R")
        print(f"- {p.item():.5f} : {n}_P")
        print(f"- {f.item():.5f} : {n}_F")
        for n_std, v_std in vals_std.items():
            print(f"- {v_std.item():.5f} : {n}_{n_std}")

    fio.close()

    # gather results
    if args.use_ddp:
        dist.barrier()
        if args.rank == 0:
            import glob
            import os

            fn = name_logging_file(args, merge_ranks=True)
            fnames = f"{args.results_dir}/{fn}"

            res_files = glob.glob(fnames)
            res_files.sort()

            fn = name_logging_file(args)
            fn_path = f"{args.results_dir}/{fn}"
            with open(fn_path, "w") as fio:
                for res_file in res_files:
                    with open(res_file, "r") as f:
                        fio.write(f.read())
                    os.remove(res_file)
            print(f"merged results: {fn_path}")
