from typing import List

import time
import torch
import torch.nn.functional as F

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import destroy_process_group
from PIL import Image
from collections import defaultdict

from evals.engine import prepare, evaluate
from evals.engine import _TEST_DATASETS, _CLIP_MODELS

from encoding import construct_text_inputs, construct_embd_inputs
from decoding import GreedyDecoder, BeamSearchDecoder, OneShotDecoder

from train import load_llama, load_clip
from loader import build_preprocess
from models.classifier import LangClassifier
from utils import load_checkpoint


def main(cfg):
    # put all the settings here
    args = cfg.args

    # ------- local settings -------
    args.batch_size = 32

    # 0: cc3m, 1: coco, 2: openimages
    args.test_dataset = _TEST_DATASETS[1]

    args.k_for_topk = 10  # top-k results
    args.xk_for_one_shot_sampling = 0  # top-k + extra k results for one-shot sampling

    args.text_decoder_strategy = "one_shot"  # greedy | beam | one_shot
    args.ranking_method = "none"  # clip, sim, ppl, prob, none
    args.max_gen_len: int = 64

    args.sim_reduction = "mean"
    args.sim_eculidean_dist = True
    args.sim_max_reduction_with_fscore = False

    args.use_clip_to_rank_topk = True if args.ranking_method == "clip" else False
    args.normalize_clip_scores = False

    args.prompt_type = "list"
    args.benchmark_infer_time = True
    if args.benchmark_infer_time:
        args.batch_size = 1
        args.running_cnt: int = 1
        args.running_avg: float = 0.0

    args.inference_mode = bool(1)
    args.jupyter_mode = bool(0)  # disable ddp

    print(
        f"use clip to rank topk: {args.use_clip_to_rank_topk}\n"
        f"k_for_topk: {args.k_for_topk}\n"
        f"xk_for_one_shot_sampling: {args.xk_for_one_shot_sampling}\n"
        f"ranking_method: {args.ranking_method}\n"
        f"text_decoder_strategy: {args.text_decoder_strategy}"
    )
    if args.ranking_method == "sim":
        print(
            f"\n"
            f"sim_reduction: {args.sim_reduction}\n"
            f"sim_eculidean_dist: {args.sim_eculidean_dist}\n"
            f"sim_max_reduction_with_fscore: {args.sim_max_reduction_with_fscore}"
        )

    if args.text_decoder_strategy == "beam" and (
        args.ranking_method not in ("sim", "clip", "none")
    ):
        raise NotImplementedError(
            f"beam search only supports ranking method: sim or clip"
        )
    print(f"eval on {args.test_dataset}")
    # ------------------------------

    args.eval_model_name = args.exp_code
    if args.xk_for_one_shot_sampling > 0:
        args.eval_model_name += f"_xk{args.xk_for_one_shot_sampling}"

    args.clip_model_for_eval = _CLIP_MODELS[3]
    args.use_open_clip = False

    # prepare
    (
        device,
        rank,
        ctx,
        metrics,
        loader,
        categories,
        clip_model,
        clip_tokenizer,
        args,
    ) = prepare(args)
    args.device = device

    # load lang classifier
    llama_model, tokenizer, _ = load_llama(args, device)
    model = LangClassifier(
        vision_encoder=load_clip(args, device), language_decoder=llama_model
    )
    model = model.to(device)
    if args.use_ddp:
        model = DDP(model, device_ids=[rank])
    model.eval()

    if args.eval_ckpt_path:
        args.resume_ckpt_path = args.eval_ckpt_path  # for loading the model
        load_checkpoint(args, model, optimizer=None, scheduler=None, strict=False)

    preprocess = build_preprocess(args, is_train=False)

    engine_func = engine_lang_classifier

    print("start evaluating...")
    evaluate(
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
        k_for_topk=args.k_for_topk,
    )

    if args.use_ddp:
        destroy_process_group()


"""
Functions
"""


def batch_process(paths, preprocess=None, device=None):
    imgs = [Image.open(p).convert("RGB") for p in paths]
    imgs = torch.stack([preprocess(img) for img in imgs], dim=0)
    imgs = imgs.to(device)
    return imgs


@torch.inference_mode()
def engine_lang_classifier(args, preprocess, model, tokenizer, imgs, paths=None):
    if args.text_decoder_strategy == "greedy":
        text_decoder = GreedyDecoder(
            temperature=args.temperature,
            eot=tokenizer.eos_id,
            k=args.top_k,
            p=args.top_p,
            func=args.greedy_func,
            penalty=args.penalty,
        )
    elif args.text_decoder_strategy == "beam":
        text_decoder = BeamSearchDecoder(
            eot=tokenizer.eos_id,
            beam_size=args.beam_size,
            patience=args.beam_patience,
            penalty=args.penalty,
        )
    elif args.text_decoder_strategy == "one_shot":
        k = args.k_for_topk + args.xk_for_one_shot_sampling
        text_decoder = OneShotDecoder(k=k)
    else:
        raise NotImplementedError

    # call reset() to initialize the decoder
    text_decoder.reset()

    device = args.device

    if args.benchmark_infer_time:
        t1 = time.perf_counter()

    h = model.encode_images(imgs)
    z = model.decode_images(h)

    embds_clss, embds_imgs = z[:, :1], z[:, 1:]
    bs, n_img_tokens = embds_imgs.shape[:2]

    caps = ["" for _ in range(bs)]
    (
        tokens_caps,
        tokens_objs,
        _,
        _,
        dummy_token_index_cap,
        dummy_token_index_obj,
    ) = construct_text_inputs(
        args, caps, tokenizer, offset=n_img_tokens, is_train=False
    )
    tokens_caps = tokens_caps.to(device)
    tokens_objs = tokens_objs.to(device)

    Wte = model.language_decoder.tok_embeddings.weight
    embds_caps = Wte[tokens_caps]
    embds_objs = Wte[tokens_objs]

    input_embds_caps, input_embds_objs, input_tokens_objs = construct_embd_inputs(
        embds_imgs,
        embds_caps,
        embds_objs,
        dummy_token_index_cap,
        dummy_token_index_obj,
        tokens_caps,
        tokens_objs,
        tokenizer,
    )

    pred_probs, pred_tokens = generate(
        args,
        n_img_tokens,
        dummy_token_index_obj,
        tokens_objs,
        input_embds_objs,
        input_tokens_objs,
        tokenizer,
        model,
        device,
        text_decoder,
    )

    objs = post_process(
        pred_probs,
        pred_tokens,
        embds_imgs,
        tokenizer,
        Wte,
        ranking_method=args.ranking_method,
        sim_reduction=args.sim_reduction,
        sim_euclidean_dist=args.sim_eculidean_dist,
        sim_max_reduction_with_fscore=args.sim_max_reduction_with_fscore,
    )

    if args.benchmark_infer_time:
        time_cnt = time.perf_counter() - t1
        args.running_avg = (
            args.running_avg * (args.running_cnt - 1) + time_cnt
        ) / args.running_cnt
        args.running_cnt += 1
        print(f"avg infer time: {args.running_avg:.3f} s")

    return objs


@torch.inference_mode()
def generate(
    args,
    n_img_tokens,
    dummy_token_index,
    input_tokens,
    input_embds,
    input_tokens_objs,
    tokenizer,
    model,
    device,
    text_decoder,
):
    # because all input samples have the same prompt and the same length of image embeddings,
    # here we only need to use one index for shaving out padding ids for all
    shave_ind = torch.where(input_tokens == tokenizer.pad_id)[1][0]

    min_prompt_size = shave_ind.item()
    max_prompt_size = shave_ind.item()

    total_len = min(args.max_seq_len, args.max_gen_len + max_prompt_size)

    bs = input_tokens.shape[0]
    Wte = model.language_decoder.tok_embeddings.weight

    input_embds = input_embds[:, : shave_ind + n_img_tokens]
    tokens = input_tokens[:, :shave_ind]  # will be final output tokens
    sum_logprobs = torch.zeros(bs, device=device)
    input_tokens_objs = input_tokens_objs[:, : shave_ind + n_img_tokens]

    if args.text_decoder_strategy == "beam":
        # repeat the input for beam search: [bs, n_embd] -> [bs * n_beam, n_embd]
        n_beam = args.beam_size
        input_embds = input_embds.repeat_interleave(n_beam, dim=0)
        tokens = tokens.repeat_interleave(n_beam, dim=0)
        sum_logprobs = sum_logprobs.repeat_interleave(n_beam, dim=0)
        input_tokens_objs = input_tokens_objs.repeat_interleave(n_beam, dim=0)

    pred_probs = []
    pred_tokens = []

    start_pos = min_prompt_size
    for cur_pos in range(start_pos, total_len):
        if cur_pos == start_pos:
            x = input_embds
        else:
            x = torch.cat(
                [
                    x,
                    Wte[next_tokens.long()],
                ],
                dim=1,
            )

        logits = model.language_decoder.forward(
            x,
            start_pos=0,
            dummy_token_index=dummy_token_index,
            offset=n_img_tokens,
            input_tokens=input_tokens_objs,
            prefix_image_tok_embeds=args.prefix_image_tok_embeds,
            decouple_label_tok_embeds=args.decouple_label_tok_embeds,
            is_train=False,
        )

        tokens, completed = text_decoder.update(tokens, logits, sum_logprobs)
        next_tokens = tokens[:, -1].unsqueeze(1)
        prob_next_tokens = text_decoder.prob_next_tokens

        # NOTE: shape of
        # tokens: torch.Size([64, 10]) -> torch.Size([64, 11]), ...
        # prob_next_tokens: torch.Size([64, 1])
        # next_tokens: torch.Size([64, 1])

        if args.text_decoder_strategy == "one_shot" and cur_pos == start_pos:
            # here only samples the first set of tokens
            input_tokens_objs = torch.cat(
                [
                    input_tokens_objs.repeat_interleave(
                        text_decoder.one_shot_size, dim=0
                    ),
                    next_tokens,
                ],
                dim=1,
            )
            break

        if args.text_decoder_strategy == "greedy":
            # update input tokens for updating attention mask
            input_tokens_objs = torch.cat([input_tokens_objs, next_tokens], dim=1)
            pred_tokens.append(next_tokens)
            pred_probs.append(prob_next_tokens)

        if completed:
            break

    if args.text_decoder_strategy == "one_shot":
        while completed == False:
            if x.shape[0] != next_tokens.shape[0]:
                assert next_tokens.shape[0] % x.shape[0] == 0
                x = x.repeat_interleave(next_tokens.shape[0] // x.shape[0], dim=0)

            x = torch.cat(
                [
                    x,
                    Wte[next_tokens],
                ],
                dim=1,
            )

            logits = model.language_decoder.forward(
                x,
                start_pos=0,
                dummy_token_index=dummy_token_index,
                offset=n_img_tokens,
                input_tokens=input_tokens_objs,
                prefix_image_tok_embeds=args.prefix_image_tok_embeds,
                decouple_label_tok_embeds=args.decouple_label_tok_embeds,
                is_train=False,
            )

            tokens, completed = text_decoder.update(tokens, logits, sum_logprobs)
            next_tokens = tokens[:, -1].unsqueeze(1).long()

    tokens, sum_logprobs = text_decoder.finalize(tokens, sum_logprobs)

    if args.text_decoder_strategy == "one_shot":
        pred_probs = torch.nested.as_nested_tensor(
            [torch.tensor(p) for p in sum_logprobs]
        ).to(device)
        pred_tokens = torch.nested.as_nested_tensor(
            [torch.tensor(t) for t in tokens]
        ).to(device)
    elif args.text_decoder_strategy == "beam":
        pred_tokens = torch.stack([torch.tensor(t) for t in tokens], dim=0).to(device)
        pred_probs = torch.zeros_like(pred_tokens).to(device)
    elif args.text_decoder_strategy == "greedy":
        pred_probs = torch.cat(pred_probs, dim=1)
        pred_tokens = torch.cat(pred_tokens, dim=1)
    else:
        raise NotImplementedError

    return pred_probs, pred_tokens


def post_process(
    pred_probs,
    pred_tokens,
    embds_imgs,
    tokenizer,
    Wte,
    ranking_method="sim",
    sim_reduction="mean",
    sim_euclidean_dist=False,
    sim_max_reduction_with_fscore=False,
):
    assert ranking_method in ("sim", "ppl", "prob", "clip", "none")
    assert sim_reduction in ("mean", "max")

    if ranking_method == "sim":
        # check the settings
        if sim_reduction == "mean" and sim_euclidean_dist is False:
            raise ValueError("cannot use mean reduction for cosine similarity")

    # NOTE: the following assert is not available for nested tensors
    #       so just ignore it for now
    # assert pred_probs.shape == pred_tokens.shape

    bs = embds_imgs.shape[0]

    batch_preds: List[List[str]] = []
    batch_probs: List[List[float]] = []

    for i in range(bs):
        current_embds_imgs = embds_imgs[i]
        current_probs = pred_probs[i]
        current_tokens = pred_tokens[i]

        probs_per_label = []
        token_per_label = []

        current_pred_tokens = defaultdict(list)
        current_pred_labels = defaultdict(list)

        # step 1: group tokens by the dilimiter
        for prob, token in zip(current_probs, current_tokens):
            if token != 29892:
                probs_per_label.append(prob)
                token_per_label.append(token.item())
            else:
                # include the delimiter score
                probs_per_label.append(prob)
                token_per_label.append(token.item())

                # compute the final score
                probs = torch.stack(probs_per_label)
                label = tokenizer.decode(token_per_label)

                current_pred_tokens[label].append(token_per_label)
                current_pred_labels[label].append(probs)

                probs_per_label = []
                token_per_label = []

        # step 2: compute the similarity between image tokens and label tokens
        if ranking_method == "sim":
            # Eq. A.4 and A.5 in the paper
            current_pred_sim = {}
            for label, tokens in current_pred_tokens.items():
                sim_per_label = []
                for group_tokens in tokens:
                    embds_label = torch.stack([Wte[t] for t in group_tokens], dim=0)
                    v = F.normalize(current_embds_imgs, dim=-1)  # [m, d]
                    t = F.normalize(embds_label, dim=-1)  # [n, d]
                    M = torch.einsum("nd,md->nm", t, v)  # [n, m]

                    if sim_euclidean_dist:
                        # euclidean distance in [0, 1], [sim, dissim]
                        M = torch.sqrt(2 - 2 * M) / 2
                        sim_reverse = False
                    else:
                        # cosine similarity in [-1, 1], [dissim, sim]
                        sim_reverse = True

                    if sim_reduction == "max":
                        Rt = M.max(dim=1).values.mean()
                        if sim_max_reduction_with_fscore:
                            Pi = M.max(dim=0).values.mean()
                            sim_score = 2 * Pi * Rt / (Pi + Rt)
                        else:
                            sim_score = Rt
                    elif sim_reduction == "mean":
                        sim_score = M.mean()
                    else:
                        raise NotImplementedError
                    sim_per_label.append(sim_score)

                # multiple groups of tokens for the same label
                # we stack them together and compute the mean for each label
                sim_per_label = torch.stack(sim_per_label).mean()
                current_pred_sim[label] = sim_per_label.item()

            # higher value means more similar for cosine similarity
            # lower value means more similar for euclidean distance
            sorted_current_pred_labels = sorted(
                current_pred_sim.items(), key=lambda x: x[1], reverse=sim_reverse
            )
        elif ranking_method == "ppl":
            # Eq. A.3 in the paper
            current_pred_ppl = {}
            for label, tokens in current_pred_tokens.items():
                probs = current_pred_labels[label]
                # multiple groups of tokens for the same label
                # we stack them together and select the one with the lowest ppl
                ppls = torch.stack([p.log().mean(dim=-1).exp() for p in probs], dim=0)
                ppl_per_label = ppls.min()  # min over all groups
                current_pred_ppl[label] = ppl_per_label.item()

            # lower perplexity is better
            sorted_current_pred_labels = sorted(
                current_pred_ppl.items(), key=lambda x: x[1], reverse=False
            )
        elif ranking_method == "prob":
            # Eq. A.2 in the paper
            current_pred_prob = {}
            for label, tokens in current_pred_tokens.items():
                probs = current_pred_labels[label]
                # multiple groups of tokens for the same label
                # we stack them together and compute the sum for each group
                probs = torch.stack([p.prod() for p in probs], dim=0)
                prob_per_label = probs.sum()  # sum over all groups
                current_pred_prob[label] = prob_per_label.item()

            # higher probability is better
            sorted_current_pred_labels = sorted(
                current_pred_prob.items(), key=lambda x: x[1], reverse=True
            )
        elif ranking_method == "clip":
            # Eq. A.1 in the paper
            current_pred_clip_score = {}
            for label, tokens in current_pred_tokens.items():
                current_pred_clip_score[
                    label
                ] = 0.0  # will have it later in the evals.engine function
            sorted_current_pred_labels = sorted(
                current_pred_clip_score.items(), key=lambda x: x[1], reverse=True
            )
        elif ranking_method == "none":
            current_pred_none_score = {}
            for label, tokens in current_pred_tokens.items():
                current_pred_none_score[label] = 0.0
            # keep the original order without sorting
            sorted_current_pred_labels = current_pred_none_score.items()
        else:
            raise NotImplementedError

        current_preds, current_scores = [], []
        for v in sorted_current_pred_labels:
            label, score = v
            current_preds.append(label.replace(",", ""))  # remove the delimiter
            current_scores.append(round(score, 5))

        batch_preds.append(current_preds)
        batch_probs.append(current_scores)

    return batch_preds, batch_probs
