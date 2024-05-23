from typing import List

import argparse
import os
import time
import torch
import matplotlib.pyplot as plt

from tqdm import tqdm
from collections import defaultdict
from PIL import Image
from loader import build_preprocess
from models.classifier import LangClassifier
from encoding import construct_text_inputs, construct_embd_inputs
from decoding import OneShotDecoder
from functions import load_llama, load_clip
from utils import load_config, set_dtype, load_checkpoint


@torch.inference_mode()
def main(
    ckpt_path: str,
    img_path: str,
    num_labels: int = 10,
    save_attention_map: bool = False,
):
    # load config
    cfg = load_config(["--config", "configs/config_g3m.py"]).args
    cfg = set_dtype(cfg)
    cfg.resume_ckpt_path = ckpt_path
    cfg.inference_mode = bool(1)

    # set device
    device = torch.device("cuda")

    # load models
    llama_model, tokenizer, model_args = load_llama(cfg, device)
    clip_model = load_clip(cfg, device)
    model = LangClassifier(vision_encoder=clip_model, language_decoder=llama_model)
    model = model.to(device)

    # load ckpt
    load_checkpoint(cfg, model, strict=False)
    model.eval()

    # show params
    total_params = sum(param.numel() for param in model.parameters()) / 1e6
    print(f"total params: {total_params:.2f} M, {total_params / 1e3:.2f} B")

    # ctx manager
    ctx = torch.amp.autocast(device_type="cuda", dtype=cfg.ptdtype)

    # load image
    img = Image.open(img_path).convert("RGB")
    img = build_preprocess(cfg.input_size)(img)
    img = img.unsqueeze(0).to(device)

    # infer
    t1 = time.perf_counter()
    with ctx:
        # get image token embeddings
        h = model.encode_images(img)
        z = model.decode_images(h)

        # drop [CLS] token embedding
        embds_clss, embds_imgs = z[:, :1], z[:, 1:]
        bs, n_img_tokens = embds_imgs.shape[:2]

        # convert text to tokens
        caps = ["" for _ in range(bs)]  # means no reference labels in prompt
        (
            tokens_caps,
            tokens_objs,
            _,
            _,
            dummy_token_index_cap,
            dummy_token_index_obj,
        ) = construct_text_inputs(
            cfg, caps, tokenizer, offset=n_img_tokens, is_train=False
        )
        tokens_caps = tokens_caps.to(device)
        tokens_objs = tokens_objs.to(device)

        # convert tokens to embeddings
        Wte = model.language_decoder.tok_embeddings.weight
        embds_caps = Wte[tokens_caps]
        embds_objs = Wte[tokens_objs]

        _, input_embds_objs, input_tokens_objs = construct_embd_inputs(
            embds_imgs,
            embds_caps,
            embds_objs,
            dummy_token_index_cap,
            dummy_token_index_obj,
            tokens_caps,
            tokens_objs,
            tokenizer,
        )

        # shave padding tokens
        shave_ind = torch.where(tokens_objs == tokenizer.pad_id)[1][0]
        input_tokens = input_tokens_objs[:, : shave_ind + n_img_tokens]
        input_embds = input_embds_objs[:, : shave_ind + n_img_tokens]

        # init text decoder for sampling
        text_decoder = OneShotDecoder(k=num_labels)
        text_decoder.reset()

        # init output tokens and logprobs
        tokens = tokens_objs[:, :shave_ind]  # will be final output tokens
        sum_logprobs = torch.zeros(bs, device=device)

        # visualize attention maps
        cached_tensors = dict() if save_attention_map else None

        # start sampling
        x = input_embds
        logits = model.language_decoder.forward(
            x,
            start_pos=0,
            dummy_token_index=dummy_token_index_obj,
            offset=n_img_tokens,
            input_tokens=input_tokens,
            prefix_image_tok_embeds=cfg.prefix_image_tok_embeds,
            decouple_label_tok_embeds=cfg.decouple_label_tok_embeds,
            is_train=False,
            cached_tensors=cached_tensors,
        )

        if save_attention_map:
            for k in cached_tensors.keys():
                if not "attn" in k:
                    continue

                # visualize relatively shallow layers in the decoder
                # if not "layer_idx_0" in k:
                #     continue

                print(f"visualizing attention map for {k}")
                attn_map = cached_tensors[k]

                # extract the attention map for image tokens
                ii = dummy_token_index_obj
                ij = dummy_token_index_obj + n_img_tokens
                attn_map = attn_map[:, :, ii:ij, ii:ij]

                # attention head index: 0-31
                for head_idx in tqdm(range(attn_map.shape[1]), leave=False):
                    # save attention map
                    fig, ax = plt.subplots(16, 16, figsize=(11, 11))
                    maps = attn_map[0, head_idx]
                    for i in range(attn_map.shape[2]):
                        _map = maps[i].reshape(16, 16)
                        _map = _map.detach().cpu().numpy()
                        _map = (_map - _map.min()) / (_map.max() - _map.min() + 1e-6)
                        ax[i // 16, i % 16].imshow(_map, cmap="Blues")
                        ax[i // 16, i % 16].axis("off")
                    plt.tight_layout()
                    os.makedirs("figs", exist_ok=True)
                    plt.savefig(f"figs/attn_map_{k}_head_idx_{head_idx}.png")
                    plt.close()

        # get the initial tokens after the first forward pass
        tokens, completed = text_decoder.update(tokens, logits, sum_logprobs)
        next_tokens = tokens[:, -1].unsqueeze(1)

        # continue sampling until all labels reach [SEP]
        while completed == False:
            if x.shape[0] != next_tokens.shape[0]:
                assert next_tokens.shape[0] % x.shape[0] == 0
                x = x.repeat_interleave(next_tokens.shape[0] // x.shape[0], dim=0)

            # here we don't use the kv-attention for computing attention
            # if needed, can be added in the future
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
                dummy_token_index=dummy_token_index_obj,
                offset=n_img_tokens,
                input_tokens=input_tokens,
                prefix_image_tok_embeds=cfg.prefix_image_tok_embeds,
                decouple_label_tok_embeds=cfg.decouple_label_tok_embeds,
                is_train=False,
            )

            tokens, completed = text_decoder.update(tokens, logits, sum_logprobs)
            next_tokens = tokens[:, -1].unsqueeze(1).long()

        # finalize the tokens and logprobs
        tokens, sum_logprobs = text_decoder.finalize(tokens, sum_logprobs)

        # wrap up
        pred_probs = torch.nested.as_nested_tensor(
            [torch.tensor(p) for p in sum_logprobs]
        ).to(device)
        pred_tokens = torch.nested.as_nested_tensor(
            [torch.tensor(t) for t in tokens]
        ).to(device)

        # convert tokens to labels
        batch_preds: List[List[str]] = []
        batch_probs: List[List[float]] = []

        for i in range(bs):
            current_probs = pred_probs[i]
            current_tokens = pred_tokens[i]

            probs_per_label = []
            token_per_label = []

            current_pred_tokens = defaultdict(list)
            current_pred_labels = defaultdict(list)

            # group tokens by the dilimiter
            for prob, token in zip(current_probs, current_tokens):
                if token != 29892:  # delimiter ","
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
            current_preds, current_scores = [], []
            for v in sorted_current_pred_labels:
                label, score = v
                current_preds.append(label.replace(",", ""))  # remove the delimiter
                current_scores.append(round(score, 5))

            batch_preds.append(current_preds)
            batch_probs.append(current_scores)

        t2 = time.perf_counter()

        batch_preds = batch_preds[0]
        batch_probs = batch_probs[0]

        print(f"\ninference time: {(t2 - t1):.3f}s")
        print(f"top-{num_labels} predictions:")
        for pred, prob in zip(batch_preds, batch_probs):
            print(f"| prob: {prob:.5f} - {pred}")
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--img-path", type=str, required=True)
    parser.add_argument("--num-labels", type=int, default=10)
    parser.add_argument("--save-attention-map", type=bool, default=False)
    args = parser.parse_args()

    main(args.ckpt_path, args.img_path, args.num_labels, args.save_attention_map)
