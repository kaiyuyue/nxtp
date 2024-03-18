from typing import List

import time
import argparse
import os
import torch
import torch.onnx
import torch.onnx.verification
import onnx
import onnxruntime
import numpy as np

from collections import defaultdict
from PIL import Image
from loader import build_preprocess
from models.classifier import LangClassifier
from tokenizer import Tokenizer
from encoding import construct_text_inputs, construct_embd_inputs
from decoding import OneShotDecoder
from functions import load_llama, load_clip
from utils import load_config, set_dtype, load_checkpoint

assert torch.__version__.startswith("2")


def to_np(tensor):
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )


def onnx_export(ckpt_path: str, img_path: str):
    # load models
    cfg = load_config(["--config", "configs/config_g3m.py"]).args
    cfg.dtype = "float32"
    cfg = set_dtype(cfg)
    cfg.resume_ckpt_path = ckpt_path
    cfg.inference_mode = bool(1)

    # init folders to save onnx models
    onnx_folder = "onnx_models"
    os.makedirs(onnx_folder, exist_ok=True)

    # set device
    device = torch.device("cpu")

    # load image
    img = Image.open(img_path).convert("RGB")
    img = build_preprocess(cfg.input_size)(img)
    img = img.unsqueeze(0).to(device)

    # load models
    llama_model, tokenizer, model_args = load_llama(cfg, device)
    clip_model = load_clip(cfg, device)
    model = LangClassifier(vision_encoder=clip_model, language_decoder=llama_model)

    # eval mode
    load_checkpoint(cfg, model, strict=False, verbose=True)
    model.eval()

    # # -------------------------------------
    # # onnx export encoder
    # # -------------------------------------
    # encoder = model.vision_encoder.float()
    # with torch.no_grad():
    #     torch_out = encoder(img)

    # onnx_fn = os.path.join(onnx_folder, "encoder.onnx")
    # torch.onnx.export(
    #     encoder,
    #     img,
    #     onnx_fn,
    #     export_params=True,
    #     verbose=False,
    #     input_names=["input"],
    #     output_names=["output"],
    #     dynamic_axes={
    #         "input": [0],
    #         "output": [0],
    #     },
    # )
    # onnx.checker.check_model(onnx_fn)

    # ort_session = onnxruntime.InferenceSession(
    #     onnx_fn,
    #     providers=["CPUExecutionProvider"],
    # )

    # # compute ONNX Runtime output prediction
    # ort_inputs = {"input": to_np(img.repeat(2, 1, 1, 1))}
    # ort_outs = ort_session.run(None, ort_inputs)

    # # compare ONNX Runtime and PyTorch results
    # # np.testing.assert_allclose(to_np(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

    # # -------------------------------------
    # # onnx export Wte
    # # -------------------------------------
    # Wte = model.language_decoder.tok_embeddings.weight
    # onnx_fn = os.path.join(onnx_folder, "Wte.npz")
    # np.savez(onnx_fn, Wte=Wte.detach().cpu().numpy())

    # # -------------------------------------
    # # onnx export decoder
    # # -------------------------------------
    # decoder = model.language_decoder.float()
    # bs = 1
    # seqlen = 256
    # tensor_inp = torch.randn(bs, seqlen, 4096)
    # mask = torch.full((bs, 1, seqlen, seqlen), float("-inf"), device=device)
    # mask = torch.triu(mask, diagonal=1)

    # decoder_part = 1  # NOTE: please check out the forward function in models/lang.py
    # onnx_fn = os.path.join(onnx_folder, f"decoder_p{decoder_part}.onnx")
    # torch.onnx.export(
    #     decoder,
    #     (tensor_inp, mask),
    #     onnx_fn,
    #     export_params=True,
    #     verbose=False,
    #     input_names=[
    #         "input",
    #         "mask",
    #     ],
    #     output_names=["output"],
    #     dynamic_axes={
    #         "input": [0, 1],
    #         "mask": [0, 2, 3],
    #         "output": [0, 1],
    #     },
    # )
    # onnx.checker.check_model(onnx_fn)


def onnx_infer(ckpt_path: str, img_path: str, num_labels: int = 10):
    # load models
    cfg = load_config(["--config", "configs/config_g3m.py"]).args
    cfg.dtype = "float32"
    cfg = set_dtype(cfg)
    cfg.resume_ckpt_path = ckpt_path
    cfg.inference_mode = bool(1)

    # init folders to save onnx models
    onnx_folder = "onnx_models"
    os.makedirs(onnx_folder, exist_ok=True)

    # set device
    device = torch.device("cpu")

    # load image
    img = Image.open(img_path).convert("RGB")
    img = build_preprocess(cfg.input_size)(img)
    img = img.unsqueeze(0).to(device)

    def to_np(tensor):
        return (
            tensor.detach().cpu().numpy()
            if tensor.requires_grad
            else tensor.cpu().numpy()
        )

    # load tokenizer
    tokenizer = Tokenizer(model_path=cfg.tokenizer_path)

    # load Wte
    Wte = np.load(os.path.join(onnx_folder, "Wte.npz"))["Wte"]
    Wte = torch.tensor(Wte, device=device)
    print(f"Wte: {Wte.shape}")

    # load onnx models
    encoder_infer_session = onnxruntime.InferenceSession(
        os.path.join(onnx_folder, "encoder.onnx"),
        providers=["CPUExecutionProvider"],
    )
    decoder_part1_session = onnxruntime.InferenceSession(
        os.path.join(onnx_folder, "decoder_p1.onnx"),
        providers=["CPUExecutionProvider"],
    )
    decoder_part2_session = onnxruntime.InferenceSession(
        os.path.join(onnx_folder, "decoder_p2.onnx"),
        providers=["CPUExecutionProvider"],
    )
    decoder_part3_session = onnxruntime.InferenceSession(
        os.path.join(onnx_folder, "decoder_p3.onnx"),
        providers=["CPUExecutionProvider"],
    )

    # compute ONNX Runtime output prediction
    t1 = time.perf_counter()

    x = {"input": to_np(img)}
    z = encoder_infer_session.run(None, x)[0]
    z = torch.tensor(z, device=device)

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
    ) = construct_text_inputs(cfg, caps, tokenizer, offset=n_img_tokens, is_train=False)
    tokens_caps = tokens_caps.to(device)
    tokens_objs = tokens_objs.to(device)

    # convert tokens to embeddings
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

    # start sampling
    x = input_embds

    def construct_mask(x):
        bs, seqlen = x.shape[:2]
        start_pos = 0
        mask = torch.full((bs, 1, seqlen, seqlen), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=start_pos + 1)

        # prefix the image token embeddings
        # we compute attention across all image token embeddings
        ii = 0
        ij = 0 + n_img_tokens
        mask[:, :, ii:ij, ii:ij] = 0.0
        return mask

    mask = construct_mask(x)
    mask_np = to_np(mask)

    # run decoder
    h = decoder_part1_session.run(None, {"input": to_np(x), "mask": mask_np})[0]
    h = decoder_part2_session.run(None, {"input": h, "mask": mask_np})[0]
    h = decoder_part3_session.run(None, {"input": h, "mask": mask_np})[0]
    logits = torch.tensor(h, device=device)

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
        mask = construct_mask(x)
        mask_np = to_np(mask)

        h = decoder_part1_session.run(None, {"input": to_np(x), "mask": mask_np})[0]
        h = decoder_part2_session.run(None, {"input": h, "mask": mask_np})[0]
        h = decoder_part3_session.run(None, {"input": h, "mask": mask_np})[0]
        logits = torch.tensor(h, device=device)

        tokens, completed = text_decoder.update(tokens, logits, sum_logprobs)
        next_tokens = tokens[:, -1].unsqueeze(1).long()

    # finalize the tokens and logprobs
    tokens, sum_logprobs = text_decoder.finalize(tokens, sum_logprobs)

    # wrap up
    pred_probs = torch.nested.as_nested_tensor(
        [torch.tensor(p) for p in sum_logprobs]
    ).to(device)
    pred_tokens = torch.nested.as_nested_tensor([torch.tensor(t) for t in tokens]).to(
        device
    )

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
    parser.add_argument("--img-path", type=str, required=False)
    parser.add_argument("--num-labels", type=int, default=20)
    parser.add_argument("--onnx-export", action="store_true")
    args = parser.parse_args()

    if args.onnx_export:
        onnx_export(args.ckpt_path, args.img_path)
        exit(0)
    onnx_infer(args.ckpt_path, args.img_path, args.num_labels)
