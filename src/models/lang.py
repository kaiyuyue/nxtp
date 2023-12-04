from typing import Tuple, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


""" 
LLaMA w/ ddp
"""


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    max_seq_len: int = 2048


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    if freqs_cis.ndim == 2:
        assert freqs_cis.shape == (x.shape[1], x.shape[-1])
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    elif freqs_cis.ndim == 3:
        assert freqs_cis.shape == (x.shape[0], x.shape[1], x.shape[-1])
        shape = [
            d if i == 0 or i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)
        ]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.n_head = args.n_heads
        self.n_embd = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.n_embd, bias=False)
        self.wk = nn.Linear(args.dim, args.n_heads * self.n_embd, bias=False)
        self.wv = nn.Linear(args.dim, args.n_heads * self.n_embd, bias=False)
        self.wo = nn.Linear(args.n_heads * self.n_embd, args.dim, bias=False)

    def forward(self, x, start_pos, freqs_cis, mask):
        bs, seqlen, _ = x.shape
        q, k, v = self.wq(x), self.wk(x), self.wv(x)

        q = q.view(bs, seqlen, self.n_head, self.n_embd)
        k = k.view(bs, seqlen, self.n_head, self.n_embd)
        v = v.view(bs, seqlen, self.n_head, self.n_embd)

        q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.n_embd)

        if mask is not None:
            attn = attn + mask
        attn = F.softmax(attn, dim=-1)

        x = torch.matmul(attn, v)  # (bs, n_head, slen, n_embd)
        x = x.transpose(1, 2).contiguous().view(bs, seqlen, -1)

        return self.wo(x)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, multiple_of):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id, args):
        super().__init__()
        self.n_head = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim, hidden_dim=4 * args.dim, multiple_of=args.multiple_of
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, start_pos, freqs_cis, mask):
        h = x + self.attention.forward(
            self.attention_norm(x), start_pos, freqs_cis, mask
        )
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class LLaMATransformer(nn.Module):
    """
    https://github.com/facebookresearch/llama/blob/main/llama/model.py#L198
    """

    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layer = params.n_layers

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)

        self.layers = torch.nn.ModuleList()
        if params.shave_language_decoder_at > 0:
            self.n_layer = min(params.shave_language_decoder_at, self.n_layer)

        for layer_id in range(self.n_layer):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )
        self.decouple_freqs_cis = False

    def forward(
        self,
        tokens,
        start_pos=0,
        dummy_token_index=0,
        offset=0,
        input_tokens=None,
        prefix_image_tok_embeds=False,
        decouple_label_tok_embeds=False,
        is_train=True,
    ):
        # tokens are ids if ndim == 2 or embeddings if ndim == 3
        bs, seqlen = tokens.shape[:2]
        h = self.tok_embeddings(tokens) if tokens.dim() == 2 else tokens

        self.freqs_cis = self.freqs_cis.to(h.device)  # torch.Size([256, 64])
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            # if we uncondition predicted tokens, each sample has its own mask
            # because of different lengths and delimiter positions
            if decouple_label_tok_embeds:
                bs_mask = bs
                if self.decouple_freqs_cis:
                    freqs_cis_bs = freqs_cis.unsqueeze(0).repeat(bs_mask, 1, 1)
            else:
                bs_mask = 1

            mask = torch.full(
                (bs_mask, 1, seqlen, seqlen), float("-inf"), device=tokens.device
            )
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

            if prefix_image_tok_embeds:
                # we compute attention across all image token embeddings
                ii = dummy_token_index
                ij = dummy_token_index + offset
                mask[:, :, ii:ij, ii:ij] = 0.0

            if decouple_label_tok_embeds:
                """
                token id 526 for "are", the last "are" in the prompt
                token id 29892 for the delimiter ","
                """
                # the last "are" is the answer begining
                answer_index = torch.nonzero(input_tokens[-1] == 526)[0][0]
                assert (
                    answer_index > dummy_token_index
                )  # make sure won't erase the image tokens
                delimiters_indices = torch.nonzero(input_tokens == 29892)

                if is_train:
                    # erase masks for tokens after the answer position
                    mask[:, :, :, answer_index + 1 :] = float("-inf")

                # seprate masks for each delimiter
                prev_batch_index = 0
                prev_token_index = answer_index

                for delimiter in delimiters_indices:
                    batch_index, token_index = delimiter

                    # the delimiter token is conditioned on the previous label tokens
                    token_index += 1

                    if batch_index != prev_batch_index:
                        # for the next sample
                        prev_batch_index = batch_index
                        prev_token_index = answer_index

                    if is_train:
                        tri_size = token_index - prev_token_index
                        tri_mask = prev_token_index + torch.tril_indices(
                            tri_size, tri_size, device=mask.device
                        )
                        mask[batch_index, 0, tri_mask[0], tri_mask[1]] = 0.0
                        if self.decouple_freqs_cis:
                            freqs_cis_bs[
                                batch_index, prev_token_index:token_index, :
                            ] = freqs_cis[answer_index : answer_index + tri_size, :]
                    else:
                        if prev_token_index != answer_index:
                            # do nothing before the first delimiter comes out
                            mask[
                                batch_index,
                                0,
                                prev_token_index:,
                                answer_index:prev_token_index,
                            ] = float("-inf")
                            if self.decouple_freqs_cis:
                                freqs_cis_bs[
                                    batch_index, prev_token_index:
                                ] = freqs_cis[
                                    answer_index : answer_index
                                    + seqlen
                                    - prev_token_index
                                ]

                    prev_token_index = token_index

                if self.decouple_freqs_cis:
                    freqs_cis = freqs_cis_bs

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)  # [bs, tokens, dim]

        output = self.output(h) if is_train else self.output(h[:, -1, :])
        return output
