from typing import Optional, List, Tuple
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from torch import Tensor


def decoder_top_p(logits: Tensor, p: float):
    """
    nucleus sampling
    """
    probs = F.softmax(logits, dim=-1)
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_tokens = torch.multinomial(probs_sort, num_samples=1)
    next_tokens = torch.gather(probs_idx, -1, next_tokens)
    probs = torch.gather(probs_sort, -1, next_tokens)
    return probs, next_tokens


def decoder_top_k(logits: Tensor, k: int):
    """
    https://github.com/openai/gpt-2/blob/master/src/sample.py#L5
    """
    assert k >= 0

    if k == 1:
        # greedy decoding with top-1
        return F.softmax(logits, dim=-1).topk(k=1)

    if k > 1:
        values, _ = torch.topk(logits, k=k)
        min_values = values[:, -1].unsqueeze(1).repeat(1, logits.shape[-1])
        logits = torch.where(
            logits < min_values,
            torch.ones_like(logits, dtype=logits.dtype) * -1e10,
            logits,
        )

    probs = F.softmax(logits, dim=-1)
    next_tokens = torch.multinomial(probs, num_samples=1)
    probs = torch.gather(probs, -1, next_tokens)
    return probs, next_tokens


class OneShotDecoder:
    def __init__(self, k: int = 1, max_update_iter: int = 20) -> None:
        self.k = k

        # if model is not fully converged, sometimes it will not stop
        self.max_update_iter = max_update_iter

    def reset(self):
        self.update_iter = 0
        self.completed_mask = None
        self.one_shot_size = None
        self.prob_next_tokens = None

    def update(
        self, tokens: Tensor, logits: Tensor, sum_logprobs: Tensor
    ) -> Tuple[Tensor, bool]:
        """
        one-shot sampling
        """
        probs = F.softmax(logits, dim=-1)

        if self.update_iter == 0:
            topk_vals, topk_inds = torch.topk(probs, k=self.k)  # [bs, k]
            topk_vals = topk_vals.reshape(-1, 1)  # [bs * k, 1]
            topk_inds = topk_inds.reshape(-1, 1)

            next_tokens = topk_inds
            if tokens.shape[0] * self.k != next_tokens.shape[0]:
                raise ValueError(
                    f"{tokens.shape[0]} * {self.k} != {next_tokens.shape[0]}"
                )
            if tokens.shape[0] != next_tokens.shape[0]:
                # for the first update
                tokens = tokens.repeat_interleave(self.k, dim=0)

            self.prob_next_tokens = topk_vals
            self.one_shot_size = self.k
            self.completed_mask = torch.ones_like(next_tokens, dtype=torch.bool)
        else:
            topk_vals, topk_inds = torch.topk(probs, k=1)  # [bs, 1]
            next_tokens = topk_inds
            self.prob_next_tokens = torch.cat(
                [self.prob_next_tokens, topk_vals], dim=-1
            )
            self.completed_mask &= (
                next_tokens != 29892
            )  # TODO: should not be hard-coded for the delimiter token

        tokens = torch.cat([tokens, next_tokens], dim=-1)
        self.update_iter += 1

        # if all tokens hit the delimiter, then stop
        if self.completed_mask is not None:
            completed = self.completed_mask.sum() == 0

        # if the number of updates exceeds the max_update_iter, then stop
        if self.update_iter >= self.max_update_iter - 1:
            # append the delimiter token
            tokens = torch.cat([tokens, torch.ones_like(tokens) * 29892], dim=-1)
            completed = True
            print(
                f"Warning: the number of updates exceeds the max_update_iter: {tokens.shape} / {self.max_update_iter}"
            )

        return tokens, completed

    def finalize(self, tokens: Tensor, sum_logprobs: Tensor, shave_prefix: bool = True):
        bs = tokens.shape[0] // self.one_shot_size

        prefix_indice = torch.where(tokens == 526)[-1][
            0
        ]  # TODO: should not be hard-coded for the token of last "are" in the prompt
        prefix_tokens = tokens[0, : prefix_indice + 1].tolist()  # [n]
        tokens = tokens[:, prefix_indice + 1 :]  # [bs, seq_len]

        tokens = tokens.reshape(bs, self.one_shot_size, -1)  # [bs, k, seq_len]
        probs = self.prob_next_tokens.reshape(bs, self.one_shot_size, -1)  # [bs, k, 1]

        def nested_dict():
            return defaultdict(list)

        delimiter_idxs = defaultdict(nested_dict)
        idxs_bs, idxs_k, idxs_seq = torch.where(tokens == 29892)

        for idx_bs, idx_k, idx_seq in zip(idxs_bs, idxs_k, idxs_seq):
            if idx_k.item() not in delimiter_idxs[idx_bs.item()]:
                delimiter_idxs[idx_bs.item()][idx_k.item()] = idx_seq.item()

        pred_tokens, pred_probs = [], []
        for idx_bs in range(bs):
            label_tokens, label_probs = [], []

            for idx_k in range(self.one_shot_size):
                end_idx = delimiter_idxs[idx_bs][idx_k] + 1

                tokens_per_label = tokens[idx_bs, idx_k, :][:end_idx]
                probs_per_label = probs[idx_bs, idx_k, :][:end_idx]

                label_tokens.extend(tokens_per_label.tolist())
                label_probs.extend(probs_per_label.tolist())

            if not shave_prefix:
                # prefix means the prefix tokens of the prompt
                label_tokens = prefix_tokens + label_tokens

            pred_tokens.append(label_tokens)
            pred_probs.append(label_probs)

        return pred_tokens, pred_probs


class GreedyDecoder:
    """
    sampling methods:
        top_k
        top_p (nucleus)
    """

    def __init__(
        self,
        eot: int,
        temperature: float = 0.0,
        k: int = 1,
        p: float = 0.95,
        func: str = "top_p",
        penalty: float = 0.0,
    ):
        self.temperature = temperature or 1.0
        self.eot = eot
        assert func in ["top_k", "top_p"]

        self.func = globals()[f"decoder_{func}"]  # call the func by str name
        self.hparam = k if func == "top_k" else p
        if func == "top_k":
            self.temperature = 1.0

        # penalized sampling (https://arxiv.org/abs/1909.05858)
        self.penalty = penalty
        self.generated_tokens: List[torch.Tensor] = []

    def reset(self):
        self.generated_tokens = []
        self.prob_next_tokens = None
        pass

    def update(
        self,
        tokens: Tensor,
        logits: Tensor,
        sum_logprobs: Tensor,
    ) -> Tuple[Tensor, bool]:
        """ """
        if self.penalty > 0.0 and self.generated_tokens:
            bs = logits.shape[0]
            col_ind = torch.cat(self.generated_tokens, dim=-1)
            row_ind = torch.arange(bs).unsqueeze(-1)
            logits[row_ind, col_ind] /= self.penalty

        probs, next_tokens = self.func(logits / self.temperature, self.hparam)
        self.prob_next_tokens = probs

        if self.penalty > 0.0:
            self.generated_tokens.append(next_tokens)

        tokens = torch.cat([tokens, next_tokens], dim=-1)
        completed = False  # always False for greedy decoding
        return tokens, completed

    def finalize(self, tokens: Tensor, sum_logprobs: Tensor, shave_prefix: bool = True):
        tokens: List[List[int]] = tokens.tolist()
        sum_logprobs: List[float] = sum_logprobs.tolist()
        return tokens, sum_logprobs


class BeamSearchDecoder:
    """
    https://github.com/openai/whisper/blob/main/whisper/decoding.py#L299
    """

    def __init__(
        self,
        eot: int,
        beam_size: int = 3,
        patience: Optional[float] = None,
        length_penalty: Optional[float] = None,
        penalty: float = 0.0,
    ):
        self.beam_size = beam_size
        self.eot = eot
        self.patience = patience or 1.0
        self.max_candidates: int = round(beam_size * self.patience)
        self.finished_sequences = None
        self.length_penalty = length_penalty

        assert (
            self.max_candidates > 0
        ), f"Invalid beam size ({beam_size}) or patience ({patience})"

        # penalized sampling (https://arxiv.org/abs/1909.05858)
        self.penalty = penalty
        self.generated_tokens: List[torch.Tensor] = []

    def reset(self):
        self.finished_sequences = None
        self.selected_beam = None
        self.beam_tokens = None
        self.beam_source_indices = None
        self.prob_next_tokens = None
        self.generated_tokens = []
        pass

    def update(
        self, tokens: Tensor, logits: Tensor, sum_logprobs: Tensor
    ) -> Tuple[Tensor, bool]:
        if tokens.shape[0] % self.beam_size != 0:
            raise ValueError(f"{tokens.shape}[0] % {self.beam_size} != 0")

        bs = tokens.shape[0] // self.beam_size
        if self.finished_sequences is None:  # for the first update
            self.finished_sequences = [{} for _ in range(bs)]

        if self.generated_tokens and self.penalty > 0.0:
            _bs = logits.shape[0]
            col_ind = torch.cat(self.generated_tokens, dim=-1)
            row_ind = torch.arange(_bs).unsqueeze(-1)
            logits[row_ind, col_ind] /= self.penalty

        logprobs = F.log_softmax(logits.float(), dim=-1)
        next_tokens, source_indices, finished_sequences = [], [], []
        for i in range(bs):
            scores, sources, finished = {}, {}, {}

            # STEP 1: calculate the cumulative log probabilities for possible candidates
            for j in range(self.beam_size):
                idx = i * self.beam_size + j
                prefix = tokens[idx].tolist()
                for logprob, token in zip(*logprobs[idx].topk(self.beam_size + 1)):
                    new_logprob = (sum_logprobs[idx] + logprob).item()
                    sequence = tuple(prefix + [token.item()])
                    scores[sequence] = new_logprob
                    sources[sequence] = idx

            # STEP 2: rank the candidates and keep the top beam_size sequences for each input
            saved = 0
            for sequence in sorted(scores, key=scores.get, reverse=True):
                if sequence[-1] == self.eot:
                    finished[sequence] = scores[sequence]
                else:
                    sum_logprobs[len(next_tokens)] = scores[sequence]
                    next_tokens.append(sequence)
                    source_indices.append(sources[sequence])

                    saved += 1
                    if saved == self.beam_size:
                        break

            finished_sequences.append(finished)

        tokens = torch.tensor(next_tokens, device=tokens.device)

        # update the beam state
        self.beam_tokens = next_tokens
        self.beam_source_indices = source_indices

        # TODO: hard to track the prob for each token
        self.prob_next_tokens = torch.zeros_like(tokens[:, -1])

        if self.penalty > 0.0:
            self.generated_tokens.append(tokens)

        # add newly finished sequences to self.finished_sequences
        assert len(self.finished_sequences) == len(finished_sequences)
        for i, (previously_finished, newly_finished) in enumerate(
            zip(self.finished_sequences, finished_sequences)
        ):
            for seq in sorted(newly_finished, key=newly_finished.get, reverse=True):
                if len(previously_finished) >= self.max_candidates:
                    break  # the candidate list is full
                # previously_finished[seq] = newly_finished[seq]
                self.finished_sequences[i][seq] = newly_finished[seq]

        # mark as completed if all inputs have enough number of samples
        completed = all(
            len(sequences) >= self.max_candidates
            for sequences in self.finished_sequences
        )

        return tokens, completed

    def finalize(self, tokens: Tensor, sum_logprobs: Tensor, shave_prefix: bool = True):
        bs = tokens.shape[0] // self.beam_size
        tokens = tokens.reshape(bs, self.beam_size, -1)
        sum_logprobs = sum_logprobs.reshape(bs, self.beam_size)

        # collect all finished sequences, including patience, and add unfinished ones if not enough
        sum_logprobs = sum_logprobs.cpu()
        for i, sequences in enumerate(self.finished_sequences):
            if (
                len(sequences) < self.beam_size
            ):  # when not enough sequences are finished
                for j in list(np.argsort(sum_logprobs[i]))[::-1]:
                    sequence = tokens[i, j].tolist() + [self.eot]
                    sequences[tuple(sequence)] = sum_logprobs[i][j].item()
                    if len(sequences) >= self.beam_size:
                        break

        tokens: List[List[Tensor]] = [
            [torch.tensor(seq) for seq in sequences.keys()]
            for sequences in self.finished_sequences
        ]
        sum_logprobs: List[List[float]] = [
            list(sequences.values()) for sequences in self.finished_sequences
        ]

        selected = self.max_likelihood_rank(tokens, sum_logprobs)
        self.selected_beam = selected

        sum_logprobs: List[float] = [lp[i] for i, lp in zip(selected, sum_logprobs)]
        pred_tokens: List[List[int]] = []
        for i, t in zip(selected, tokens):
            ti = t[i].tolist()
            if shave_prefix:
                ti = ti[ti.index(526) + 1 :]
            pred_tokens.append(ti)
        return pred_tokens, sum_logprobs

    def max_likelihood_rank(
        self, tokens: List[List[Tensor]], sum_logprobs: List[List[float]]
    ):
        def _scores(logprobs, lengths):
            result = []
            for logprob, length in zip(logprobs, lengths):
                if self.length_penalty is None:
                    penalty = length
                else:
                    # from the Google NMT paper
                    penalty = ((5 + length) / 6) ** self.length_penalty
                result.append(logprob / penalty)
            return result

        # get the sequence with the highest score
        lengths = [[len(t) for t in s] for s in tokens]
        return [np.argmax(_scores(p, l)) for p, l in zip(sum_logprobs, lengths)]
