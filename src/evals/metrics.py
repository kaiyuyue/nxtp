from typing import List

import torch
import torch.nn.functional as F
import clip

from torchmetrics import Metric
from transformers import AutoModel, AutoTokenizer

__all__ = [
    "SemanticFScore",
]


class SemanticFScore(Metric):
    def __init__(
        self,
        *args,
        model_name="bert",
        cache_dir: str = ".cache",
        max_seq_len: int = 128,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.cache_dir = cache_dir
        self.max_seq_len = max_seq_len
        print(f"max_seq_len for Metric::SemanticFScore: {self.max_seq_len}")

        self._set_embed_model(model_name)
        self._add_states()

    def _set_embed_model(self, model_name: str):
        assert model_name in ["bert", "clip"]

        if model_name == "bert":
            self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, cache_dir=self.cache_dir
            )
            self._model = AutoModel.from_pretrained(
                self.model_name, cache_dir=self.cache_dir
            )
            self._model.eval()
            self._model.to(self.device)
            self.encode = self.encode_bert
        elif model_name == "clip":
            model, _ = clip.load(
                "ViT-L/14",
                device="cpu",
                download_root=self.cache_dir,
                jit=False,
            )
            del model.visual

            self._model = model.eval().to(self.device)
            self._tokenizer = clip.tokenize
            self.encode = self.encode_clip
        else:
            raise NotImplementedError

    def _add_states(self):
        self.add_state("recall", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("precision", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("f", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

        self.add_state("div_avg", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("div_std", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def encode_bert(self, sentence: str, **kwargs) -> torch.Tensor:
        """
        Encode the input sentence with the BERT model.

        Args:
            sentence (str): Input sentence.
        """
        tokens = self._tokenizer.encode_plus(
            sentence,
            max_length=self.max_seq_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        ).to(self.device)
        embeddings = self._model(**tokens).last_hidden_state

        # mask out padding tokens
        mask = tokens["attention_mask"].unsqueeze(-1).expand(embeddings.size()).float()
        masked_embeddings = embeddings * mask

        # sum over all tokens
        summed = torch.sum(masked_embeddings, dim=1)
        summed_mask = torch.clamp(mask.sum(dim=1), min=1e-9)

        # normalise and remove batch dimension
        embeddings = summed / summed_mask
        embeddings = embeddings.squeeze(0)

        return embeddings

    def encode_clip(self, sentence: str, **kwargs) -> torch.Tensor:
        tokens = self._tokenizer([sentence], truncate=True).to(self.device)
        embeddings = self._model.encode_text(tokens).float()[0]
        return embeddings

    def update(self, values: List[List[str]], targets: List[List[str]]):
        """
        Args:
            values (List[List[str]]): predicted fragments for each sample.
            targets (List[List[str]]): ground truth fragments for each sample.
        """
        if len(values) != len(targets):
            raise ValueError(
                f"values ({len(values)}) and targets ({len(targets)}) must have same length"
            )

        # register hooks for logging
        self.scores = []
        self.div_scores = []

        for vals, tgts in zip(values, targets):
            x_tgts = torch.stack([self.encode(t) for t in tgts])  # [M, D]
            x_vals = torch.stack([self.encode(v) for v in vals])  # [N, D]

            x_tgts = F.normalize(x_tgts, dim=1)
            x_vals = F.normalize(x_vals, dim=1)

            s = x_tgts @ x_vals.T  # [M, N]
            r = s.max(dim=1).values.mean()  # [M]
            p = s.max(dim=0).values.mean()  # [N]
            f = torch.nan_to_num(2 * r * p / (r + p))

            self.recall += r
            self.precision += p
            self.f += f

            self.scores.append(s.detach().cpu().numpy().tolist())

            # compute diversity
            d = x_vals @ x_vals.T
            d = torch.triu(d, diagonal=1)
            d = d[d != 0]
            self.div_scores.append(d.detach().cpu().numpy().tolist())

            # if there is only one prediction,
            # i.e., one val sitting on the diagonal alone,
            # the mean and std will be nan after shaving off the diagonal
            self.div_avg += d.mean().nan_to_num(0.0)
            self.div_std += d.std().nan_to_num(0.0)

        self.total += len(values)

    def compute(self):
        recall = self.recall / self.total
        precision = self.precision / self.total
        f = self.f / self.total

        div_avg = self.div_avg / self.total
        div_std = self.div_std / self.total
        return recall, precision, f, {"div_avg": div_avg, "div_std": div_std}
