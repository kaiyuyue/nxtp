from sentencepiece import SentencePieceProcessor
from typing import List
import os


class Tokenizer:
    """
    https://github.com/facebookresearch/llama/blob/main/llama/tokenizer.py

    enable to process special tokens
    """

    def __init__(self, model_path: str):
        # reload tokenizer
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)
        print(f"Reloaded SentencePiece model from {model_path}")

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        print(
            f"LLaMA tokenizer info:",
            f"self.n_words: {self.n_words},",
            f"self.bos_id: {self.bos_id},",
            f"self.eos_id: {self.eos_id},",
            f"self.pad_id: {self.pad_id}",
        )
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

        # init special tokens
        self.special_tokens_itot = {}  # id to token
        self.special_tokens_ttoi = {}  # token to id
        self.n_special_tokens = 0

    def add_special_tokens(self, special_tokens: List[str]):
        for i, t in enumerate(special_tokens):
            tid = self.sp_model.vocab_size() + i
            self.special_tokens_itot[tid] = t
            self.special_tokens_ttoi[t] = tid
        self.n_special_tokens = len(special_tokens)
        self.n_words += self.n_special_tokens
        print(f"- add special tokens: {self.special_tokens_itot}")
        print(f"- self.n_special_tokens: {self.n_special_tokens}")
        print(f"- self.n_words: {self.n_words}")

    def _find_all_indices(
        self, input: List[int] or List[str], query: int or str
    ) -> List[int]:
        is_input_list = isinstance(input, list) or isinstance(input, str)
        indices = []
        index = -1  # Begin at -1 so that the first run increments it to 0
        while True:
            # Find next index from end of last
            if is_input_list:
                if query in input[index + 1 :]:
                    index = input.index(query, index + 1)
                else:
                    index = -1
            else:
                index = input.find(query, index + 1)
            if index == -1:
                break  # All occurrences have been found
            indices.append(index)
        return indices

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert isinstance(s, str)

        # parse
        specials_indtolen = {}  # indice: token length
        specials_indtotkn = {}  # indice: token string
        for t in self.special_tokens_ttoi:
            inds = self._find_all_indices(s, t)
            for ind in inds:
                specials_indtolen[ind] = len(t)
                specials_indtotkn[ind] = t

        # split
        ss = []
        prev_ind = 0
        for i in sorted(specials_indtolen.keys()):
            ss.append(s[prev_ind:i])
            ss.append(specials_indtotkn[i])
            prev_ind = i + specials_indtolen[i]
        ss.append(s[prev_ind:])

        # encode
        t = []
        for s in ss:
            if s in self.special_tokens_ttoi:
                t.append(self.special_tokens_ttoi[s])
            else:
                t += self.sp_model.encode(s)

        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        # parse
        specials_indtotkn = {}  # indice: token string
        for i in self.special_tokens_itot:
            inds = self._find_all_indices(t, i)
            for ind in inds:
                specials_indtotkn[ind] = self.special_tokens_itot[i]

        # decode
        ss = []
        prev_ind = 0
        for i in sorted(specials_indtotkn.keys()):
            ss.append(self.sp_model.decode(t[prev_ind:i]))
            ss.append(specials_indtotkn[i])
            prev_ind = i + 1
        ss.append(self.sp_model.decode(t[prev_ind:]))

        s = "".join(ss)
        return s
