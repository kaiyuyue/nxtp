import torch.nn as nn

"""
Language Model-based Classifier
"""


class LangClassifier(nn.Module):
    def __init__(self, vision_encoder, language_decoder):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.language_decoder = language_decoder
        self.sampler = nn.Identity()

    def encode_images(self, x):
        h = self.vision_encoder.forward(x)
        z = self.vision_encoder.proj(h)  # [bs, CLS + patches, dim]
        return z

    def decode_images(self, z):
        x = self.sampler.forward(z)
        return x

    def forward(self):
        return
