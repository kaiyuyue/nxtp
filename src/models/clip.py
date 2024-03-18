from collections import OrderedDict

import torch
import torch.nn as nn
import clip

""" 
CLIP 
"""


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = clip.model.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", clip.model.QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = clip.model.LayerNorm(d_model)

    def attention(self, x: torch.Tensor, attn_mask: torch.Tensor = None):
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None):
        x = x + self.attention(self.ln_1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int):
        super().__init__()
        self.layers = layers
        self.width = width

        self.resblocks = torch.nn.ModuleList()
        for _ in range(layers):
            self.resblocks.append(ResidualAttentionBlock(width, heads))

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None):
        for resblock in self.resblocks:
            x = resblock(x, attn_mask)
        return x


class CLIPViT(nn.Module):
    """
    https://github.com/openai/CLIP/blob/main/clip/model.py#L206
    """

    def __init__(self, input_resolution, patch_size, width, layers, heads, output_dim):
        super().__init__()
        self.use_default_input_resolution = True
        self.input_resolution = input_resolution
        self.patch_size = patch_size
        self.width = width
        self.heads = heads
        self.output_dim = output_dim
        self.n_img_tokens = (input_resolution // patch_size) ** 2
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

        self.scale = width**-0.5
        self.class_embedding = nn.Parameter(self.scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            self.scale * torch.randn(self.n_img_tokens + 1, width)
        )
        self.ln_pre = clip.model.LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = clip.model.LayerNorm(width)
        self.proj = nn.Parameter(self.scale * torch.randn(width, output_dim))

    def forward(self, x):
        x = self.conv1(x)  # shape = [*, width, grid, grid]

        bs, c, h, w = x.shape
        x = x.reshape(bs, c, h * w)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        class_embed = self.class_embedding.expand(bs, 1, -1)
        x = torch.cat([class_embed, x], dim=1)  # shape = [*, grid ** 2 + 1, width]

        x = x + self.positional_embedding
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        x = self.proj(x)
        return x


def build_clip_model(args, state_dict, embed_dim):
    """
    https://github.com/openai/CLIP/blob/main/clip/model.py#L399
    """
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [
                k
                for k in state_dict.keys()
                if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")
            ]
        )
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round(
            (state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5
        )
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [
            len(
                set(
                    k.split(".")[2]
                    for k in state_dict
                    if k.startswith(f"visual.layer{b}")
                )
            )
            for b in [1, 2, 3, 4]
        ]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round(
            (state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5
        )
        vision_patch_size = None
        assert (
            output_width**2 + 1
            == state_dict["visual.attnpool.positional_embedding"].shape[0]
        )
        image_resolution = output_width * 32

    clip_embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(
        set(
            k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")
        )
    )  # 12

    model = clip.model.CLIP(
        clip_embed_dim,
        image_resolution,
        vision_layers,
        vision_width,
        vision_patch_size,
        context_length,
        vocab_size,
        transformer_width,
        transformer_heads,
        transformer_layers,
    )

    # hack to allow us to load in the weights from the CLIP model
    model.visual = CLIPViT(
        input_resolution=image_resolution,
        patch_size=vision_patch_size,
        width=vision_width,
        layers=vision_layers,
        heads=vision_width // 64,
        output_dim=clip_embed_dim,
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    clip.model.convert_weights(model)
    msgs = model.load_state_dict(state_dict)
    del state_dict

    # rebuild the projection layer to match the new embed dim
    del model.visual.proj
    model.visual.proj = nn.Linear(vision_width, embed_dim, bias=False)
    model.visual.proj.weight.data = model.visual.scale * torch.randn(
        embed_dim, vision_width
    )
    model.visual.output_dim = embed_dim

    if args.input_size != image_resolution:
        if args.group_size_per_batch_merge > 1:
            image_resolution = args.input_size * args.group_size_per_batch_merge // 2
        else:
            image_resolution = args.input_size
        model.visual.input_resolution = image_resolution
        model.visual.n_img_tokens = (image_resolution // model.visual.patch_size) ** 2
        model.visual.positional_embedding = nn.Parameter(
            model.visual.scale
            * torch.randn(model.visual.n_img_tokens + 1, model.visual.width)
        )
        model.visual.use_default_input_resolution = False

    # del text model
    del model.transformer
    del model.token_embedding
    del model.positional_embedding
    del model.ln_final
    del model.text_projection

    return model
