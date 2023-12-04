from typing import List

import os
import io
import torchvision.transforms as T

from PIL import Image
from torch.utils.data import DataLoader
from torchdata.datapipes.iter import FileLister, FileOpener


_DATASETS_META = {
    # --- for validation ---
    "cc3m_valid": {
        "length": 12478,
        "root": f"cc3m/cc3m_valid",
    },
    "coco_valid": {
        "length": 5000,
        "root": f"coco/coco_valid",
    },
    "openimages_v7_valid": {
        "length": 41686,
        "root": f"openimages_v7/openimages_v7_valid",
    },
    # --- for training ---
    "coco": {
        "length": 118287,
        "root": f"coco/coco_train",
    },
    "cc3m": {
        "length": 2698118,
        "root": f"cc3m/cc3m",
    },
    "sbu": {
        "length": 828816,
        "root": f"sbu/sbu",
    },
    "laion115m": {
        "length": 67034168,
        "root": f"laion115m/laion115m",
    },
}


def build_datapipe(
    ds_root: str,
    ds_name: List[str],
    image_transform=None,
    text_transform=None,
    shuffle=False,
    num_shards=1,
    rank=0,
):
    """
    torchdata: 0.6.1 + torch: 2.0.1.
    """
    if isinstance(ds_name, str):
        ds_name = [ds_name]

    ds_length = 0
    ds_roots = []
    for ds in ds_name:
        ds = ds.lower()
        assert (
            ds in _DATASETS_META
        ), f"dataset {ds} not found in {_DATASETS_META.keys()}"
        ds_length += _DATASETS_META[ds]["length"]
        ds_roots += [os.path.join(ds_root, _DATASETS_META[ds]["root"])]
        print(f"datapipe + {ds} with length {_DATASETS_META[ds]['length']}")

    dp = FileLister(ds_roots, "*.tar", recursive=True)
    dp = FileOpener(dp, mode="b")
    dp = dp.load_from_tar(length=ds_length)
    dp = dp.webdataset()

    if shuffle:
        # do NOT set `shuffle=False` later in the DataLoader
        dp = dp.shuffle()

    dp = dp.sharding_filter()
    dp.apply_sharding(num_shards, rank)

    print(
        f"dataloader shards info: ",
        f"rank {rank} / world_size {num_shards} : ",
        f"len {len(dp)} / {ds_length}",
    )

    dp = dp.map(lambda x: apply_transform(x, image_transform, text_transform))
    return dp


def apply_transform(item, image_transform=None, text_transform=None):
    def decode_img(stream):
        img = Image.open(io.BytesIO(stream)).convert("RGB")
        if image_transform is not None:
            img = image_transform(img)
        return img

    def decode_txt(stream):
        txt = stream.decode("utf-8")
        if text_transform is not None:
            txt = text_transform(txt)
        return txt

    key = item["__key__"]
    img = decode_img(item[".jpg"].read())
    txt = decode_txt(item[".txt"].read())
    return img, txt, key


def build_preprocess(args, is_train=False):
    to_rgb = [T.Lambda(lambda x: x.convert("RGB"))]

    # NOTE: because we freeze CLIP, won't apply augmentations on images for now
    resized_crop = [
        # https://github.com/openai/CLIP/blob/main/clip/clip.py#L79
        T.Resize(args.input_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(args.input_size),
    ]

    norm = [
        T.ToTensor(),
        T.Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
        ),
    ]
    return T.Compose([*resized_crop, *to_rgb, *norm])


def build_dataloder(args, local_rank, world_size, is_train=True):
    dp = build_datapipe(
        ds_root=args.data_root,
        ds_name=args.data_name,
        image_transform=build_preprocess(args, is_train=is_train),
        text_transform=None,
        shuffle=True,
        num_shards=world_size,
        rank=local_rank,
    )

    dl = DataLoader(
        dp,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    return dl
