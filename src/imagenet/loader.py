from typing import Optional, Callable

import torchvision.transforms as T

from PIL import Image
from pathlib import Path
from torch.utils.data import DataLoader


_DATASETS_META = {
    # --- for validation ---
    "in1k_valid": {
        "length": 50000,
        "root": f"imagenet/val",
        "list": f"imagenet/val.list",
    },
    # --- for training ---
    "in1k": {
        "length": 1281167,
        "root": f"imagenet/train",
        "list": f"imagenet/train.list",
    },
}


class DistributedImageList:
    def __init__(
        self,
        list_path: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        num_shards: int = 1,
        rank: int = 0,
    ):
        super().__init__()
        self.samples = self.loading_list(list_path)

        self.samples = self.samples[rank::num_shards]
        print(f"rank={rank:2d}/{num_shards:2d}, num_samples={len(self.samples)}")

        self.transform = transform
        self.target_transform = target_transform

    def loading_list(self, list_path):
        samples = []
        with open(list_path, "r") as f:
            for line in f:
                line = line.strip().split(":")
                path = line[0].lstrip().strip()
                target = line[1].lstrip().strip()
                samples.append((path, target))
        return samples

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img) if self.transform is not None else img
        if self.target_transform is not None:
            target = [self.target_transform(t) for t in target]
        return img, target, path

    def __len__(self):
        return len(self.samples)


def build_preprocess(input_size):
    to_rgb = [T.Lambda(lambda x: x.convert("RGB"))]

    # NOTE: because we freeze CLIP, won't apply augmentations on images for now
    resized_crop = [
        # https://github.com/openai/CLIP/blob/main/clip/clip.py#L79
        T.Resize(input_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(input_size),
    ]

    norm = [
        T.ToTensor(),
        T.Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
        ),
    ]
    return T.Compose([*resized_crop, *to_rgb, *norm])


def build_dataloader(args, global_rank, world_size, is_train=True):
    assert len(args.data_name) == 1

    ds = DistributedImageList(
        list_path=Path(args.data_root) / _DATASETS_META[args.data_name[0]]["list"],
        transform=build_preprocess(args.input_size),
        target_transform=None,
        num_shards=world_size,
        rank=global_rank,
    )

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True if is_train else False,
    )
    return dl
