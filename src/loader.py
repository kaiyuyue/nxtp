from typing import List, Sized

import os
import io
import torchvision.transforms as T

from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.datapipes.iter.sharding import SHARDING_PRIORITIES
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
        ds_dir = os.path.join(ds_root, _DATASETS_META[ds]["root"])
        ds_roots += [ds_dir]
        print(f"datapipe + {ds} : length {_DATASETS_META[ds]['length']} : {ds_dir}")

    dp = FileLister(ds_roots, "*.tar", recursive=True)
    dp = FileOpener(dp, mode="b")
    dp = dp.load_from_tar(length=ds_length)
    dp = dp.webdataset()

    def __len__(self):
        if isinstance(self.source_datapipe, Sized):
            return len(self.source_datapipe)
        raise TypeError(f"{type(self).__name__} instance doesn't have valid length")

    # NOTE: monkey-patch the __len__ method
    # pull: https://github.com/pytorch/data/pull/1289
    type(dp).__len__ = __len__
    
    if shuffle:
        # do NOT set `shuffle=False` later in the DataLoader
        dp = dp.shuffle()

    def _new_apply_sharding(
        self, num_of_instances, instance_id, sharding_group=SHARDING_PRIORITIES.DEFAULT
    ):
        if instance_id >= num_of_instances:
            raise ValueError(
                f"instance_id({instance_id}) should be smaller than num_of_ins    tances({num_of_instances})"
            )
        if sharding_group == SHARDING_PRIORITIES.DEFAULT:
            # avoid setting the same group for multiple times with default group
            return
        self.groups[sharding_group] = (num_of_instances, instance_id)
        self._update_num_of_instances()

    sharding_group = SHARDING_PRIORITIES.DISTRIBUTED
    assert (
        sharding_group != SHARDING_PRIORITIES.DEFAULT
    ), "sharding_group should not be DEFAULT"
    dp = dp.sharding_filter()
    dp.apply_sharding = _new_apply_sharding.__get__(dp, type(dp))
    dp.apply_sharding(
        num_shards,
        rank,
        sharding_group=sharding_group,
    )

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
    dp = build_datapipe(
        ds_root=args.data_root,
        ds_name=args.data_name,
        image_transform=build_preprocess(args.input_size),
        text_transform=None,
        shuffle=True,
        num_shards=world_size,
        rank=global_rank,
    )

    def _worker_init_fn(worker_id):
        """
        handle multi-process loading for sharding
        """
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process loading
            num_workers = 1
            worker_id = 0
        else:  # multi-process loading
            num_workers = worker_info.num_workers
            worker_id = worker_info.id

        """
        def __iter__(self):
            for i, item in enumerate(self.source_datapipe):
                if i % self.num_of_instances == self.instance_id:
                    yield item
        """
        dp.datapipe.num_of_instances = dp.datapipe.num_of_instances * num_workers
        dp.datapipe.instance_id = dp.datapipe.instance_id * num_workers + worker_id
        
    dl = DataLoader(
        dp,
        batch_size=args.batch_size,
        shuffle=True,
        worker_init_fn=_worker_init_fn,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True if is_train else False,
    )
    return dl
