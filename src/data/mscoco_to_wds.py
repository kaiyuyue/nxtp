#!/usr/bin/env python
# coding: utf-8
#
# Usage:
#   ./scripts/run mscoco_to_wds.py

import sys
import json
import os
import os.path as osp
import shutil
import torchvision.datasets as dset
import torchvision.transforms as transforms

from tqdm import tqdm

# args
assert len(sys.argv) == 2
root = sys.argv[1]
coco_split = "val2017"  # train2017 | val2017

save_dir = osp.join(root, "coco_valid" if "val" in coco_split else "coco_train")
sub_fold_size: int = 2999
sub_fold_cnt: int = 1

if not os.path.exists(save_dir):
    os.makedirs(save_dir)


class MSCOCO(dset.CocoCaptions):
    def __getitem__(self, index: int):
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)
        path = self.coco.loadImgs(id)[0]["file_name"]

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target, path


caps = MSCOCO(
    root=osp.join(root, coco_split),
    annFile=osp.join(root, "annotations", f"captions_{coco_split}.json"),
    transform=transforms.ToTensor(),
)


for i, cap in tqdm(enumerate(caps), total=len(caps)):
    _, tgt, path = cap
    fname = path.split(".")[0]

    tgt = " ".join([t.strip() for t in tgt])
    src = osp.join(root, coco_split, path)

    if (i + 1) % sub_fold_size == 0:
        sub_fold_cnt += 1
    sub_fold = osp.join(save_dir, str(sub_fold_cnt - 1).zfill(5))
    if not os.path.exists(sub_fold):
        os.makedirs(sub_fold)

    img_path = osp.join(root, sub_fold, f"{fname}.jpg")
    txt_path = osp.join(root, sub_fold, f"{fname}.txt")
    jsn_path = osp.join(root, sub_fold, f"{fname}.json")

    shutil.copy(src, img_path)

    with open(txt_path, "w") as f:
        f.write(tgt)

    with open(jsn_path, "w") as f:
        json.dump({"path": path, "cap": tgt, "split": coco_split}, f)
