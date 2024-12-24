# download data
How can researchers in the Chinese time zone download this g3m dataset?

### prepare VPN
For the VPN, you can refer to this project.
https://github.com/Elegycloud/clash-for-linux-backup

### directory structure:
you can refer to https://github.com/kaiyuyue/nxtp/blob/main/docs/README.md
and https://github.com/kaiyuyue/nxtp/blob/main/src/loader.py

### cc3m-wds
https://huggingface.co/datasets/pixparse/cc3m-wds


```shell
download cc3m-wds
huggingface-cli download --repo-type dataset pixparse/cc3m-wds --local-dir object_path
```

### openimages_v7
```shell
download validation set
pip install awscli
mkdir openimages_v7
aws s3 --no-sign-request cp s3://open-images-dataset/tar/validation.tar.gz openimages_v7
```

download open_images_validation_localized_narratives file
```shell
wget https://storage.googleapis.com/localized-narratives/annotations/open_images_validation_localized_narratives.jsonl
```

Processing openimages_v7 data
```python
#!/usr/bin/env python
# coding: utf-8
#
# Usage:
#   python3 openimages_to_wds.py

import os
import os.path as osp
import json
import shutil
from tqdm import tqdm

root = "/data/openimages_v7"
save_dir = osp.join(root, "openimages_v7_valid")

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

json_file_path = osp.join(root, "open_images_validation_localized_narratives.jsonl")
sub_fold_cnt = 1
with open(json_file_path, "r") as f:
    for i, line in tqdm(enumerate(f)):
        line = json.loads(line)
        image_id = line["image_id"]

        cap = line["caption"]
        src = osp.join(root, "valid", f"{image_id}.jpg")

        if (i + 1) % 2999 == 0:
            sub_fold_cnt += 1
        sub_fold = osp.join(save_dir, str(sub_fold_cnt - 1).zfill(5))
        if not os.path.exists(sub_fold):
            os.makedirs(sub_fold)

        img_path = osp.join(root, sub_fold, f"{image_id}.jpg")
        txt_path = osp.join(root, sub_fold, f"{image_id}.txt")
        jsn_path = osp.join(root, sub_fold, f"{image_id}.json")

        shutil.copy(src, img_path)

        with open(txt_path, "w") as f:
            f.write(cap)

        with open(jsn_path, "w") as f:
            json.dump(line, f)
```

### sbucaptions 
download sbu_captions
```shell
#!/bin/bash
cd object object_path
kaggle datasets download akashnuka/sbucaptions
```

### coco2017
https://cocodataset.org/#download
```shell
#!/bin/bash
cd object object_path
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
```
After downloading its 2017 train/val annotations and images in the root folder `coco`, we have the directory structure as:

```bash
- coco/
    - annotations/
        - captions_train2017.json
        - captions_val2017.json
    - train2017/
        - 000000000009.jpg
        - *.jpg
    - val2017/
        - 000000000139.jpg
        - *.jpg
```
We have a simple script [src/data/mscoco_to_wds.py](../src/data/mscoco_to_wds.py) for reorganizing `val2017` pairs into a new folder `coco_valid`.
Then run

```python
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
```
