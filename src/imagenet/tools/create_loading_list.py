#!/usr/bin/env python3

import os
from tqdm import tqdm


samples_id_to_fold = dict()
samples_fold_to_name = dict()

with open("in1k_label.index", "r") as fio:
    for line in fio:
        line = line.strip().split(":")

        class_id = int(line[0])
        class_fold = str(line[1].split(",")[0].lstrip().strip())
        class_name = str(line[1].split(",")[1].lstrip().strip())

        samples_id_to_fold[class_id] = {
            "class_fold": class_fold,
            "class_name": class_name,
        }

with open("in1k_label.name", "r") as fio:
    for line in fio:
        line = line.strip().split(":")
        class_id = int(line[0])
        class_name = [n.lstrip().strip() for n in line[1].split(",")]
        samples_fold_to_name[samples_id_to_fold[class_id]["class_fold"]] = class_name

"""
writing the list files
"""

ROOT_PATH = "/data/imagenet"  # change this to the path

for split in ["train", "val"]:

    fio = open(f"{split}.list", "w")

    for subfold in tqdm(os.listdir(f"{ROOT_PATH}/{split}")):
        for img_path in os.listdir(f"{ROOT_PATH}/{split}/{subfold}"):
            img_path = os.path.join(f"{ROOT_PATH}/{split}/{subfold}", img_path)
            cls_name = ",".join(samples_fold_to_name[subfold])
            fio.write(f"{img_path} : {cls_name}\n")

    fio.close()
