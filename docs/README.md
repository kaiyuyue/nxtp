# Documentation

- [Dependencies](#dependencies)
- [Data](#data)
    - [Using img2dataset Tool](#using-img2dataset-tool)
    - [Packing Image-Caption Pairs](#packing-image-caption-pairs)
- [Configuring](#configuring)
- [Pareparing LLaMA 2](#pareparing-llama-2)
- [Training](#training)
    - [Logging](#logging)
- [Evaluation](#evaluation)
    - [Results](#results)

## Dependencies

The project is developed with the following PyTorch versions:

- [python](https://www.python.org/) 3.10
- [torch](https://github.com/pytorch/pytorch) 2.0.1
- [torchvision](https://github.com/pytorch/vision) 0.15.2
- [torchdata](https://github.com/pytorch/data) 0.6.1

To install torch packages, please run

```bash
conda install pytorch==2.0.1 torchvision==0.15.2 pytorch-cuda=11.8 -c pytorch -c nvidia
```

To install other dependencies, please run

```bash
# for training and inference
pip install --no-cache-dir torchdata==0.6.1 fairscale nltk sentencepiece ftfy regex tqdm
pip install --no-cache-dir git+https://github.com/openai/CLIP.git

# for evaluation
pip install --no-cache-dir torchmetrics transformers
```

## Data

We pack the image-caption pairs in [WebDataset](https://pytorch.org/data/0.6/generated/torchdata.datapipes.iter.WebDataset.html#torchdata.datapipes.iter.WebDataset) format, with the `*.tar` files, for both training and evaluation.
In this section, we introduce two ways for downloading and saving datasets in this format.

### Using img2dataset Tool

[img2dataset](https://github.com/rom1504/img2dataset) is a python tool for downloading image-caption pairs given their URLs.
It has [multiple examples](https://github.com/rom1504/img2dataset/tree/main/dataset_examples) to show the usage for downloading the datasets like CC3M and LAION-400M.
Here is an example for downloading [CC3M](https://ai.google.com/research/ConceptualCaptions/download) training split:

```bash
#!/usr/bin/env bash
sed -i '1s/^/caption\turl\n/' cc3m.tsv
img2dataset --url_list cc3m.tsv --input_format "tsv" --url_col "url" --caption_col "caption" \
            --output_format webdataset --output_folder cc3m_shards --processes_count 16 --thread_count 64 \
            --image_size 512 --resize_mode keep_ratio --resize_only_if_bigger True \
            --enable_wandb False --save_metadata False --oom_shard_count 6
mv cc3m_shards cc3m_train
```
It is the same for downloading its validation split.
After finishing the download, the `cc3m` root folder should be

```bash
- cc3m/
    - cc3m_train/
        - 000000.tar
        - 000001.tar
        - *.tar
    - cc3m_valid/
        - 000000.tar
        - 000001.tar
```

Then append the CC3M splits to the [`_DATASETS_META`](../src/loader.py#L18) in [src/loader.py](../src/loader.py):

```python
_DATASETS_META = {
    # --- for validation ---
    "cc3m_valid": {
        "length": 12478,
        "root": "cc3m/cc3m_valid",
    },
    # --- for training ---
    "cc3m": {
        "length": 2698118,
        "root": "cc3m/cc3m_train",
    },
}
```

where

- `length`: the number of image-caption pairs in the split.

### Packing Image-Caption Pairs

While other datasets, like [OpenImages V7](https://storage.googleapis.com/openimages/web/index.html) and [Visual Genome](https://homes.cs.washington.edu/~ranjay/visualgenome/index.html), have isolated caption files and separated image folders.
Thus, we need to write a customized script to reorganize them into the WebDataset structure.
Here we use [COCO](https://cocodataset.org/#home) as an example to demonstrate each step of packing data from scratch.

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

```bash
python data/mscoco_to_wds.py
```

Under the new folder `coco_valid` and `coco_train`, it has multiple subfolders, each subfolder has a same number of triplets of `json`, `txt`, and `jpg`:

```bash
- coco/
    - coco_valid/
        - 00000
            - 000000000139.jpg
            - 000000000139.json
            - 000000000139.txt
            - *.jpg/json/txt
        - 00001
            - *.jpg/json/txt
    - coco_train/
        - 00000
            - *.jpg/json/txt
        - 00001
        - *
```

where

- `json` file: one line of image information `{"path": "000000000139.jpg", "cap": "A woman stands in the dining area at the table. A room with chairs, a table, and a woman in it. A woman standing in a kitchen by a window A person standing at a table in a room. A living area with a television and a table", "split": "val2017"}`;
- `txt` file: one line of caption `A woman stands in the dining area at the table. A room with chairs, a table, and a woman in it. A woman standing in a kitchen by a window A person standing at a table in a room. A living area with a television and a table`;
- `jpg` file: the corresponding image file.

Then, make `tar -cf` for those subfolders:

```bash
- coco_valid/
    - 00000.tar
    - 00001.tar
- coco_train/
    - 00000.tar
    - 00001.tar
    - *.tar
```

Finally, append the dataset COCO to the [`_DATASETS_META`](../src/loader.py#L18) in [src/loader.py](../src/loader.py):

```python
_DATASETS_META = {
    # --- for validation ---
    "cc3m_valid": {
        "length": 12478,
        "root": "cc3m/cc3m_valid",
    },
    "coco_valid": {
        "length": 5000,
        "root": "coco/coco_valid",
    },
    # --- for training ---
    "cc3m": {
        "length": 2698118,
        "root": "cc3m/cc3m_train",
    },
    "coco": {
        "length": 118287,
        "root": "coco/coco_train",
    },
}
```

For any other datasets, it is easy to write scripts to reorganize them into the aformentioned structure.

## Configuring

We use [src/configs/config_base.py](../src/configs/config_base.py) to configure all hyper-parameters for experiments.
To better maintain the different settings for different experiments, we won't modify the base config file.
Instead, we create a new configuration file for each experiment, which only has the different settings from the base config file, for example, [src/configs/config_g3m.py](../src/configs/config_g3m.py).

If want to add a new hyper-parameter, please add it to the base config file [src/configs/config_base.py](../src/configs/config_base.py) and set its default value.
Then, in the experiment config file, we can overwrite the default value of the hyper-parameter.
Before training, please check the config file to make sure all hyper-parameters are set correctly.

## Pareparing LLaMA 2

The language decoder is truncated from the 7B model of [LLaMA 2](https://github.com/facebookresearch/llama), so need to follow the intruction to download it. 
Then set the path of the model checkpoint and tokenizer in the config file, for example, `cfg.llama_version`, `cfg.llama_model`, `cfg.llama_dir`, and `cfg.tokenizer_path` in [src/configs/config_g3m.py](../src/configs/config_g3m.py#L22-L30).

## Training

To train on a single machine node with multiple GPUs, please run

```bash
./scripts/run train.py configs/config_g3m.py
```

To train on multiple machine node with multiple GPUs, we provide a script as an example to launch the training with [slurm](https://slurm.schedmd.com/overview.html):

```bash
#!/usr/bin/env bash

#SBATCH --ntasks=4
#SBATCH --nodes=4
#SBATCH --cpus-per-gpu=12
#SBATCH --gres=gpu:8
#SBATCH --partition=learn
#SBATCH --mem=256G
#SBATCH --job-name=dist_g3m
#SBATCH --output=logs/log.g3m
#SBATCH --wait-all-nodes=1
#SBATCH --exclusive

./scripts/run_dist train.py configs/config_g3m.py
```

### Logging

For reference, we provide the training logs of G3M in [src/logs/log.g3m.training](../src/logs/log.g3m.training) and those of G70M in [src/logs/log.g70m.training](../src/logs/log.g70m.training).
The following figures are the training loss curve and learning rate curve of G3M.

<p align="left">
  <img src="../assets/loss-dark-mode.svg#gh-dark-mode-only", height="256">
  <img src="../assets/loss-light-mode.svg#gh-light-mode-only", height="256">
  <img src="../assets/lr-dark-mode.svg#gh-dark-mode-only", height="256">
  <img src="../assets/lr-light-mode.svg#gh-light-mode-only", height="256">
</p>

## Evaluation

Once having the trained model, please set the path of the model checkpoint in the config file, for example, `cfg.eval_ckpt_path: str` in [src/configs/config_g3m.py](../src/configs/config_g3m.py#L82).
To evaluate the model on a single machine node with multiple GPUs, please run

```bash
./scripts/run eval_hook.py evals/eval.py configs/config_g3m.py
```

### Results

The following table shows the reproduced results on the validation splits with top-10 predictions.
The numbers should be the same if using the provided checkpoint trained on G3M.

|                   |                          | CC3M    |         |         | COCO    |         |         | OpenImages |            |         |
|:-----------------:|:------------------------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:----------:|:----------:|:-------:|
| **#&nbsp;params** | **training&nbsp;group**  |  **R**  |  **P**  |  **F1** |  **R**  |  **P**  |  **F1** |  **R**     |    **P**   |  **F1** |
|     1.78B         |        &nbsp; G3M        | 0.73970 | 0.53118 | 0.61234 | 0.70325 | 0.71873 | 0.70859 | 0.61588    |   0.54895  | 0.57380 |
|     1.78B         |        G70M              | 0.72099 | 0.51171 | 0.59288 | 0.76546 | 0.75607 | 0.75811 | 0.66123    |   0.56283  | 0.60118 |