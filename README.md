# Object Recognition as Next Token Prediction

[Colab](https://colab.research.google.com/drive/1pJX37LP5xGLDzD3H7ztTmpq1RrIBeWX3?usp=sharing) | [Documentation](docs/README.md)

## Introduction

This repository contains code for producing the results of the paper, which approaches recognition as next token prediction.
It jointly trains an image encoder (ViT-L/14) from [CLIP](https://github.com/openai/CLIP) and a truncated language decoder from [LLaMA 2](https://github.com/facebookresearch/llama)'s 7B model, resulting in a compact model with 1.78B total parameters.
The auto-regressive model, employing the one-shot sampling strategy that samples multiple labels in parallel, can efficiently perform large-scale predictions, such as predicting the top-100 labels.

<p align="center">
  <img src="./assets/method-dark-mode.svg#gh-dark-mode-only">
  <img src="./assets/method-light-mode.svg#gh-light-mode-only">
  <br/>
</p>

## Models

The following table shows the reproduced results of recall (**R** column in Table 1 of the paper) on the validation splits with top-10 predictions.

<table>
  <tbody>
  <th valign="bottom"># params</th>
  <th valign="bottom">training group</th>
  <th valign="bottom">checkpoint</th>
  <th valign="bottom">md5</th>
  <th valign="bottom">CC3M</th>
  <th valign="bottom">COCO</th>
  <th valign="bottom">OpenImages</th>
  <tr>
    <td align="center">1.78B</td>
    <td align="center">&nbsp;&nbsp;G3M</td>
    <td align="center"><a href="https://drive.google.com/file/d/1QYT7kXD9qks6rQh0m2PnVlnSffj8VXNh/view?usp=sharing">Google Drive</a> | <a href="https://huggingface.co/kaiyuyue/nxtp/tree/main">Hugging Face</a></td>
    <td align="center"><tt>b2a69b</tt></td>
    <td align="center">0.740</td>
    <td align="center">0.703</td>
    <td align="center">0.616</td>
  </tr> 
  <tr>
    <td align="center">1.78B</td>
    <td align="center">G70M</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">0.721</td>
    <td align="center">0.765</td>
    <td align="center">0.662</td>
  </tr>
  </tbody>
</table>

> [!NOTE]
> The model trained on G70M won't be released due to substantial privacy and safety risks associated with [LAION](https://laion.ai/blog/laion-400-open-dataset/#disclaimer--content-warning)'s large-scale, untargeted content scraping.

### Downloading

The checkpoints can be downloaded from the links in the table above.

- For downloading from Google Drive, one option is to use [gdown](https://github.com/wkentaro/gdown) instead of the web browser:

  ```bash
  # install gdown toolkit
  pip install gdown

  # download the checkpoint in terminal
  gdown --fuzzy https://drive.google.com/file/d/1QYT7kXD9qks6rQh0m2PnVlnSffj8VXNh/view
  ```

- For downloading from Hugging Face, one option is to use [git-lfs](https://huggingface.co/docs/hub/models-downloading#using-git):

  ```bash
  # install git lfs
  git lfs install

  # download the checkpoint in terminal
  git clone https://huggingface.co/kaiyuyue/nxtp
  ```
  Also, the checkpoint can be downloaded from the [model page](https://huggingface.co/kaiyuyue/nxtp/tree/main) in the web browser.

## Inference

There is an image [assets/starbux.jpg](./assets/starbux.jpg) for a quick test.
First, please follow the instructions in [Dependencies](./docs/README.md#dependencies) to prepare the environment.

To infer an image, please run

```bash
python src/infer.py \
  --ckpt-path path/to/model/checkpoint \
  --img-path assets/starbux.jpg \
  --num-labels 20
```

The output from model trained on G3M will be

```bash
top-20 predictions:
| prob: 0.05742 - coffee
| prob: 0.05525 - restaurant
| prob: 0.04402 - shop
| prob: 0.02528 - room
| prob: 0.02468 - store
| prob: 0.02381 - interior
| prob: 0.01732 - area
| prob: 0.01640 - building
| prob: 0.01616 - food
| prob: 0.01408 - bar
| prob: 0.01247 - customer
| prob: 0.01134 - view
| prob: 0.01059 - floor
| prob: 0.01045 - table
| prob: 0.00933 - kitchen
| prob: 0.00926 - home
| prob: 0.00872 - look
| prob: 0.00841 - people
| prob: 0.00693 - cup
| prob: 0.00665 - counter
```

For reference, the output from model trained on G70M is

```bash
top-20 predictions:
| prob: 0.15203 - coffee
| prob: 0.09728 - shop
| prob: 0.09182 - counter
| prob: 0.03848 - interior
| prob: 0.03389 - bar
| prob: 0.03215 - restaurant
| prob: 0.02440 - table
| prob: 0.02245 - store
| prob: 0.01950 - area
| prob: 0.01905 - inside
| prob: 0.01590 - starbucks
| prob: 0.01313 - cafe
| prob: 0.01220 - chair
| prob: 0.01172 - floor
| prob: 0.01020 - cup
| prob: 0.00879 - drink
| prob: 0.00794 - room
| prob: 0.00746 - customer
| prob: 0.00635 - wood
| prob: 0.00345 - bakery
```

## Documentation

For training and evaluation, please see [docs](docs/README.md).

## License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.