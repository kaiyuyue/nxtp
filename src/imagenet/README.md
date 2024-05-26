## Experiments on ImageNet

Although our method is designed for open-world recognition, we also provide the code for experiments on ImageNet for reference.

### Create Image Loading List

Change the ImageNet dataset path in `tools/create_loading_list.py` and run:
```bash
cd imagenet/tools && python3 create_loading_list.py
```

### Training

```bash
./scripts/run train.py configs/config_in1k.py
```

### Evaluation

```bash
./scripts/run eval_hook.py evals/eval.py configs/config_in1k.py
```

### Results
||||||
|:----------:|:------------------------:|:-----:|:-----:|:------:|
| **params** | **training&nbsp;group**  | **R** | **P** | **F1** |
|      1.78B | ImageNet-1K              | 0.896 | 0.432 | 0.574  | 

The results are obtained by training the model on ImageNet-1K only and evaluating on the validation set.