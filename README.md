# [IJCAI 2025] Exploring the Over-smoothing Problem of Graph Neural Networks for Graph Classification: An Entropy-based Viewpoint

<img src=".\Framework.png">  

## Dependencies

- Pytorch == 2.4.1 

- torch_geometric == 2.5.3

- networkx == 3.2.1
- scikit-learn
- tqdm
- scipy

## Quick Start

To run the training script with default settings:

```bash
bash run.sh
```
If you want to try other datasets or models, simply modify the run.sh file. For example, you can change the variables inside the script:
```
dataset="PROTEINS"
model="PairNorm"
```

## Datasets
 Datasets can be download from https://chrsmrrs.github.io/datasets/

## Citing

If you find this work is helpful to your research, please consider citing our paper:
```bibtex
@inproceedings{QianSDE,
  author       = {Feifei Qian and
                  Lu Bai and
                  Lixin Cui and
                  Ming Li and
                  Hangyuan Du and
                  Yue Wang and
                  Edwin R. Hancock},
  title        = {Exploring the Over-smoothing Problem of Graph Neural Networks for Graph Classification: An Entropy-based Viewpoint},
  booktitle    = {Proceedings of IJCAI},
  year         = {2025}
}
```


