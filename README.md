# I'm Me, We're Us, and I'm Us: Tri-directional Contrastive Learning on Hypergraphs
This repository contains the source code for the paper [I'm Me, We're Us, and I'm Us: Tri-directional Contrastive Learning on Hypergraphs](https://arxiv.org/abs/2206.04739), by [Dongjin Lee](https://github.com/wooner49) and [Kijung Shin](https://kijungs.github.io/).

In this paper, we propose **TriCL** (<ins>Tri</ins>-directional <ins>C</ins>ontrastive <ins>L</ins>earning), a general framework for contrastive learning on hypergraphs.
Its main idea is tri-directional contrast, and specifically, it aims to maximize in two augmented views the agreement (a) between the same node, (b) between the same group of nodes, and (c) between each group and its members. 
Together with simple but surprisingly effective data augmentation and negative sampling schemes, these three forms of contrast enable TriCL to capture both microscopic and mesoscopic structural information in node embeddings.
Our extensive experiments using 14 baseline approaches, 10 datasets, and two tasks demonstrate the effectiveness of TriCL, and most noticeably, TriCL almost consistently outperforms not just unsupervised competitors but also (semi-)supervised competitors mostly by significant margins for node classification. 


## Reference
This code is free and open source for only academic/research purposes (non-commercial).
If you use this code as part of any published research, please acknowledge the following paper.
```
@article{lee2022m,
  title={I'm Me, We're Us, and I'm Us: Tri-directional Contrastive Learning on Hypergraphs},
  author={Lee, Dongjin and Shin, Kijung},
  journal={arXiv preprint arXiv:2206.04739},
  year={2022}
}
```
