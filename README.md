# PhyGCN

## Contents
- [Overview](#overview)
- [Software Requirements](#software_requirements)
- [Installation](#installation)
- [Usage Demo](#usage_demo)
    - [Pre-training with Hyperedge Prediction](#pre-training-with-hyperedge-prediction)
    - [Node Classification](#node-classification)

# Overview
Hypergraphs are powerful tools for modeling complex interactions across various domains, including biomedicine. However, learning meaningful node representations from hypergraphs remains a challenge. Existing supervised methods often lack generalizability, thereby limiting their real-world applications. We propose a new method, Pre-trained Hypergraph Convolutional Neural Networks with Self-supervised Learning (PhyGCN), which leverages hypergraph structure for self-supervision to enhance node representations. PhyGCN introduces a unique training strategy that integrates variable hyperedge sizes with self-supervised learning, enabling improved generalization to unseen data. Applications on multi-way chromatin interactions and polypharmacy side-effects demonstrate the effectiveness of PhyGCN.
As a generic framework for high-order interaction datasets with abundant unlabeled data, PhyGCN holds strong potential for enhancing hypergraph node representations across various domains.

# Software Requirements
This package is developed and tested on *Linux*.
+ Linux: Ubuntu 16.04

# Installation
```
conda create -n myenv python=3.6.8
conda activate myenv
pip install -r requirements.txt
```
The package should take less than 5 min to install.

# Usage Demo
Data is provided in [HNHN_data](HNHN_data) and pre-trained model checkpoints are provided in [checkpoints](checkpoints).
## Pre-training with Hyperedge Prediction
Example:
```
python pretrain.py --data pubmed --f conv --num-epoch 300 --dropedge 0.7 --layers 2
```
Note that our pre-trained checkpoints are provided and can be directly used for note classification.

## Node Classification
Example: 
```
python main.py --f conv --data  cora-cite --num-epoch 300 --layers 1 --split 1  
```
To get the results on 10 random train-test splits, use
```
chmod +x script.sh
./script.sh
```
The test accuracy on the 10 splits will be collected in result.txt

# Acknowledgement
This repo is built upon [Hyper-SAGNN](https://github.com/ma-compbio/Hyper-SAGNN). 
