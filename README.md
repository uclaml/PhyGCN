# PhyGCN

### Installing
```
conda create -n "myenv" python=3.6.8
pip install -r requirements.txt
```

### Pre-training
Example:
```
python pretrain.py --data pubmed --f conv --num-epoch 300 --dropedge 0.7 --layers 2
```

### To-do
- Update node classification codes.
- Update readme.