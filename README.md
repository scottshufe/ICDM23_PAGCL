# PAGCL code for ICDM 2023

### 1. Link Inference Attacks Results against GCA:
To obtain the link inference attack results against GCA (for example, on Cora datasets), please run ```unsuper_attacks.py``` by:

```python unsuper_attacks.py --dataset -Cora --param -local:cora.json```

### 2. Experimental Results of PAGCL:
The values of main parameters are restored in directory ```param```.

To obtain the experimental results of PAGCL (for example, on Cora dataset), please run ```pagcl.py``` by:

```python pagcl.py --dataset=Cora --weight=5```