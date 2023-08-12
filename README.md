# PAGCL code for ICDM 2023

### 1. Link Inference Attacks Results against GCA:
To obtain the link inference attack results against GCA (For example, on Cora datasets), please run unsuper_attacks.py by:

```python unsuper_attacks.py -- ```

### 2. Experimental Results of PAGCL:
The values of main parameters are restored in directory param.

To obtain the experimental results of PAGCL (For example, on Cora dataset), please run pagcl.py by:
python pagcl.py --dataset=Cora --weight=15 --topk=2 --drop_scheme=PAGCL