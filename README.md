# MMRGT
MGTAB: A Multi-Relational Graph-Based Twitter Account Detection Benchmark

## Introduction
MGTAB is the first standardized graph-based benchmark for stance and bot detection. MGTAB contains 10,199 expert-annotated users
and 7 types of relationships, ensuring high-quality annotation and diversified relations. For more details, please refer to the MGTAB paper.

## Adapted Dataset Link
Please download the dataset from the link present in Datasets and name the new folder inside Datasets exactly as TwiBot22-as-MGTAB-10k-new!


## Train Model
To start training process:
Train RGT Models
```shell script
python RGT_multimodal_feedforward.py --relation_select 0 1 2 3 4 4 6
python RGT_multimodal_crossmodal.py --relation_select 0 1 2 3 4 4 6
```
