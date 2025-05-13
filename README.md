# MMRGT
Multi-Modal Relational Graph Transformer for Bot-Detection


## Adapted Dataset Link
Please download the dataset from the link present in Datasets and name the new folder inside Datasets exactly as TwiBot22-as-MGTAB-10k-new!


## Train Model
To start training process:
Train RGT Models
```shell script
python RGT_multimodal_feedforward.py --relation_select 0 1 2 3 4 4 6
python RGT_multimodal_crossmodal.py --relation_select 0 1 2 3 4 4 6
```
