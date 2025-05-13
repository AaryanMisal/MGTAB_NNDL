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

## Requirements
You may just do 
```shell script
pip install -r requirements.txt
```

pandas
sentence-transformers
ijson
torch==2.1.0
torchvision==0.16.0
torchaudio==2.1.0
torch-geometric==2.4.0
scikit-learn==1.0.2
numpy==1.23.5
pytorch-lightning==1.9.5
matplotlib
