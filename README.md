# Radio Frequency Signal Based Human Silhouette Segmentation: A Sequential Diffusion Approach


## Dataset

- Download the HIBER Dataset from [HIBER ( Human Indoor Behavior Exclusive RF dataset )](https://github.com/Intelligent-Perception-Lab/HIBER). 


## Conda Installation

``` bash
# 1. Create a conda virtual environment.
conda create -n SDM python=3.6
conda activate SDM
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

# 2. Clone the Repo and Install dependencies
git clone https://github.com/ph-w2000/SDM
pip install -r requirements.txt

```
## Method

<img src=Figures/architecture.png>

## Training 

This code supports multi-GPUs training.

  ```bash
python -m torch.distributed.launch --nproc_per_node=1 --master_port 48949 train.py -batch_size 8

  ```

- You can change the training hyperparameters in train.py file, such as dataset path, batch_size etc.
- To switch "WALK" dataset to "MULTI", you can change it in hiber_dataset.py file. 

## Inference 

  ```bash
python -m torch.distributed.launch --nproc_per_node=1 --master_port 48949 test.py -batch_size 8

  ```

- You can change the training hyperparameters in test.py file, such as dataset path, batch_size etc.
- To switch "WALK" dataset to "MULTI", you can change it in hiber_dataset.py file.

## Citation

If you use the results and code for your research, please cite our paper:

```

```

 
