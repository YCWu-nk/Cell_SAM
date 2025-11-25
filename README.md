# Cell-SAM

Official PyTorch implementation and demonstration of <*DNA structural state as a label-free biomarker for intraoperative diagnosis of CNS tumors by AI-augmented third harmonic generation microscopy*>

<p float="right">
  <img src="cell-sam.png?raw=true" width="95%" />
</p>

## Introduction

The framework of Cell-SAM consists of four parts, including object detection module, image generation module, prompt SAM module, and fine-tuning SAM module.

## Environment

We suggest confirming that the basic environment is installed correctly first:
```
$ nvidia-smi
$ conda --version
$ nvcc -V
```
The installation is mainly composed of MMDetection. Please note that mmcv is a package that is prone to installation issues. When encountering issues, you can refer to [installation](https://mmdetection.readthedocs.io/en/latest/get_started.html).
```
$ conda create -n cell-sam python=3.8
$ conda activate cell-sam
$ conda install cudatoolkit=11.0
$ pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
$ pip install -U openmim
$ mim install mmengine
$ mim install mmcv==2.0.*
$ cd prompt_SAM; pip install -v -e .; cd ..
$ apt-get install git
$ pip install git+https://github.com/facebookresearch/segment-anything.git
```
Install other related packages using pip. 
```
$ pip install -r requirements.txt
```
If you encounter other uninstalled packages, please use pip to manually install them.
***NOTE*** the most prone to installation errors is 'mmcv'. Please check if the torch installation is GPU version instead of CPU version. If you encounter compilation errors, the best way is to install again using 'pip' [here](https://pytorch.org/get-started/previous-versions), like:
```
$ pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```

## Train and Test
Train your own dataset.

### Image Generation
Generate your dataset from pathological images using the "image_generation" file above\
Used for training and then obtaining weight 1：
```
$ python train.py --dataroot [path-to-dataset] --name [experiment-name] --mode sb \
--phase train --epoch [epoch-for-train] --eval --gpu_ids 0
```
For testing：
```
$ python test.py --dataroot [path-to-dataset] --name [experiment-name] --mode sb \
--phase test --epoch [epoch-for-test] --eval --num_test [num-test-image] \
--gpu_ids 0 --checkpoints_dir ./checkpoints/{weight 1}
```

### Pre-training
Use the MMPreTraining version of SparK, reference to [MMPreTraining](https://github.com/open-mmlab/mmpretrain).
Using weight 1 to generate images for training and then obtaining weight 2：
```
$ python tools/train.py ${CONFIG_FILE} [ARGS]
```
Testing：
```
$ python tools/test.py ${CONFIG_FILE} ${weight 2} [ARGS]
```

### Detection
Use the [MMDetection](https://github.com/open-mmlab/mmdetection), and the "prompt_SAM" file above.
Used weight 2 for training and then obtaining weight 3：
```
$ python tools/train.py ${CONFIG_FILE} [ARGS]
```
Testing：
```
$ python tools/test.py ${CONFIG_FILE} ${weight 3} [ARGS]
```

### Prompt SAM
Use the object detection model and weight 3:
```
$ cd mmdet_sam
```
```
$ python detector_sam_demo.py ${DATASET} ${CONFIG_FILE} ${weight 3} --sam-device cuda:0
```

### Finetune SAM
Finetune SAM using adapter, using the "finetune_SAM" file above, or refer to [SAM-Adapter](https://github.com/tianrun-chen/SAM-Adapter-PyTorch) 
```
$ python train.py --config configs/cod-sam-vit-b.yaml
```
```
$ python save.py --config configs/cod-sam-vit-b.yaml --model ${CONFIG_FILE} --save_dir {PATH}
```

## Demo
Directly use the trained model.

### Boxes-Prompted SAM Mode
Download the Demo dataset from [here](https://drive.google.com/drive/folders/1JcD8LF9rsgVToCnQGWb5i5SRY4eRTCaI?usp=sharing), Put them under the /Cell_SAM/prompt_SAM/mmdet_sam/datasam/ path.
Download the weight from [here](https://drive.google.com/file/d/1LvvXAwY4q_ELSWzvQc2_jTKddxI7fH13/view?usp=sharing), Put them under the /Cell_SAM/prompt_SAM/mmdet_sam/ path.
Enter in the command box opened in the /Cell_SAM/ path.
```
$ cd prompt_SAM/mmdet_sam
```
```
$ python detector_sam_demo.py datasam det4sam_spark_8xb32_r50-300e.py epoch_300.pth --sam-device cuda:0
```
The results saved in /Cell_SAM/prompt_SAM/mmdet_sam/outputs/.

### Fine-Tuned SAM Mode
Download the Demo dataset from [here](https://drive.google.com/drive/folders/1JcD8LF9rsgVToCnQGWb5i5SRY4eRTCaI?usp=sharing), Put them under the /Cell_SAM/finetune_SAM/load/Ima/ path.
Download the weight from [here](https://drive.google.com/file/d/1me0ptuTqTE2pWK0O88kHLm5SYPFjT05R/view?usp=sharing), Put them under the /Cell_SAM/finetune_SAM/ path.
Enter in the command box opened in the /Cell_SAM/ path.
```
$ cd finetune_SAM
```
```
$ python save.py --config configs/cod-sam-vit-b.yaml --model model_finetune.pth
```
The results saved in /Cell_SAM/finetune_SAM/visualizations/.

### Docker
Download the Docker image [here](https://drive.google.com/file/d/1AiiJNnYxzKex_3CaBe3K_tqG-EFLWIdC/view?usp=sharing). 
```
$ docker import - Cell-SAM < ‘cellsamV4.tar’	
```
We only mapped the input & output files of the Boxes-Prompted SAM, you can use this command to map more files.
```
$ docker run -it --gpus all -v ${your local file 1}:/prompt_SAM/mmdet_sam/datasam -v ${your local file 2}:/prompt_SAM/mmdet_sam/outputs ${id} /bin/bash
```

#### Boxes-Prompted SAM Mode
Put the Demo dataset at the corresponding path in {your local file 1}.
```
$ docker cp {your local path}/det4sam_spark_8xb32_r50-300e.py ${id}:/prompt_SAM/mmdet_sam 
$ docker cp {your local path}/epoch_300.pth ${id}:/prompt_SAM/mmdet_sam 
```
```
$ cd prompt_SAM/mmdet_sam/
```
```
$ python detector_sam_demo.py  datasam det4sam_spark_8xb32_r50-300e.py epoch_300.pth --sam-device cuda:0
```
The results saved in {your local file 2}.

#### Fine-Tuned SAM Mode
Using docker cp instruction to put the demo dataset in /finetune_SAM/load/Ima/, and put the [weight](https://drive.google.com/file/d/1me0ptuTqTE2pWK0O88kHLm5SYPFjT05R/view?usp=sharing) in finetune_SAM/.
```
python save.py --config configs/cod-sam-vit-b.yaml --model model_finetune.pth
```

#### Image Generation
Using docker cp instruction to put the demo dataset in image_generation/datasets/maps/testA/, and put the [weight](https://drive.google.com/file/d/1jZ1ks9W-ADopYmUougBA8jI6-Ia5U5Ue/view?usp=sharing) in image_generation/checkpoints/maps/, named latest_net_G.pth.
```
python test.py --dataroot ./datasets/maps --name maps --mode sb --lambda_SB 1.0 --lambda_NCE 1.0 --gpu_ids 0
```

### Utilize MMdetection metrics
Download the weight from [here](https://drive.google.com/file/d/1Nk0mGBh6eFSmAT9IJ7tfluPm3eq8kzHp/view?usp=sharing), Put them under the /Cell_SAM/prompt_SAM/ path.
Ensure that the number of 'num_classes' is correct in det4sam_spark_8xb32_r50-300e.py , and the content of 'classes' is correct in prompt_sam/mmdet/datasets/coco.py , When there are normal cells, it contains 1 , 2 and 3 ; when there are no normal cells, it contains 1 and 2 .
```
cd /prompt_sam/
```
```
python tools/test.py det4sam_spark_8xb32_r50-300e.py epoch_300.pth
```


## SAM2 branch
Demo of SAM2 version of Cell-SAM. Please use a suitable graphics card and CUDA version to be compatible with the SAM2 environment, otherwise please use our initial SAM version. Besides, the initial Cell-SAM can run on this higher-level version environment.

### Environment

```
$ conda create -n cell-sam2 python=3.10
$ conda activate cell-sam2
$ conda install --yes -c pytorch pytorch=2.0.1 torchvision=0.15.2
$ pip install -U openmim
$ mim install mmengine
$ mim install mmcv==2.0.*
$ cd prompt_sam; pip install -v -e .; cd ..
$ pip install ultralytics
```
Install other related packages using pip or conda.
```
$ pip install -r requirements.txt
```

### Boxes-Prompted SAM2 Mode
Download the weight from [here](https://drive.google.com/file/d/1LvvXAwY4q_ELSWzvQc2_jTKddxI7fH13/view?usp=sharing), Put them under the /Cell_SAM/prompt_SAM/mmdet_sam/ path.
Enter in the command box opened in the /Cell_SAM/ path.
```
$ cd prompt_SAM/mmdet_sam
$ python detector_sam2_demo.py datasam det4sam_spark_8xb32_r50-300e.py epoch_300.pth --sam-device cuda:0
```
The results saved in /Cell_SAM/prompt_SAM/mmdet_sam/outputs/.

## Data
The summary of open source files mentioned in the demo is as follows：\
THG data: https://drive.google.com/drive/folders/1JcD8LF9rsgVToCnQGWb5i5SRY4eRTCaI?usp=sharing \
Boxes-Prompted SAM weight: https://drive.google.com/file/d/1LvvXAwY4q_ELSWzvQc2_jTKddxI7fH13/view?usp=sharing \
Fine-Tuned SAM weight: https://drive.google.com/file/d/1me0ptuTqTE2pWK0O88kHLm5SYPFjT05R/view?usp=sharing \
Image Generation weight: https://drive.google.com/file/d/1jZ1ks9W-ADopYmUougBA8jI6-Ia5U5Ue/view?usp=sharing \

## Acknowledgement
For environment configuration, we used a workstation to fine-tune the the SAM and pre-trained object detection model, configured with Ubuntu 18.04, Python 3.9, PyTorch 1.9.0, CUDA 11.1, and an NVIDIA Tesla V100 (32GB) GPU. During the writing and testing of the lightweight code, we used a desktop computer configured with Windows 10, Python 3.6, and PyTorch 1.9.1, equipped with an NVIDIA GeForce RTX 3060 GPU. In addition, we tested the functionality of our code with two modes on more than 6 servers, including very different operating systems (Windows, Ubuntu, and Arch Linux), and with 4x1080Ti, 1xV100, 1x3080, 4x3090Ti, and 4x4090 GPUs. No performance degradation was observed in any of these environments or on any of the servers. In addition, the Docker image was packaged on Ubuntu 16.04, Python 3.8, and PyTorch 1.7.1. \
Thanks to the open source community for your good work! \

Recent updates：Oct 7,2024; Oct 9,2024; Dec 23,2024; March 3rd,2025. New installation notice on July 28th,2025. New submission 2025/11/25.
