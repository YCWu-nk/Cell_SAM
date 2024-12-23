# Cell-SAM

Official PyTorch implementation of <*Large-scale segmentation model facilitates intraoperative histopathology by third harmonic generation microscopy*> by Yuchen Wu, Sylvia Spies, Pieter Wesseling, Marie Louise Groot and Zhiqing Zhang.

<p float="right">
  <img src="cell-sam.png?raw=true" width="90%" />
</p>

## Note

The framework of Cell-SAM consists of four parts, including object detection module, image generation module, prompt SAM module, and fine-tuning SAM module.

## Environment

We suggest establishing an integrated installation package:
```
$ conda create -n cell-sam python=3.8
$ conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
$ pip install -U openmim
$ mim install mmengine
$ mim install mmcv==2.0.1
$ cd detection; pip install -v -e .; cd ..
$ pip install git+https://github.com/facebookresearch/segment-anything.git
```
Install other related packages using pip or conda.

## Train and Test
### Image Generation
Generate your dataset from pathological images using the "image_generation" file above\
Used for training and then obtaining weight 1：
```
$ python train.py --dataroot [path-to-dataset] --name [experiment-name] --mode sb \
--phase train --epoch [epoch-for-train] --eval --gpu_ids 0,1,2,3 
```
For testing：
```
$ python test.py --dataroot [path-to-dataset] --name [experiment-name] --mode sb \
--phase test --epoch [epoch-for-test] --eval --num_test [num-test-image] \
--gpu_ids 0 --checkpoints_dir ./checkpoints/weight 1
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
Use the [MMDetection](https://github.com/open-mmlab/mmdetection), and the "detection" file above.
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
$ python detector_sam_demo.py ${DATASET} ${CONFIG_FILE} ${weight 3} --not-show-label --sam-device cuda:0
```

### Finetune SAM
Finetune SAM using adapter, using the "finetune_SAM" file above, or refer to [SAM-Adapter](https://github.com/tianrun-chen/SAM-Adapter-PyTorch) 
```
$ python train.py --config \configs\cod-sam-vit-b.yaml
```
```
$ python test.py --config \configs\cod-sam-vit-b.yaml --model \${CONFIG_FILE} --save_dir /
```

## Docker
If you want to download our Docker container
```
docker import - Cell-SAM < ‘cell_sam.tar’	
docker run -it --gpus all -v ${your local file}:/mmdetection/mmdet_sam/datasam ${id} /bin/bash
```
```
cd /mmdetection/mmdet_sam/
python detector_sam_demo.py datasam ${CONFIG} ${weight 3} --not-show-label --sam-device cuda:0 
```

## Pre-trained Models
https://pan.baidu.com/s/1Jj7emLg0QenHY6xMPu7pWw?pwd=4i1c code: 4i1c

### Acknowledgement
Our source code is based on [UNSB](https://github.com/cyclomon/UNSB), [OpenMMLab](https://github.com/open-mmlab) and [SAM-Adapter](https://github.com/tianrun-chen/SAM-Adapter-PyTorch). \
We thank them for their good work! \
The code repository is currently being updated. Recent updates：Oct 7,2024; Oct 9,2024; Dec 23,2024.
