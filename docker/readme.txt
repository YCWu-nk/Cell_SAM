docker load -i cell_sam.tar	
docker run -it --gpus all -v ${your local file}:/mmdetection/mmdet_sam/datasam ${id} /bin/bash
cd /mmdetection/mmdet_sam/
python all614/detector_sam_demo.py datasam pre/rtmdet_l_8xb32_r50-300e.py pre/epoch_300.pth --not-show-label --sam-device cuda:0 


