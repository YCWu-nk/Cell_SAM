import argparse
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from mmdet.apis import inference_detector, init_detector
from mmengine.config import Config
from mmengine.utils import ProgressBar
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry
from utils import apply_exif_orientation, get_file_list


def parse_args():
    parser = argparse.ArgumentParser('Detect-Segment-Anything Demo', add_help=True)
    parser.add_argument('image', type=str, help='path to image file')
    parser.add_argument('det_config', type=str, help='path to det config file')
    parser.add_argument('det_weight', type=str, help='path to det weight file')
    parser.add_argument('--only-det', action='store_true')
    parser.add_argument('--not-show-label', action='store_true', default=True)
    parser.add_argument('--sam-type', type=str, default='vit_h', choices=['vit_h', 'vit_l', 'vit_b'], help='sam type')
    parser.add_argument('--sam-weight', type=str, default='sam_vit_h_4b8939.pth', help='path to checkpoint file')
    parser.add_argument('--out-dir', '-o', type=str, default='outputs', help='output directory')
    parser.add_argument('--box-thr', '-b', type=float, default=0.32, help='box threshold')
    parser.add_argument('--det-device', '-d', default='cuda:0', help='Device used for inference')
    parser.add_argument('--sam-device', '-s', default='cuda:0', help='Device used for inference')
    parser.add_argument('--cpu-off-load', '-c', action='store_true')
    parser.add_argument('--use-detic-mask', '-u', action='store_true')
    return parser.parse_args()

#Functions for constructing object detection models
def build_detecter(args):
    config = Config.fromfile(args.det_config)
    if 'init_cfg' in config.model.backbone:
        config.model.backbone.init_cfg = None
    if 'detic' in args.det_config and not args.use_detic_mask:
        config.model.roi_head.mask_head = None
    detecter = init_detector(config, args.det_weight, device='cpu', cfg_options={})
    return detecter

#Functions for running object detection models
def run_detector(model, image_path, args):
    pred_dict = {}
    result = inference_detector(model, image_path)
    pred_instances = result.pred_instances[result.pred_instances.scores > args.box_thr]

    pred_dict['boxes'] = pred_instances.bboxes
    pred_dict['scores'] = pred_instances.scores.cpu().numpy().tolist()
    pred_dict['labels'] = [model.dataset_meta['classes'][label] for label in pred_instances.labels]
    if args.use_detic_mask:
        pred_dict['masks'] = pred_instances.masks
    return model, pred_dict

#Function for drawing detection and segmentation results and saving images
def draw_and_save(image, pred_dict, save_path, random_color=True, show_label=False):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)

    with_mask = 'masks' in pred_dict
    labels = pred_dict['labels']
    scores = pred_dict['scores']

    bboxes = pred_dict['boxes'].cpu().numpy()

    for box, label, score in zip(bboxes, labels, scores):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        if show_label:
            plt.gca().text(x0, y0, f'{label}', color='white')

            
    if with_mask:
        masks = pred_dict['masks'].cpu().numpy()
        for i, mask in enumerate(masks):
            if pred_dict['labels'][i] == '1':
                color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
            else:
                color = np.array([255 / 255, 0 / 255, 138 / 255, 0.6])
            h, w = mask.shape[-2:]
            mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
            plt.gca().imshow(mask_image) 

    plt.axis('off')
    plt.savefig(save_path)
    plt.close()

#Main function
def main():
    args = parse_args()
    if args.cpu_off_load is True:
        if 'cpu' in args.det_device and 'cpu' in args.sam_device:
            raise RuntimeError('args.cpu_off_load is an invalid parameter due to detection and sam model are on the cpu.')

    only_det = args.only_det
    cpu_off_load = args.cpu_off_load
    out_dir = args.out_dir

    det_model = build_detecter(args)
    if not cpu_off_load:
        det_model = det_model.to(args.det_device)

    if not only_det:
        build_sam = sam_model_registry[args.sam_type]
        sam_model = SamPredictor(build_sam(checkpoint=args.sam_weight))
        if not cpu_off_load:
            sam_model.model = sam_model.model.to(args.sam_device)

    if 'Detic' in args.det_config:
        from projects.Detic.detic.utils import get_text_embeddings
        text_prompt = args.text_prompt
        text_prompt = text_prompt.lower()
        text_prompt = text_prompt.strip()
        if text_prompt.endswith('.'):
            text_prompt = text_prompt[:-1]
        custom_vocabulary = text_prompt.split('.')
        det_model.dataset_meta['classes'] = [c.strip() for c in custom_vocabulary]
        embedding = get_text_embeddings(custom_vocabulary=custom_vocabulary)
        __reset_cls_layer_weight(det_model, embedding)

    os.makedirs(out_dir, exist_ok=True)

    files, source_type = get_file_list(args.image)
    progress_bar = ProgressBar(len(files))
    for image_path in files:
        save_path = os.path.join(out_dir, os.path.basename(image_path))
        det_model, pred_dict = run_detector(det_model, image_path, args)

        if pred_dict['boxes'].shape[0] == 0:
            print('No objects detected !')
            continue

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if not only_det:
            if cpu_off_load:
                sam_model.model = sam_model.model.to(args.sam_device)
            sam_model.set_image(image)

            transformed_boxes = sam_model.transform.apply_boxes_torch(pred_dict['boxes'], image.shape[:2])
            transformed_boxes = transformed_boxes.to(sam_model.model.device)

            masks, _, _ = sam_model.predict_torch(point_coords=None, point_labels=None, boxes=transformed_boxes, multimask_output=False)
            pred_dict['masks'] = masks

            if cpu_off_load:
                sam_model.model = sam_model.model.to('cpu')

        draw_and_save(image, pred_dict, save_path, show_label=not args.not_show_label)
        progress_bar.update()


if __name__ == '__main__':
    main()