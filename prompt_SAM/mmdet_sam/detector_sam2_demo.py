import sys
import argparse
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import json
from collections import defaultdict
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry
from ultralytics import SAM
from mmengine.config import Config
from mmengine.utils import ProgressBar
from PIL import Image
sys.path.append('../')
from mmdet_sam.utils import apply_exif_orientation, get_file_list  

# Grounding DINO
try:
    import groundingdino
    import groundingdino.datasets.transforms as T
    from groundingdino.models import build_model
    from groundingdino.util import get_tokenlizer
    from groundingdino.util.utils import (clean_state_dict,
                                          get_phrases_from_posmap)
    grounding_dino_transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
except ImportError:
    groundingdino = None

# mmdet
try:
    import mmdet
    from mmdet.apis import inference_detector, init_detector
except ImportError:
    mmdet = None

# GLIP
try:
    import maskrcnn_benchmark

    from mmdet_sam.predictor_glip import GLIPDemo
except ImportError:
    maskrcnn_benchmark = None


def parse_args():
    parser = argparse.ArgumentParser(
        'Detect-Segment-Anything Demo', add_help=True)
    parser.add_argument('image', type=str, help='path to image directory or file')
    parser.add_argument('det_config', type=str, help='path to det config file')
    parser.add_argument('det_weight', type=str, help='path to det weight file')
    parser.add_argument('--only-det', action='store_true')
    parser.add_argument('--not-show-label', action='store_true')
    parser.add_argument(
        '--sam-type',
        type=str,
        default='vit_h',
        choices=['vit_h', 'vit_l', 'vit_b'],
        help='sam type')
    parser.add_argument(
        '--sam-weight',
        type=str,
        default='sam_vit_h_4b8939.pth',
        help='path to checkpoint file')
    parser.add_argument(
        '--out-dir',
        '-o',
        type=str,
        default='outputs',
        help='output directory')
    parser.add_argument(
        '--box-thr', '-b', type=float, default=0.4, help='box threshold')
    parser.add_argument(
        '--det-device',
        '-d',
        default='cuda:0',
        help='Device used for inference')
    parser.add_argument(
        '--sam-device',
        '-s',
        default='cuda:0',
        help='Device used for inference')
    parser.add_argument('--cpu-off-load', '-c', action='store_true')

    # Detic param
    parser.add_argument('--use-detic-mask', '-u', action='store_true')

    # GroundingDINO param
    parser.add_argument('--text-prompt', '-t', type=str, help='text prompt')
    parser.add_argument(
        '--text-thr', type=float, default=0.25, help='text threshold')
    parser.add_argument(
        '--apply-original-groudingdino',
        action='store_true',
        help='use original groudingdino label predict')

    # GLIP param
    parser.add_argument(
        '--apply-other-text',
        action='store_true',
        help='means use text prompt only conclude label name')
    return parser.parse_args()


def build_detecter(args):
    if 'GroundingDINO' in args.det_config:
        gdino_args = Config.fromfile(args.det_config)
        model = build_model(gdino_args)
        checkpoint = torch.load(args.det_weight, map_location='cpu')
        model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
        model.eval()
        return model
    elif 'glip' in args.det_config:
        assert maskrcnn_benchmark is not None
        from maskrcnn_benchmark.config import cfg
        cfg.merge_from_file(args.det_config)
        cfg.merge_from_list(['MODEL.WEIGHT', args.det_weight])
        cfg.merge_from_list(['MODEL.DEVICE', 'cpu'])
        model = GLIPDemo(
            cfg,
            min_image_size=800,
            confidence_threshold=args.box_thr,
            show_mask_heatmaps=False)
        return model
    else:
        config = Config.fromfile(args.det_config)
        if 'init_cfg' in config.model.backbone:
            config.model.backbone.init_cfg = None
        if 'detic' in args.det_config and not args.use_detic_mask:
            config.model.roi_head.mask_head = None
        detecter = init_detector(
            config, args.det_weight, device='cpu', cfg_options={})
        return detecter


def run_detector(model, image_path, args):
    pred_dict = {}
    if args.cpu_off_load:
        if 'glip' in args.det_config:
            model.model = model.model.to(args.det_device)
            model.device = args.det_device
        else:
            model = model.to(args.det_device)

    if 'GroundingDINO' in args.det_config:
        image_pil = Image.open(image_path).convert('RGB')  # load image
        image_pil = apply_exif_orientation(image_pil)
        image, _ = grounding_dino_transform(image_pil, None)  # 3, h, w

        text_prompt = args.text_prompt
        text_prompt = text_prompt.lower()
        text_prompt = text_prompt.strip()
        if not text_prompt.endswith('.'):
            text_prompt = text_prompt + '.'

        # Original GroundingDINO use text-thr to get class name,
        # the result will always result in categories that we don't want,
        # so we provide a category-restricted approach to address this

        if not args.apply_original_groudingdino:
            # custom label name
            custom_vocabulary = text_prompt[:-1].split('.')
            label_name = [c.strip() for c in custom_vocabulary]
            tokens_positive = []
            start_i = 0
            separation_tokens = ' . '
            for _index, label in enumerate(label_name):
                end_i = start_i + len(label)
                tokens_positive.append([(start_i, end_i)])
                if _index != len(label_name) - 1:
                    start_i = end_i + len(separation_tokens)
            tokenizer = get_tokenlizer.get_tokenlizer('bert-base-uncased')
            tokenized = tokenizer(
                args.text_prompt, padding='longest', return_tensors='pt')
            positive_map_label_to_token = create_positive_dict(
                tokenized, tokens_positive, list(range(len(label_name))))

        image = image.to(next(model.parameters()).device)

        with torch.no_grad():
            outputs = model(image[None], captions=[text_prompt])

        logits = outputs['pred_logits'].cpu().sigmoid()[0]  # (nq, 256)
        boxes = outputs['pred_boxes'].cpu()[0]  # (nq, 4)

        if not args.apply_original_groudingdino:
            logits = convert_grounding_to_od_logits(
                logits, len(label_name),
                positive_map_label_to_token)  # [N, num_classes]

        # filter output
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > args.box_thr
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

        if args.apply_original_groudingdino:
            # get phrase
            tokenlizer = model.tokenizer
            tokenized = tokenlizer(text_prompt)
            # build pred
            pred_labels = []
            pred_scores = []
            for logit, box in zip(logits_filt, boxes_filt):
                pred_phrase = get_phrases_from_posmap(logit > args.text_thr,
                                                      tokenized, tokenlizer)
                pred_labels.append(pred_phrase)
                pred_scores.append(str(logit.max().item())[:4])
        else:
            scores, pred_phrase_idxs = logits_filt.max(1)
            # build pred
            pred_labels = []
            pred_scores = []
            for score, pred_phrase_idx in zip(scores, pred_phrase_idxs):
                pred_labels.append(label_name[pred_phrase_idx])
                pred_scores.append(str(score.item())[:4])

                pred_dict['labels'] = pred_labels
        pred_dict['scores'] = pred_scores

        size = image_pil.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]
        pred_dict['boxes'] = boxes_filt
    elif 'glip' in args.det_config:
        image = cv2.imread(image_path)
        text_prompt = args.text_prompt
        text_prompt = text_prompt.lower()
        text_prompt = text_prompt.strip()
        if not text_prompt.endswith('.') and not args.apply_other_text:
            text_prompt = text_prompt + '.'

        custom_vocabulary = text_prompt[:-1].split('.')
        label_name = [c.strip() for c in custom_vocabulary]

        if args.apply_other_text:
            top_predictions = model.inference(
                image, args.text_prompt, use_other_text=True)
        else:
            top_predictions = model.inference(
                image, args.text_prompt, use_other_text=False)
        scores = top_predictions.get_field('scores').tolist()
        labels = top_predictions.get_field('labels').tolist()

        if args.apply_other_text:
            new_labels = []
            if model.entities and model.plus:
                for i in labels:
                    if i <= len(model.entities):
                        new_labels.append(model.entities[i - model.plus])
                    else:
                        new_labels.append('object')
            else:
                new_labels = ['object' for i in labels]
        else:
            new_labels = [label_name[i] for i in labels]

        pred_dict['labels'] = new_labels
        pred_dict['scores'] = scores
        pred_dict['boxes'] = top_predictions.bbox
    else:
        result = inference_detector(model, image_path)
        pred_instances = result.pred_instances[
            result.pred_instances.scores > args.box_thr]

        pred_dict['boxes'] = pred_instances.bboxes
        pred_dict['scores'] = pred_instances.scores.cpu().numpy().tolist()
        pred_dict['labels'] = [
            model.dataset_meta['classes'][label]
            for label in pred_instances.labels
        ]
        if args.use_detic_mask:
            pred_dict['masks'] = pred_instances.masks

    if args.cpu_off_load:
        if 'glip' in args.det_config:
            model.model = model.model.to('cpu')
            model.device = 'cpu'
        else:
            model = model.to('cpu')
    return model, pred_dict


def draw_and_save(image,
                  pred_dict,
                  save_path,
                  random_color=True,
                  show_label=False):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)

    with_mask = 'masks' in pred_dict
    labels = pred_dict['labels']
    scores = pred_dict['scores']

    bboxes = pred_dict['boxes'].cpu().numpy()
    
    for box, label, score in zip(bboxes, labels, scores):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        plt.gca().add_patch(
            plt.Rectangle((x0, y0),
                          w,
                          h,
                          edgecolor='green',
                          facecolor=(0, 0, 0, 0),
                          lw=2))

        if show_label:
            plt.gca().text(x0, y0, f'{label}: {score}', color='white')

    if with_mask:
        masks = pred_dict['masks'].cpu().numpy()
        for i, mask in enumerate(masks):
            color = np.random.rand(3)
            h, w = mask.shape[-2:]
            mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
            plt.gca().imshow(mask_image, alpha=0.5)

    plt.axis('off')
    plt.savefig(save_path)
    plt.close()


def main():
    if groundingdino is None and maskrcnn_benchmark is None and mmdet is None:
        raise RuntimeError('Detection model is not installed, please install it.')

    args = parse_args()
    if args.cpu_off_load and ('cpu' in args.det_device or 'cpu' in args.sam_device):
        raise RuntimeError('CPU offload is invalid when models are on CPU.')

    det_model = build_detecter(args)
    if not args.cpu_off_load:
        det_model = det_model.to(args.det_device)

    if not args.only_det:
        sam_model = SAM("sam2_s.pt")  # Initialize SAM model

    os.makedirs(args.out_dir, exist_ok=True)
    files, source_type = get_file_list(args.image)
    progress_bar = ProgressBar(len(files))

        
    for image_path in files:
        save_path = os.path.join(args.out_dir, os.path.basename(image_path))
        det_model, pred_dict = run_detector(det_model, image_path, args)

        if pred_dict['boxes'].shape[0] == 0:
            print(f'No objects detected in {image_path}!')
            continue

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if not args.only_det:
            if args.cpu_off_load:
                sam_model.to(args.sam_device)

            # Convert detected boxes to SAM input format
            bboxes = pred_dict['boxes'].cpu().numpy().tolist()
            results = sam_model(image, bboxes=bboxes, save=True, boxes=False)

        #draw_and_save(image, pred_dict, save_path, show_label=not args.not_show_label)
        progress_bar.update()

if __name__ == '__main__':
    main()