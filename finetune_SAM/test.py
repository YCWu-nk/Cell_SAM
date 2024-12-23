import argparse
import os
import yaml 
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
from mmcv.runner import load_checkpoint
import numpy as np
from skimage import io, exposure
from PIL import Image 
import datasets
import models
import utils

def tensor2PIL(tensor):
    toPIL = transforms.ToPILImage()
    return toPIL(tensor)

def acfiji(image):
    if image.ndim == 3:  
        a_image = np.zeros_like(image)
        for i in range(3):
            a_image[:, :, i] = exposure.rescale_intensity(image[:, :, i], out_range=(0, 255)).astype(np.uint8)
    else:  
        a_image = exposure.rescale_intensity(image, out_range=(0, 255)).astype(np.uint8)
    return a_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def eval_psnr(loader, model, save_dir=None):
    model.eval()
    pbar = tqdm(loader, leave=False, desc='val')

    for idx, batch in enumerate(pbar):
        for k, v in batch.items():
            batch[k] = v.to(device)

        with torch.no_grad():
            inp = batch['inp']
            pred = torch.sigmoid(model.infer(inp))

            for i in range(inp.shape[0]):  
                pred_img = tensor2PIL(pred[i].cpu()) 
                pred_img_np = np.array(pred_img)  
                a_img_np = acfiji(pred_img_np) 
                a_img = Image.fromarray(a_img_np) 
                a_img.save(os.path.join(save_dir, f'pred_{idx}_{i}.png')) 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--model')
    parser.add_argument('--save_dir', default='visualizations', help='Directory to save visualizations')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    spec = config['test_dataset']
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    loader = DataLoader(dataset, batch_size=spec['batch_size'], 
                        num_workers=0)

    model = models.make(config['model']).to(device) 
    sam_checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(sam_checkpoint, strict=True)
    
    eval_psnr(loader, model, save_dir=args.save_dir)