import argparse
import os
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from skimage import exposure
import numpy as np

import datasets
import models
import utils

from torchvision import transforms
from mmengine.runner import load_checkpoint

# Convert a PyTorch tensor to a PIL image
def tensor2PIL(tensor):
    toPIL = transforms.ToPILImage()
    return toPIL(tensor)

# Use GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Adjust the intensity range of the input image to be between 0 and 255.
def acfiji(image):
    """
    Adjust the intensity range of the input image to be between 0 and 255.

    Args:
        image (np.ndarray): The input image, which can be single-channel or three-channel.

    Returns:
        np.ndarray: The intensity-adjusted image with data type uint8.
    """
    # If it is a three-channel image, adjust the intensity of each channel separately
    if image.ndim == 3:  
        a_image = np.zeros_like(image)
        for i in range(3):
            a_image[:, :, i] = exposure.rescale_intensity(image[:, :, i], out_range=(0, 255)).astype(np.uint8)
    # If it is a single-channel image, directly adjust the intensity
    else:  
        a_image = exposure.rescale_intensity(image, out_range=(0, 255)).astype(np.uint8)
    return a_image

# Evaluate PSNR and save the predicted images
def eval_psnr(loader, model, save_dir=None, filenames=None):
    # Set the model to evaluation mode
    model.eval()
    # Create a progress bar for the validation process
    pbar = tqdm(loader, leave=False, desc='val')

    for idx, batch in enumerate(pbar):
        # Move the batch data to the specified device (GPU or CPU)
        for k, v in batch.items():
            batch[k] = v.to(device)

        with torch.no_grad():
            # Get the input data from the batch
            inp = batch['inp']
            # Make predictions using the model and apply the sigmoid function
            pred = torch.sigmoid(model.infer(inp))

            for i in range(inp.shape[0]):
                # Convert the predicted tensor to a PIL image
                pred_img = tensor2PIL(pred[i].cpu())
                # Convert the PIL image to a numpy array
                pred_np = np.array(pred_img)
                # Apply automatic contrast adjustment to the numpy array
                adjusted_pred_np = acfiji(pred_np)
                # Convert the adjusted numpy array back to a PIL image
                adjusted_pred_img = transforms.ToPILImage()(adjusted_pred_np)
                # Get the corresponding file name
                filename = filenames[idx * loader.batch_size + i]
                # Save the predicted image with the original file name
                adjusted_pred_img.save(os.path.join(save_dir, filename))

if __name__ == '__main__':
    # Create an argument parser
    parser = argparse.ArgumentParser()
    # Add an argument for the configuration file
    parser.add_argument('--config')
    # Add an argument for the pre - trained model file
    parser.add_argument('--model')
    # Add an argument for the directory to save visualizations, with a default value
    parser.add_argument('--save_dir', default='visualizations', help='Directory to save visualizations')
    # Parse the command - line arguments
    args = parser.parse_args()

    # Open and load the configuration file
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    # Get the test dataset configuration
    spec = config['test_dataset']
    # Create the test dataset
    dataset = datasets.make(spec['dataset'])
    # Wrap the test dataset
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})

    # Get the root path of the test dataset images
    test_image_root = spec['dataset']['args']['root_path_1']
    # Get all image file names in sorted order
    filenames = sorted(os.listdir(test_image_root))

    # Create a data loader for the test dataset
    loader = DataLoader(dataset, batch_size=spec['batch_size'], num_workers=0)

    # Create the model and move it to the specified device
    model = models.make(config['model']).to(device)
    # Load the pre - trained model checkpoint
    sam_checkpoint = torch.load(args.model, map_location=device)
    # Load the model state dictionary
    model.load_state_dict(sam_checkpoint, strict=True)

    # Create the save directory if it does not exist
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Evaluate PSNR and save the predicted images
    eval_psnr(loader, model, save_dir=args.save_dir, filenames=filenames)
