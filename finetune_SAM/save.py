import os
import yaml 
import torch
import cv2
import glob
from tqdm import tqdm
from torchvision import transforms
from mmengine.runner import load_checkpoint
import numpy as np
from skimage import exposure
import models
import utils

# Define a function to convert a tensor to a PIL image
def tensor2PIL(tensor):
    """
    Convert a PyTorch tensor to a PIL image.

    Args:
        tensor (torch.Tensor): The input PyTorch tensor.

    Returns:
        PIL.Image: The converted PIL image.
    """
    toPIL = transforms.ToPILImage()
    return toPIL(tensor)

# Define a function to adjust the intensity of an image
def acfiji(image):
    """
    Adjust the intensity range of the input image to be between 0 and 255.

    Args:
        image (np.ndarray): The input image, which can be single-channel or three-channel.

    Returns:
        np.ndarray: The intensity-adjusted image with data type uint8.
    """
    if image.ndim == 3:  
        # If it is a three-channel image, adjust the intensity of each channel separately
        a_image = np.zeros_like(image)
        for i in range(3):
            a_image[:, :, i] = exposure.rescale_intensity(image[:, :, i], out_range=(0, 255)).astype(np.uint8)
    else:  
        # If it is a single-channel image, directly adjust the intensity
        a_image = exposure.rescale_intensity(image, out_range=(0, 255)).astype(np.uint8)
    return a_image

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the image resizing transformation
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((1024, 1024)),
    transforms.ToTensor()
])

# Define the evaluation function
def eval_psnr(image_paths, model, save_dir=None):
    """
    Evaluate the model on a set of images and save the predictions.

    Args:
        image_paths (list): A list of paths to the input images.
        model (torch.nn.Module): The trained model.
        save_dir (str, optional): The directory to save the predicted images. Defaults to None.
    """
    # Set the model to evaluation mode
    model.eval()
    # Create a progress bar
    pbar = tqdm(image_paths, leave=False, desc='val')

    for image_path in pbar:
        # Get the image name
        image_name = os.path.basename(image_path)
        # Read the image
        image = cv2.imread(image_path)
        # Convert the image from BGR to RGB format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Apply the transformation and move the image to the device
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            # Perform inference on the image
            pred = torch.sigmoid(model.infer(image))
            # Convert the prediction tensor to a NumPy array and move it to the CPU
            pred_np = pred.cpu().squeeze().numpy()
            # Adjust the intensity of the predicted image
            a_img_np = acfiji(pred_np)
            # Define the save path for the predicted image
            save_path = os.path.join(save_dir, image_name)
            # Convert the predicted image back to BGR format and save it
            cv2.imwrite(save_path, cv2.cvtColor(a_img_np, cv2.COLOR_RGB2BGR))

if __name__ == '__main__':
    import argparse
    # Create an argument parser
    parser = argparse.ArgumentParser()
    # Add an argument for the configuration file path
    parser.add_argument('--config')
    # Add an argument for the model checkpoint path
    parser.add_argument('--model')
    # Add an argument for the directory containing input images
    parser.add_argument('--image_dir', default='load/Ima', help='Directory containing input images')
    # Add an argument for the directory to save visualizations
    parser.add_argument('--save_dir', default='visualizations', help='Directory to save visualizations')
    # Parse the command-line arguments
    args = parser.parse_args()

    # Open and load the configuration file
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Load the model and move it to the device
    model = models.make(config['model']).to(device) 
    # Load the model checkpoint
    sam_checkpoint = torch.load(args.model, map_location=device)
    # Load the model state dictionary
    model.load_state_dict(sam_checkpoint, strict=True)

    # Get the paths of all JPEG images in the input directory
    image_paths = glob.glob(os.path.join(args.image_dir, '*.jpg'))

    # Perform the evaluation
    eval_psnr(image_paths, model, save_dir=args.save_dir)