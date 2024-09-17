from pathlib import Path
import os
import torch
from PIL import Image
from os.path import basename
from os.path import splitext
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
import os
import random
from tqdm import tqdm
from util.utils import load_pretrained
import shutil

def select_random_images(root_dir, num_images, save_dir=None):
    images = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.jpg') or filename.endswith('.png'):
                images.append(os.path.join(dirpath, filename))

    if len(images) < num_images:
        print("Warning: Number of images in folder is less than required.")

    selected_images = random.sample(images, min(num_images, len(images)))
    
    if save_dir is not None:
        # Ensure the destination directory exists
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Copy selected images to the destination directory
        for image in selected_images:
            shutil.copy(image, save_dir)
    return selected_images


def test_transform(size, crop):
    transform_list = []
   
    if size != 0: 
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

def style_transform(h,w):
    k = (h,w)
    size = int(np.max(k))
    transform_list = []    
    transform_list.append(transforms.CenterCrop((h,w)))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

def content_transform():
    
    transform_list = []   
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


def eval(args):
    print("Name: ", args.model_name)
    # Advanced options
    content_size=args.img_size
    style_size=args.img_size
    crop='store_true'
    save_ext='.jpg'
    output_dir=args.output_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.content:
        content_paths = [Path(args.content)]
    else:
        content_dir = Path(args.content_dir)
        content_paths = [f for f in content_dir.glob('*')]

    if args.style:
        style_paths = [Path(args.style)]    
    else:
        style_dir = Path(args.style_dir)
        style_paths = [f for f in style_dir.glob('*')]

    random.seed(args.seed)
    num_images_to_select = 40
    style_paths = select_random_images(args.style_dir, num_images_to_select)
    num_images_to_select = 20
    content_paths = select_random_images(args.content_dir, num_images_to_select)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    network = load_pretrained(args)
    network.eval()
    network.to(device)

    content_tf = test_transform(content_size, crop)
    style_tf = test_transform(style_size, crop)
    content_loss = 0.0
    style_loss = 0.0
        
    for content_path in tqdm(content_paths):
        for style_path in tqdm(style_paths):
            content = content_tf(Image.open(content_path).convert("RGB"))
            style = style_tf(Image.open(style_path).convert("RGB"))
        
            style = style.to(device).unsqueeze(0)
            content = content.to(device).unsqueeze(0)
            
            with torch.no_grad():
                output, loss_c, loss_s, _, _, _, _ = network(content,style) 
        
            content_loss += loss_c
            style_loss += loss_s
            
            output_name = '{:s}/{:s}_stylized_{:s}{:s}'.format(
                output_dir, splitext(basename(content_path))[0],
                splitext(basename(style_path))[0], save_ext
            )
            
            if args.output_dir != "":
                save_image(output, output_name)
                
    print("Image size: ", args.img_size)
    print("Mamba")
    print(f"Content loss total: {content_loss.item()} - Style loss total: {style_loss.item()}")
    print(f"Content loss mean: {content_loss.item()/(len(content_paths)*len(style_paths))} - Style loss mean: {style_loss.item()/(len(content_paths)*len(style_paths))}")