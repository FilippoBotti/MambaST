import os
import torch
import torch.nn as nn
import torch.utils.data as data
from PIL import Image
from tensorboardX import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
from pathlib import Path
import models.MambaST  as MambaST 
import models.mamba  as Mamba 
import models.models_helper as models_helper
from sampler import InfiniteSamplerWrapper
from torchvision.utils import save_image
import numpy as np
import random
from util.utils import load_pretrained

def select_random_images(root_dir, num_images):
    images = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.jpg') or filename.endswith('.png'):
                images.append(os.path.join(dirpath, filename))

    if len(images) < num_images:
        print("Warning: Number of images in folder is less than required.")

    selected_images = random.sample(images, min(num_images, len(images)))
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
  
def test_transform(size, crop):
    transform_list = []
   
    if size != 0: 
        transform_list.append(transforms.Resize(size, antialias=True))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        self.path = os.listdir(self.root)
        if os.path.isdir(os.path.join(self.root,self.path[0])):
            self.paths = []
            for file_name in os.listdir(self.root):
                for file_name1 in os.listdir(os.path.join(self.root,file_name)):
                    self.paths.append(self.root+"/"+file_name+"/"+file_name1)             
        else:
            self.paths = list(Path(self.root).glob('*'))
        self.transform = transform
        
    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(str(path)).convert('RGB')
        img = self.transform(img)
        return img
    
    def __len__(self):
        return len(self.paths)
    
    def name(self):
        return 'FlatFolderDataset'

def adjust_learning_rate(args, optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = 2e-4 / (1.0 + args.lr_decay * (iteration_count - 1e4))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def warmup_learning_rate(args, optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr * 0.1 * (1.0 + 3e-4 * iteration_count)
    # print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(args):
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")

    if not os.path.exists(args.checkpoints_dir):
        os.makedirs(args.checkpoints_dir)
        
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)

    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
    writer = SummaryWriter(args.log_dir + "/" + args.model_name)

    vgg = models_helper.vgg
    vgg.load_state_dict(torch.load(args.vgg))
    vgg = nn.Sequential(*list(vgg.children())[:44])

    decoder = models_helper.decoder
    embedding = models_helper.PatchEmbed()

    mamba = Mamba.Mamba(args=args)
    
    with torch.no_grad():
        network = MambaST.MambaST(vgg,decoder,embedding,mamba,args)
        
    if args.continue_train:
        network = load_pretrained(args)
        
    network.train()

    network.to(device)
    content_tf = train_transform()
    style_tf = train_transform()


    content_dataset = FlatFolderDataset(args.content_dir, content_tf)
    style_dataset = FlatFolderDataset(args.style_dir, style_tf)

    content_iter = iter(data.DataLoader(
        content_dataset, batch_size=args.batch_size,
        sampler=InfiniteSamplerWrapper(content_dataset),
        num_workers=args.n_threads))
    style_iter = iter(data.DataLoader(
        style_dataset, batch_size=args.batch_size,
        sampler=InfiniteSamplerWrapper(style_dataset),
        num_workers=args.n_threads))

    
    optimizer = torch.optim.Adam([ 
                                {'params': network.mamba.parameters()},
                                {'params': network.decode.parameters()},
                                {'params': network.embedding.parameters()},        
                                ], lr=args.lr)


    if not os.path.exists(args.results_dir+"/test"):
        os.makedirs(args.results_dir+"/test")
    if not os.path.exists(args.results_dir+"/eval"):
        os.makedirs(args.results_dir+"/eval")

    if args.resume_iter != 0:
        print(f"Resume iter:{args.resume_iter}")

    mean_loss_c = 0
    mean_loss_s = 0
    mean_l_identity1 = 0
    mean_l_identity2 = 0

    print("Network name: ", args.model_name)

    with tqdm(range(args.resume_iter, args.max_iter)) as titer:
        for i in titer:
            titer.set_description(f"Iter {i}")
            content_images = next(content_iter).to(device)
            style_images = next(style_iter).to(device)  
            out, loss_c, loss_s,l_identity1, l_identity2, content_out, style_out = network(content_images, style_images)

            if i % 100 == 0:
                output_name = '{:s}/test/{:s}{:s}'.format(
                                args.results_dir, str(i),".jpg"
                            )
                out = torch.cat((content_images,out),0)
                out = torch.cat((style_images,out),0)
                save_image(out, output_name)
                
                output_name = '{:s}/test/content_{:s}{:s}'.format(
                                args.results_dir, str(i),".jpg"
                            )
                content_out = torch.cat((content_images,content_out),0)
                content_out = torch.cat((content_images,content_out),0)
                save_image(content_out, output_name)
                
                output_name = '{:s}/test/style_{:s}{:s}'.format(
                                args.results_dir, str(i),".jpg"
                            )
                style_out = torch.cat((style_images,style_out),0)
                style_out = torch.cat((style_images,style_out),0)
                save_image(style_out, output_name)

                
            loss_c = args.content_weight * loss_c
            loss_s = args.style_weight * loss_s
            l_identity1 = args.l1_weight * l_identity1
            l_identity2 = args.l2_weight * l_identity2
            loss = loss_c + loss_s + l_identity1 + l_identity2 # - l_diversity *0.1

            mean_loss_c += loss_c 
            mean_loss_s += loss_s
            mean_l_identity1 += l_identity1
            mean_l_identity2 += l_identity2
            titer.set_postfix(content=loss_c.item() , style=loss_s.item(),
                            l1=l_identity1.item(), l2=l_identity2.item())
            
            if (i+1) % args.print_every == 0:
                mean_loss_c = mean_loss_c/args.print_every
                mean_loss_s = mean_loss_s/args.print_every
                mean_l_identity1 = mean_l_identity1/args.print_every
                mean_l_identity2 = mean_l_identity2/args.print_every
                mean_total_loss = mean_loss_c + mean_loss_s + mean_l_identity1 + mean_l_identity2

                print(f"Iter {i+1} - Total loss: {mean_total_loss} - Content loss: {mean_loss_c} - Style loss: {mean_loss_s} - l1: {mean_l_identity1} - l2: {mean_l_identity2}", flush=True)
                mean_loss_c = 0
                mean_loss_s = 0
                mean_l_identity1 = 0
                mean_l_identity2 = 0
        
        
            optimizer.zero_grad()
            loss.sum().backward()
            optimizer.step()

            writer.add_scalar('loss_content', loss_c.sum().item(), i + 1)
            writer.add_scalar('loss_style', loss_s.sum().item(), i + 1)
            writer.add_scalar('loss_identity1', l_identity1.sum().item(), i + 1)
            writer.add_scalar('loss_identity2', l_identity2.sum().item(), i + 1)
            writer.add_scalar('total_loss', loss.sum().item(), i + 1)

            if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
                state_dict = network.mamba.state_dict()
                for key in state_dict.keys():
                    state_dict[key] = state_dict[key].to(torch.device('cpu'))
                torch.save(state_dict,
                        '{:s}/mamba_iter_{:d}.pth'.format(args.checkpoints_dir,
                                                                i + 1))

                state_dict = network.decode.state_dict()
                for key in state_dict.keys():
                    state_dict[key] = state_dict[key].to(torch.device('cpu'))
                torch.save(state_dict,
                        '{:s}/decoder_iter_{:d}.pth'.format(args.checkpoints_dir,
                                                                i + 1))
                state_dict = network.embedding.state_dict()
                for key in state_dict.keys():
                    state_dict[key] = state_dict[key].to(torch.device('cpu'))
                torch.save(state_dict,
                        '{:s}/embedding_iter_{:d}.pth'.format(args.checkpoints_dir,
                                                                i + 1))
    writer.close()

        