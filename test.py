
import os
from tqdm import tqdm
import matplotlib.pyplot as plt 
import torch
import argparse
import numpy as np
import math

from model import StyledGenerator

def tensor2rgb(tensor):
    arr = tensor.data
    arr = np.array(arr).transpose(1, 2, 0)
    arr = arr/2 + 0.5
    return arr


if __name__ == '__main__':
    SIZE = [256, 512, 1024]
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else torch.device('cpu'))
    CODE_SIZE = 512
    ######################################
    # simple testing                     #
    ######################################
    parser = argparse.ArgumentParser(description='Progressive Growing of GANs')
    parser.add_argument('--batch_size', type=int, default=1)

    opt = parser.parse_args()


    generator = StyledGenerator(CODE_SIZE).to(DEVICE)
    g_running = StyledGenerator(CODE_SIZE).to(DEVICE)

    cur_size = 512
    ## load models
    ckp_pth = os.path.join('checkpoint', 'stylegan-{}px-new.model'.format(cur_size))
    ckp1, ckp2 = torch.load(ckp_pth, map_location='cpu')['generator'], torch.load(ckp_pth, map_location='cpu')['g_running']
    
    generator.load_state_dict(ckp1)
    g_running.load_state_dict(ckp2)
    print("Successfully loading the trained models!")

    ## get step
    step = int(math.log2(cur_size)) - 2

    ## get inputs(styles)
    input_style = torch.randn(opt.batch_size, CODE_SIZE).to(DEVICE)
    
    ## output
    otuput_imgs = generator(input_style, step=step)
    output_img  = tensor2rgb(otuput_imgs[0, ...])

    plt.subplot(111)
    plt.imshow(output_img)
    plt.show()


