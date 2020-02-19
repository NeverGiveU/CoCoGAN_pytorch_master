import os
from tqdm import tqdm
#import matplotlib.pyplot as plt 
import torch
import argparse
import numpy as np
import math
import yaml
from torchvision import utils
from tqdm import tqdm

from model import StyledGenerator
from coord_handler import CoordHandler
from patch_handler import PatchHandler


def tensor2rgb(tensor):
    arr = tensor.data
    arr = np.array(arr).transpose(1, 2, 0)
    arr = arr/2 + 0.5
    return arr
    

def precompute_parameters(config):
    full_image_size = config["data_params"]["full_image_size"] 
    micro_patch_size = config["data_params"]["micro_patch_size"] 
    macro_patch_size = config["data_params"]["macro_patch_size"] 

    # Let NxM micro matches to compose a macro patch,
    #    `ratio_macro_to_micro` is N or M
    ratio_macro_to_micro = [
        macro_patch_size[0] // micro_patch_size[0],
        macro_patch_size[1] // micro_patch_size[1],
    ] 
    num_micro_compose_macro = ratio_macro_to_micro[0] * ratio_macro_to_micro[1] 

    # Let NxM micro matches to compose a full image,
    #    `ratio_full_to_micro` is N or M
    ratio_full_to_micro = [
        full_image_size[0] // micro_patch_size[0],
        full_image_size[1] // micro_patch_size[1],
    ] 
    num_micro_compose_full = ratio_full_to_micro[0] * ratio_full_to_micro[1] 

    config["data_params"]["ratio_macro_to_micro"] = ratio_macro_to_micro 
    config["data_params"]["ratio_full_to_micro"] = ratio_full_to_micro 
    config["data_params"]["num_micro_compose_macro"] = num_micro_compose_macro 
    config["data_params"]["num_micro_compose_full"] = num_micro_compose_full 
    config["train_params"]["batch_size"] = 1


def micros_to_macro(input, ratio_macro_to_micro):
    '''
    @input: <tsr> (B x num_micros_per_macro, 3, h, w)
    @ratio: <list>(n_row, n_col)
    '''
    n_row, n_col = ratio_macro_to_micro
    N = n_row * n_col

    patches_rows = []
    for i in range(n_row):
        patches_in_a_row = []
        for j in range(n_col):
            patches_in_a_row.append(input[i*n_row+j::N])
        patches_rows.append(torch.cat(patches_in_a_row, dim=-2))

    return torch.cat(patches_rows, dim=-1)
    

if __name__ == '__main__':
    ##
    CODE_SIZE = 256
    ## args
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    ## config
    with open(args.config) as f:
        config = yaml.load(f)#, Loader=yaml.FullLoader)

        # Basic protect. Otherwise, I don't know what will happen. OuO
        micro_size = config["data_params"]['micro_patch_size']
        macro_size = config["data_params"]['macro_patch_size']
        full_size = config["data_params"]['full_image_size']
        assert macro_size[0] % micro_size[0] == 0
        assert macro_size[1] % micro_size[1] == 0
        assert full_size[0] % micro_size[0] == 0
        assert full_size[1] % micro_size[1] == 0

    precompute_parameters(config)
    ratio_full_to_micro = config["data_params"]["ratio_full_to_micro"]
        
    ## generator and model
    model_pth = "checkpoint_coco/160000.model"
    n_layers = int(math.log(micro_size[0], 2)) - 1
    step = n_layers - 1
    
    generator = StyledGenerator(CODE_SIZE, n_layers).cuda()
    generator.load_state_dict(torch.load(model_pth))
    
    print("Successfully loading the trained models!")
    
    ##
    coord_handler = CoordHandler(config)
    
    # get micros coordinates for full image
    ROWS = ratio_full_to_micro[0]
    COLS = ratio_full_to_micro[1]
    micro_coords = []
    for row in range(ROWS):
        for col in range(COLS):
            micro_coord = torch.Tensor([coord_handler.euclidean_coord_int_full_to_float_micro(row, ROWS),
                                        coord_handler.euclidean_coord_int_full_to_float_micro(col, COLS)])
            micro_coords.append(micro_coord.unsqueeze(0))
            #print(micro_coord.unsqueeze(0).size())
    micro_coords = torch.cat(micro_coords, dim=0).cuda()

    ## data
    i = 0
    for i in tqdm(range(256)):
        input_style = torch.randn(1, CODE_SIZE-2).cuda()
        
        #print(input_style.repeat(ratio_full_to_micro[0]*ratio_full_to_micro[1], 1).size())
        #print(micro_coords.size())
        
        input = torch.cat([input_style.repeat(ratio_full_to_micro[0]*ratio_full_to_micro[1], 1), micro_coords], dim=1)
        #print(input)
        #break
        micros = generator(input, step=step)
        macro = micros_to_macro(micros, ratio_full_to_micro)
        
        utils.save_image(macro, r'test_coco/%06d.png'%i, nrow=1, normalize=True, range=(-1, 1))
        
    































    