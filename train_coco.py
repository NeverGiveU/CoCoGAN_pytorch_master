import argparse
import random
import math
import yaml
import itertools

from tqdm import tqdm
import numpy as np
from PIL import Image

import torch
import numpy as np 
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

from dataset import MultiResolutionDataset_imgs as MultiResolutionDataset
from model import StyledGenerator, ConditionalDiscriminatorH, SpatialCoordinatePredictor
from coord_handler import CoordHandler
from patch_handler import PatchHandler


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
    

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)


def sample_data(dataset, batch_size, image_size=4):
    dataset.resolution = image_size
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=1)

    return loader


def adjust_lr(optimizer, lr):
    for group in optimizer.param_groups:
        mult = group.get('mult', 1)
        group['lr'] = lr * mult


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
        patches_rows.append(torch.cat(patches_in_a_row, dim=3))

    return torch.cat(patches_rows, dim=2)


def train(args, generator, discriminator):
    step = int(math.log2(args.max_size)) - 2 #-> 1
    resolution = 4 * 2 ** step
    batch_size = args.batch.get(resolution, args.batch_default)
    dataset = MultiResolutionDataset(args.path, transform, resolution=resolution)
    
    loader = sample_data(
        dataset, batch_size, resolution
    )
    data_loader = iter(loader)

    adjust_lr(g_optimizer, args.lr.get(resolution, 0.001))
    adjust_lr(d_optimizer, args.lr.get(resolution, 0.001))

    pbar = tqdm(range(3000000))

    requires_grad(generator, False)
    requires_grad(discriminator, True)

    disc_loss_val = 0
    gen_loss_val = 0
    grad_loss_val = 0

    alpha = 0
    used_sample = 0 #-> how many images has been used

    max_step = int(math.log2(args.max_size)) - 2 #-> log2(1024) - 2 = 8
    final_progress = False

    for i in pbar:
        discriminator.zero_grad()

        alpha = min(1, 1 / args.phase * (used_sample + 1)) #-> min(1, (cur+1)/60_0000)
        #-> when more than 60_0000 sampels is used, alpha will be in const to 1.0
        #-> which means we the "skip_rgb" will not be applied

        if (resolution == args.init_size and args.ckpt is None) or final_progress:
            alpha = 1
        #-> also, if initially, no previous outputs for skip-connection

        if used_sample > args.phase * 2: #-> if > 1_200_000
            ## num_of_epoch_each_phase = args.phase * 2 / training_dataset_size
            used_sample = 0
            
            step += 1

            if step > max_step:
                step = max_step
                final_progress = True
                ckpt_step = step + 1

            else:
                alpha = 0
                ckpt_step = step
            

            resolution = 4 * 2 ** step_D

            loader = sample_data(
                dataset, args.batch.get(resolution, args.batch_default), resolution
            )
            data_loader = iter(loader)

            torch.save(
                {
                    'generator': generator.module.state_dict(),
                    'discriminator': discriminator.module.state_dict(),
                    'g_optimizer': g_optimizer.state_dict(),
                    'd_optimizer': d_optimizer.state_dict(),
                    'g_running': g_running.state_dict(),
                }, r'checkpoint_coco/train_step-{}.model'.format(ckpt_step))

            adjust_lr(g_optimizer, args.lr.get(resolution, 0.001))
            adjust_lr(d_optimizer, args.lr.get(resolution, 0.001))

        #### update discriminator
        try:
            real_image = next(data_loader)

        except (OSError, StopIteration):
            data_loader = iter(loader)
            real_image = next(data_loader)

        used_sample += real_image.shape[0]
        real_image = real_image.cuda()

        b_size = real_image.size(0)
        select = np.hstack([[i*b_size+j for i in range(num_micro_in_macro)] for j in range(b_size)])
        # get sample coords
        coord_handler.batch_size = b_size
        patch_handler.batch_size = b_size
        d_macro_coord_real, g_micro_coord_real, _ = coord_handler._euclidean_sample_coord()
        d_macro_coord_fake1, g_micro_coord_fake1, _ = coord_handler._euclidean_sample_coord()
        d_macro_coord_fake2, g_micro_coord_fake2, _ = coord_handler._euclidean_sample_coord()
        
        d_macro_coord_real = torch.from_numpy(d_macro_coord_real).float().cuda()
        d_macro_coord_fake1, g_micro_coord_fake1 = torch.from_numpy(d_macro_coord_fake1).float().cuda(), torch.from_numpy(g_micro_coord_fake1).float().cuda()
        d_macro_coord_fake2, g_micro_coord_fake2 = torch.from_numpy(d_macro_coord_fake2).float().cuda(), torch.from_numpy(g_micro_coord_fake2).float().cuda()

        real_macro = micros_to_macro(patch_handler.crop_micro_from_full_gpu(real_image, g_micro_coord_real[:, 1:2], g_micro_coord_real[:, 0:1]), config["data_params"]["ratio_macro_to_micro"])
        
        if args.loss == 'wgan-gp':
            real_predict, real_H = discriminator(real_macro, d_macro_coord_real, step=step_D, alpha=alpha)
            real_predict = real_predict.mean() - 0.001 * (real_predict ** 2).mean()
            
            sp_loss_real = criterion_mse(spatial_predictor(real_H), d_macro_coord_real) * coord_loss_w
            (-real_predict+sp_loss_real).backward()

        elif args.loss == 'r1':
            real_macro.requires_grad = True
            real_scores, real_H = discriminator(real_macro, d_macro_coord_real, step=step_D, alpha=alpha)
            real_predict = F.softplus(-real_scores).mean()
            sp_loss_real = criterion_mse(spatial_predictor(real_H), d_macro_coord_real) * coord_loss_w
            (real_predict+sp_loss_real).backward(retain_graph=True)

            grad_real = grad(
                outputs=real_scores.sum(), inputs=real_macro, create_graph=True
            )[0]
            grad_penalty = (
                grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2
            ).mean()
            grad_penalty = 10 / 2 * grad_penalty
            grad_penalty.backward()
            if i%10 == 0:
                grad_loss_val = grad_penalty.item()

        if args.mixing and random.random() < 0.9:
            gen_in11, gen_in12, gen_in21, gen_in22 = torch.randn(
                4, b_size, code_size-2, device='cuda'
            ).chunk(4, 0)
            
            gen_in11 = gen_in11.squeeze(0)
            gen_in11 = torch.cat([gen_in11.repeat(num_micro_in_macro, 1)[select], g_micro_coord_fake1], dim=1)
            
            gen_in12 = gen_in12.squeeze(0)
            gen_in12 = torch.cat([gen_in12.repeat(num_micro_in_macro, 1)[select], g_micro_coord_fake1], dim=1)
            
            gen_in21 = gen_in21.squeeze(0)
            gen_in21 = torch.cat([gen_in21.repeat(num_micro_in_macro, 1)[select], g_micro_coord_fake2], dim=1)
            
            gen_in22 = gen_in22.squeeze(0)
            gen_in22 = torch.cat([gen_in22.repeat(num_micro_in_macro, 1)[select], g_micro_coord_fake2], dim=1)
            
            gen_in1 = [gen_in11, gen_in12]
            gen_in2 = [gen_in21, gen_in22]
            
            #print(gen_in11[:16])

        else:
            gen_in1, gen_in2 = torch.randn(2, b_size, code_size-2, device='cuda').chunk(
                2, 0                                  # 512
            )
            gen_in1 = gen_in1.squeeze(0)# (B, 254)
            gen_in2 = gen_in2.squeeze(0)# (B, 254)

            # repeat and copy
            gen_in1 = torch.cat([gen_in1.repeat(num_micro_in_macro, 1)[select], g_micro_coord_fake1], dim=1)
            gen_in2 = torch.cat([gen_in2.repeat(num_micro_in_macro, 1)[select], g_micro_coord_fake2], dim=1)
        
        fake_image = generator(gen_in1, step=step_G, alpha=alpha)
        fake_image = micros_to_macro(fake_image, config["data_params"]["ratio_macro_to_micro"])
        fake_predict, fake_H = discriminator(fake_image, d_macro_coord_fake1, step=step_D, alpha=alpha)
        sp_loss_fake = criterion_mse(spatial_predictor(fake_H), d_macro_coord_fake1) * coord_loss_w

        if args.loss == 'wgan-gp':
            fake_predict = fake_predict.mean()
            (fake_predict+sp_loss_fake).backward()

            eps = torch.rand(b_size, 1, 1, 1).cuda()
            x_hat = eps * real_image.data + (1 - eps) * fake_image.data
            x_hat.requires_grad = True
            hat_predict = discriminator(x_hat, step=step_D, alpha=alpha)
            grad_x_hat = grad(
                outputs=hat_predict.sum(), inputs=x_hat, create_graph=True
            )[0]
            grad_penalty = (
                (grad_x_hat.view(grad_x_hat.size(0), -1).norm(2, dim=1) - 1) ** 2
            ).mean()
            grad_penalty = 10 * grad_penalty
            grad_penalty.backward()
            if i%10 == 0:
                grad_loss_val = grad_penalty.item()
                disc_loss_val = (real_predict - fake_predict).item()

        elif args.loss == 'r1':
            fake_predict = F.softplus(fake_predict).mean()
            (fake_predict+sp_loss_fake).backward()
            if i%10 == 0:
                disc_loss_val = (real_predict + fake_predict).item()

        d_optimizer.step()
        if i%10 == 0:
            spatial_loss_D_val = (sp_loss_real.item() + sp_loss_fake.item()) / 2


        #### update generator
        if (i + 1) % n_critic == 0:
            generator.zero_grad()

            requires_grad(generator, True)
            requires_grad(discriminator, False)

            fake_image = generator(gen_in2, step=step_G, alpha=alpha)
            fake_image = micros_to_macro(fake_image, config["data_params"]["ratio_macro_to_micro"])
            predict, H = discriminator(fake_image, d_macro_coord_fake2, step=step_D, alpha=alpha)
            spatial_loss = criterion_mse(spatial_predictor(H), d_macro_coord_fake2) * coord_loss_w

            if args.loss == 'wgan-gp':
                loss = -predict.mean()

            elif args.loss == 'r1':
                loss = F.softplus(-predict).mean()

            if i%10 == 0:
                gen_loss_val = loss.item()
                spatial_loss_G_val = spatial_loss.item()

            (loss+spatial_loss).backward()
            g_optimizer.step()
            accumulate(g_running, generator.module)

            requires_grad(generator, False)
            requires_grad(discriminator, True)


        #### validation
        if (i + 1) % 100 == 0:
            images = []

            gen_i, gen_j = args.gen_sample.get(resolution, (10, 5))
            
            coord_handler.batch_size = gen_i * gen_j
            _, g_micro_coord_val, _ = coord_handler._euclidean_sample_coord()
            g_micro_coord_val = torch.from_numpy(g_micro_coord_val).float().cuda()
            #print(g_micro_coord_val.shape)
            
            select = np.hstack([[i*gen_j+j for i in range(num_micro_in_macro)] for j in range(gen_j)])

            with torch.no_grad():
                for ii in range(gen_i):
                    style = torch.randn(gen_j, code_size-2).cuda().repeat(num_micro_in_macro, 1)[select]
                    #print(style.size())
                    coords = g_micro_coord_val[ii*gen_j*num_micro_in_macro:(ii+1)*gen_j*num_micro_in_macro]
                    #print(coords.size())
                    style = torch.cat([style, coords], dim=1)
                    
                    image = g_running(style, step=step_G, alpha=alpha).data.cpu()
                    image = micros_to_macro(image, config['data_params']['ratio_macro_to_micro'])
                    
                    images.append(
                        image
                    )

            utils.save_image(
                torch.cat(images, 0),
                r'sample_coco/%06d.png'%(i+1),
                nrow=gen_i,
                normalize=True,
                range=(-1, 1),
            )

        if (i + 1) % 10000 == 0:
            torch.save(
                g_running.state_dict(), r'checkpoint_coco/%06d.model'%(i+1)
            )

        state_msg = (
            r'Size: {}; G: {:.3f}; D: {:.3f}; Grad: {:.3f}; sp_G: {:.3f}; sp_D: {:.3f}; Alpha: {:.5f}'.format(4 * 2 ** step, gen_loss_val, disc_loss_val, grad_loss_val, spatial_loss_G_val, spatial_loss_D_val, alpha)
        )

        pbar.set_description(state_msg)


if __name__ == '__main__':
    code_size = 256#512
    batch_size = 16
    n_critic = 1

    #### Hyper-parameter
    parser = argparse.ArgumentParser(description='Progressive Growing of GANs')

    parser.add_argument('--path', type=str, help='path of specified dataset', required=True)
    parser.add_argument(
        '--phase',
        type=int,
        default=600000,
        help='number of samples used for each training phases',
    )
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--sched', action='store_true', help='use lr scheduling')
    parser.add_argument('--init_size', default=8, type=int, help='initial image size')
    parser.add_argument('--max_size', default=128, type=int, help='max image size')
    parser.add_argument(
        '--ckpt', default=None, type=str, help='load from previous checkpoints'
    )
    parser.add_argument(
        '--no_from_rgb_activate',
        action='store_true',
        help='use activate in from_rgb (original implementation)',
    )
    parser.add_argument(
        '--mixing', action='store_true', help='use mixing regularization'
    )
    parser.add_argument(
        '--loss',
        type=str,
        default='wgan-gp',
        choices=['wgan-gp', 'r1'],
        help='class of gan loss',
    )

    parser.add_argument('--config', type=str, required=True)

    args = parser.parse_args()

    #### configs
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

        # Basic protect. Otherwise, I don't know what will happen. OuO
        micro_size = config["data_params"]['micro_patch_size']
        macro_size = config["data_params"]['macro_patch_size']
        full_size = config["data_params"]['full_image_size']
        assert macro_size[0] % micro_size[0] == 0
        assert macro_size[1] % micro_size[1] == 0
        assert full_size[0] % micro_size[0] == 0
        assert full_size[1] % micro_size[1] == 0

    precompute_parameters(config)
    args.max_size = full_size[0]
    coord_loss_w = config["loss_params"]["coord_loss_w"]

    #### Networks
    n_layers_G = int(math.log(micro_size[0], 2)) - 1
    n_layers_D = int(math.log(macro_size[0], 2)) - 1
    step_G = n_layers_G - 1
    step_D = n_layers_D - 1
        
    generator = nn.DataParallel(StyledGenerator(code_size, n_layers_G)).cuda()
    discriminator = nn.DataParallel(
        ConditionalDiscriminatorH(layers=n_layers_D, from_rgb_activate=not args.no_from_rgb_activate)
    ).cuda()

    g_running = StyledGenerator(code_size, n_layers_G).cuda()
    g_running.train(False)
    
    spatial_predictor = nn.DataParallel(SpatialCoordinatePredictor(in_dim=512, aux_dim=config["model_params"]["aux_dim"], spatial_dim=config["model_params"]["spatial_dim"])).cuda()

    #### optimizers
    g_optimizer = optim.Adam(
        generator.module.generator.parameters(), lr=args.lr, betas=(0.0, 0.99)
    )
    g_optimizer.add_param_group(
        {
            'params': generator.module.style.parameters(),
            'lr': args.lr,# * 0.1,
            'mult': 0.01,
        }
    )
    g_optimizer.add_param_group(
        {
            'params': spatial_predictor.parameters(),
            'lr': args.lr,
            'mult': 0.01,
        }
    )
    d_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.0, 0.99))
    d_optimizer.add_param_group(
        {
            'params': spatial_predictor.parameters(),
            'lr': args.lr,
            'mult': 0.01,
        }
    )
    
    criterion_mse = nn.MSELoss()

    accumulate(g_running, generator.module, 0)


    #### continue training
    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt)

        generator.module.load_state_dict(ckpt['generator'])
        discriminator.module.load_state_dict(ckpt['discriminator'])
        g_running.load_state_dict(ckpt['g_running'])
        g_optimizer.load_state_dict(ckpt['g_optimizer'])
        d_optimizer.load_state_dict(ckpt['d_optimizer'])

    #### data transform
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    if args.sched:
        args.lr = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}
        args.batch = {4: 512, 8: 256, 16: 128, 32: 64, 64: 32, 128: 32, 256: 32}

    else:
        args.lr = {}
        args.batch = {}

    args.gen_sample = {512: (8, 4), 1024: (4, 2)}

    args.batch_default = 32

    #### patch handler and coordinate handler
    # coordinates
    coord_base = torch.Tensor([[-1.0, 1.0], [1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]]).float().cuda()
    coord_handler = CoordHandler(config)
    patch_handler = PatchHandler(config)
    num_micro_in_macro = config['data_params']['ratio_macro_to_micro'][0] * config['data_params']['ratio_macro_to_micro'][1]
    
    train(args, generator, discriminator)
    
    '''
    CUDA_VISIBLE_DEVICES=0 python train_coco.py --config=configs/CelebA_128x128_N2M2S64.yaml --path=datasets/celeba/ --loss r1 --mixing --sched
    '''
