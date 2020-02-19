import argparse
import random
import math

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
from model import StyledGenerator, Discriminator
    

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

            resolution = 4 * 2 ** step

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
                }, r'checkpoint/train_step-{}.model'.format(ckpt_step))

            adjust_lr(g_optimizer, args.lr.get(resolution, 0.001))
            adjust_lr(d_optimizer, args.lr.get(resolution, 0.001))

        #### update discriminator
        try:
            real_image = next(data_loader)

        except (OSError, StopIteration):
            data_loader = iter(loader)
            real_image = next(data_loader)

        used_sample += real_image.shape[0]

        b_size = real_image.size(0)
        coords = coord_base.repeat(b_size, 1)
        select = np.hstack([[i*b_size+j for i in range(4)] for j in range(b_size)])
        real_image = real_image.cuda()
        
        if args.loss == 'wgan-gp':
            real_predict = discriminator(real_image, step=step, alpha=alpha)
            real_predict = real_predict.mean() - 0.001 * (real_predict ** 2).mean()
            (-real_predict).backward()

        elif args.loss == 'r1':
            real_image.requires_grad = True
            real_scores = discriminator(real_image, step=step, alpha=alpha)
            real_predict = F.softplus(-real_scores).mean()
            real_predict.backward(retain_graph=True)

            grad_real = grad(
                outputs=real_scores.sum(), inputs=real_image, create_graph=True
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
            gen_in11 = torch.cat([gen_in11.repeat(4, 1)[select], coords], dim=1)
            
            gen_in12 = gen_in12.squeeze(0)
            gen_in12 = torch.cat([gen_in12.repeat(4, 1)[select], coords], dim=1)
            
            gen_in21 = gen_in21.squeeze(0)
            gen_in21 = torch.cat([gen_in21.repeat(4, 1)[select], coords], dim=1)
            
            gen_in22 = gen_in22.squeeze(0)
            gen_in22 = torch.cat([gen_in22.repeat(4, 1)[select], coords], dim=1)
            
            gen_in1 = [gen_in11, gen_in12]
            gen_in2 = [gen_in21, gen_in22]

        else:
            gen_in1, gen_in2 = torch.randn(2, b_size, code_size-2, device='cuda').chunk(
                2, 0                                  # 512
            )
            gen_in1 = gen_in1.squeeze(0)# (B, 254)
            gen_in2 = gen_in2.squeeze(0)# (B, 254)

            # repeat and copy
            gen_in1 = torch.cat([gen_in1.repeat(4, 1)[select], coords], dim=1)
            gen_in2 = torch.cat([gen_in2.repeat(4, 1)[select], coords], dim=1)


        fake_image = generator(gen_in1, step=step-1, alpha=alpha)
        
        fake_image_up = torch.cat([fake_image[0::4], fake_image[1::4]], dim=3)
        fake_image_dn = torch.cat([fake_image[2::4], fake_image[3::4]], dim=3)
        fake_image = torch.cat([fake_image_up, fake_image_dn], dim=2)
        
        fake_predict = discriminator(fake_image, step=step, alpha=alpha)

        if args.loss == 'wgan-gp':
            fake_predict = fake_predict.mean()
            fake_predict.backward()

            eps = torch.rand(b_size, 1, 1, 1).cuda()
            x_hat = eps * real_image.data + (1 - eps) * fake_image.data
            x_hat.requires_grad = True
            hat_predict = discriminator(x_hat, step=step, alpha=alpha)
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
            fake_predict.backward()
            if i%10 == 0:
                disc_loss_val = (real_predict + fake_predict).item()

        d_optimizer.step()


        #### update generator
        if (i + 1) % n_critic == 0:
            generator.zero_grad()

            requires_grad(generator, True)
            requires_grad(discriminator, False)

            fake_image = generator(gen_in2, step=step-1, alpha=alpha)
            
            fake_image_up = torch.cat([fake_image[0::4], fake_image[1::4]], dim=3)
            fake_image_dn = torch.cat([fake_image[2::4], fake_image[3::4]], dim=3)
            fake_image = torch.cat([fake_image_up, fake_image_dn], dim=2)

            predict = discriminator(fake_image, step=step, alpha=alpha)

            if args.loss == 'wgan-gp':
                loss = -predict.mean()

            elif args.loss == 'r1':
                loss = F.softplus(-predict).mean()

            if i%10 == 0:
                gen_loss_val = loss.item()

            loss.backward()
            g_optimizer.step()
            accumulate(g_running, generator.module)

            requires_grad(generator, False)
            requires_grad(discriminator, True)


        #### validation
        if (i + 1) % 100 == 0:
            images = []

            gen_i, gen_j = args.gen_sample.get(resolution, (10, 5))
            coords = coord_base.repeat(gen_j, 1)
            select = np.hstack([[i*gen_j+j for i in range(4)] for j in range(gen_j)])

            with torch.no_grad():
                for ii in range(gen_i):
                    style = torch.randn(gen_j, code_size-2).cuda().repeat(4, 1)[select]
                    style = torch.cat([style, coords], dim=1)
                    image = g_running(style, step=step-1, alpha=alpha).data.cpu()
                    
                    image_up = torch.cat([image[0::4], image[1::4]], dim=3)
                    image_dn = torch.cat([image[2::4], image[3::4]], dim=3)
                    image = torch.cat([image_up, image_dn], dim=2)
                    
                    images.append(
                        image
                    )

            utils.save_image(
                torch.cat(images, 0),
                r'sample/%06d.png'%(i+1),
                nrow=gen_i,
                normalize=True,
                range=(-1, 1),
            )

        if (i + 1) % 10000 == 0:
            torch.save(
                g_running.state_dict(), r'checkpoint/%06d.model'%(i+1)
            )

        state_msg = (
            r'Size: {}; G: {:.3f}; D: {:.3f}; Grad: {:.3f}; Alpha: {:.5f}'.format(4 * 2 ** step, gen_loss_val, disc_loss_val, grad_loss_val, alpha)
        )

        pbar.set_description(state_msg)


if __name__ == '__main__':
    code_size = 512
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
    parser.add_argument('--max_size', default=64, type=int, help='max image size')
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

    #parser.add_argument('--config', type=str, required=True)

    args = parser.parse_args()

    #### Networks
    n_layers_D = int(math.log(args.max_size, 2)) - 1
    n_layers_G = int(math.log(args.max_size, 2)) - 2
        
    generator = nn.DataParallel(StyledGenerator(code_size, n_layers_G)).cuda()
    discriminator = nn.DataParallel(
        Discriminator(layers=n_layers_D, from_rgb_activate=not args.no_from_rgb_activate)
    ).cuda()

    g_running = StyledGenerator(code_size, n_layers_G).cuda()
    g_running.train(False)

    #### optimizers
    g_optimizer = optim.Adam(
        generator.module.generator.parameters(), lr=args.lr, betas=(0.0, 0.99)
    )
    g_optimizer.add_param_group(
        {
            'params': generator.module.style.parameters(),
            'lr': args.lr * 0.01,
            'mult': 0.01,
        }
    )
    d_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.0, 0.99))

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
    
    
    train(args, generator, discriminator)
