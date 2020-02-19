import numpy as np
import torch

from math import floor, sqrt, pi
from numpy import sin, cos
#import tensorflow as tf


class PatchHandler():

    def __init__(self, config):
        self.config = config

        self.batch_size = self.config["train_params"]["batch_size"] # 4
        self.micro_patch_size = self.config["data_params"]["micro_patch_size"] # 16
        self.macro_patch_size = self.config["data_params"]["macro_patch_size"] # 64
        self.full_image_size = self.config["data_params"]["full_image_size"]   # 128
        self.coordinate_system = self.config["data_params"]["coordinate_system"]# "euclidean"
        self.c_dim = self.config["data_params"]["c_dim"] # 3

        self.num_micro_compose_macro = config["data_params"]["num_micro_compose_macro"] # 16


    def reord_patches(self, x, batch_size, patch_count):
        """
        # Reorganize image order from [a0, b0, c0, a1, b1, c1, ...] to [a0, a1, ..., b0, b1, ..., c0, c1, ...]
        a b c ...    #batch_size
        0 1 2 ...    #patch_count
        ...
        """
        select = np.hstack([[i*batch_size+j for i in range(patch_count)] for j in range(batch_size)])
        return x[select]


    def concat_micro_patches_cpu(self, generated_patches, ratio_over_micro):

        patch_count = ratio_over_micro[0] * ratio_over_micro[1] # 4 * 4 = 16
        generated_patches = np.concatenate(generated_patches, axis=0)

        stage1_shape = [
            -1, 
            patch_count*self.micro_patch_size[0],
            self.micro_patch_size[1], 
            self.c_dim
        ]

        merge_stage1 = generated_patches.reshape(*stage1_shape)
        merge_stage1_slice = []
        for i in range(ratio_over_micro[1]):
            x_st  = self.micro_patch_size[0] * ratio_over_micro[0] * i
            x_ed = x_st + self.micro_patch_size[0] * ratio_over_micro[0]
            y_st  = 0
            y_ed  = self.micro_patch_size[1]
            merge_stage1_slice.append(merge_stage1[:, x_st:x_ed, y_st:y_ed, :])
        merge_stage1_slice = np.concatenate(merge_stage1_slice, axis=2)

        final_shape = [
            -1, 
            ratio_over_micro[0]*self.micro_patch_size[0], 
            ratio_over_micro[1]*self.micro_patch_size[1], 
            self.c_dim
        ]
        merge_stage2 = merge_stage1_slice.reshape(*final_shape)

        return merge_stage2[0]


    def concat_micro_patches_gpu(self, x, ratio_over_micro):
        """
        @x <tsr> <N, C, H, W>
        """
        assert ratio_over_micro[0]==ratio_over_micro[1], "Didn't test x!=y case"
        # ratio_over_micro = int(sqrt(self.full_patch_count))
        num_patches = ratio_over_micro[0] * ratio_over_micro[1]

        # Step 1: micro patches -> stripes of images
        # merge_stage1 = tf.reshape(x, [-1, num_patches*self.micro_patch_size[0], self.micro_patch_size[1], 3])
        #                            [bs, num_patches*h,                      , w                       , 3]
        # merge_stage1 = torch.reshape(x, [-1, self.c_dim, num_patches*self.micro_patch_size[0], self.micro_patch_size[1]])
        merge_stage1 = torch.reshape(x, [-1, num_patches*self.micro_patch_size[0], self.micro_patch_size[1], 3])

        slices = []
        for i in range(ratio_over_micro[1]): # 0, 1, 2, 3
            slice_st = [0, self.micro_patch_size[0]*ratio_over_micro[0]*i, 0, 0]
            slice_ed = [-1, self.micro_patch_size[1]*ratio_over_micro[0], self.micro_patch_size[1], -1]
            # slices.append(tf.slice(merge_stage1, slice_st, slice_ed))
            slice = merge_stage1[:, 
                                 self.micro_patch_size[0]*ratio_over_micro[0]*i:self.micro_patch_size[0]*ratio_over_micro[0]*i+self.micro_patch_size[1]*ratio_over_micro[0],
                                 0:self.micro_patch_size[1],
                                 :]
            # print(slice.size())
            slices.append(slice)
        merge_stage1_slice = torch.cat(slices, dim=2)

        # Step 2: stripes of images -> target image (macro patch or full image)
        final_shape = [
            -1, 
            ratio_over_micro[0]*self.micro_patch_size[0], 
            ratio_over_micro[1]*self.micro_patch_size[1], 
            self.c_dim,
        ]
        merge_stage2 = torch.reshape(merge_stage1_slice, final_shape)

        return merge_stage2


    def reord_micro_patches(self, x, ratio_over_micro):
        """
        @x <tsr> (ratio_over_micro[0]*ratio_over_micro[1], 3, h, w)

        The original order of x:
            [(1, 1) (2, 1) (3, 1) (4, 1), ..., (n, 1)
             (1, 2) (2, 2) (3, 2) (4, 2), ..., (n, 2)
             (1, 3) (2, 3) (3, 3) (4, 3), ..., (n, 3)
             (1, 4) (2, 4) (3, 4) (4, 4), ..., (n, 4)
             ...    ...    ...    ...     ...  ...
             (1, n) (2, n) (3, n) (4, n), ..., (n, n)]

        We need to reorder it such that:
            [(1, 1) (1, 2) (1, 3) (1, 4), ..., (1, n)
             (2, 1) (2, 2) (2, 3) (2, 4), ..., (2, n)
             (3, 1) (3, 2) (3, 3) (3, 4), ..., (3, n)
             (4, 1) (4, 2) (4, 3) (4, 4), ..., (4, n)
             ...    ...    ...    ...     ...  ...
             (n, 1) (n, 2) (n, 3) (n, 4), ..., (n, n)]
        """
        select = np.hstack([[i*ratio_over_micro[0]+j for i in range(ratio_over_micro[0]*ratio_over_micro[1])] for j in range(ratio_over_micro[0])])
        return x[select]


    def concat_micro_patches_to_macro_single_patch(self, micro_patches, ratio_over_micro, macro_coord=None, display_full_img=False):
        # concat the micro patchs
        '''
        @micro_patches <tsr> (num_patches, 3, h, w)
        '''
        reord_micro_patches = reord_micro_patches(micro_patches, ratio_over_micro)

        num_patches = ratio_over_micro[0] * ratio_over_micro[1]
        _, _, h, w = micro_patches.size()
        #reshapeed_patches = torch.reshape(micro_patches, [-1, num_patches*])

        if display_full_img:
            assert macro_coord is not None, "Coordinate of the macro patch is required but got null here."
        else:
            pass


    def concat_micro_patches_to_macro_patches(self):
        pass


    def concat_micro_patches_to_single_full_image(self):
        pass


    def crop_micro_from_full_gpu(self, imgs, crop_pos_x, crop_pos_y):

        ps_x, ps_y = self.micro_patch_size # i.e. Patch-Size, e.g. [16, 16]

        valid_area_x = self.full_image_size[0] - self.micro_patch_size[0] # 128 - 16 = 112
        if self.coordinate_system == "cylindrical":
            valid_area_y = self.full_image_size[1] # Horizontal don't need padding
        elif self.coordinate_system == "euclidean":
            valid_area_y = self.full_image_size[1] - self.micro_patch_size[1] # 128 - 16 = 112

        crop_result = []
        batch_size = imgs.shape[0] # [B, 3, H, W], H/W is the full image size
        for i in range(batch_size*self.num_micro_compose_macro): # B * 16
            i_idx = i // self.num_micro_compose_macro
            x_idx = int(round((crop_pos_x[i, 0]+1)/2*valid_area_x))
            y_idx = int(round((crop_pos_y[i, 0]+1)/2*valid_area_y))
            # print(x_idx, y_idx)

            # Only cylindrical coordinate system provide overflow protection
            # The code is complicated because:
            #     1. Need to use where to handle "360-degree-edge-crossing" edge case.
            #     2. `tf.where` requires input shape to be the same.
            #
            # P.S. I hate myself selecting TF in the very beginning...
            if self.coordinate_system == "cylindrical":
                pass
                """
                # Wrap the end if out-of-bound
                y_idx_st, y_idx_ed = y_idx, y_idx+ps_y
                y_idx_st = tf.where(tf.greater(y_idx_st, self.full_image_size[1]), 
                                    y_idx_st-self.full_image_size[1], 
                                    y_idx_st)
                y_idx_ed = tf.where(tf.greater(y_idx_ed, self.full_image_size[1]), 
                                    y_idx_ed-self.full_image_size[1], 
                                    y_idx_ed)
                y_idx_st = torch.where(y_idx_ed > self.full_image_size[1], y_idx_ed-self.full_image_size[1],)

                # Protect zero selection later, select some trash values instead if the assertion is triggered
                direct_y_idx_st = tf.where(tf.greater(y_idx_st, y_idx_ed), 
                                           y_idx_ed, 
                                           y_idx_st)
                direct_y_idx_ed = tf.where(tf.greater(y_idx_st, y_idx_ed), 
                                           y_idx_st, 
                                           y_idx_ed)

                # `direct_crop` is the default case
                # `wrap_crop` is when the cropped patch will cross the 360 degree line.
                direct_crop = imgs[i_idx, x_idx:x_idx+ps_x, direct_y_idx_st:direct_y_idx_ed, :]
                wrap_crop = tf.concat([
                    imgs[i_idx, x_idx:x_idx+ps_x, y_idx_st:, :],
                    imgs[i_idx, x_idx:x_idx+ps_x, :y_idx_ed, :],
                ], axis=1)

                # Protect selection
                # Remove redundant trash values, force `direct_crop` and `wrap_crop` become the same shape
                direct_crop = direct_crop[:, :ps_y, :]
                wrap_crop   = wrap_crop[:, :ps_y, :]

                selected_crop = tf.where(tf.greater(y_idx_st, y_idx_ed), wrap_crop, direct_crop)
                crop_result.append(selected_crop)
                """

            # Euclidean is so easy...
            elif self.coordinate_system == "euclidean":
                y_idx_st = y_idx
                crop_result.append(imgs[i_idx, :, x_idx:x_idx+ps_x, y_idx:y_idx+ps_y])
                # [(C, h, w), (C, h, w), ..., (C, h, w)]

        return torch.stack(crop_result, dim=0)

