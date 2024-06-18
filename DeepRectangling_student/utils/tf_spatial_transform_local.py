# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import torch
import utils.tensorDLT_local as tensorDLT_local
from torch.nn import Upsample

import utils.constant as constant
grid_w = constant.GRID_W
grid_h = constant.GRID_H

def transformer(U, mask, theta, name='SpatialTransformer', **kwargs):

    def _repeat(x, n_repeats):
        rep = torch.transpose(torch.ones(n_repeats).unsqueeze(1), 0,1)
        # print("x:", x.shape)
        # print("rep:",rep.shape)
        rep = rep.to(dtype= torch.float32)
        x = x.to(dtype=torch.float32)
        x = torch.matmul(x.view(-1, 1), rep)
        return x.view(-1)


    def _interpolate(im, x, y, out_size):
        # constants
        num_batch,height,width,channels = im.shape

        x = x.to(dtype= torch.float32)
        y = y.to(dtype= torch.float32)
        out_height = out_size[0]
        out_width = out_size[1]
        zero = torch.zeros([],dtype=torch.int32)
        max_y = int(im.shape[1] - 1)
        max_x = int(im.shape[2] - 1)

        # do sampling
        x0 = torch.floor(x).to(dtype= torch.int32)
        x1 = x0 + 1
        y0 = torch.floor(y).to(dtype= torch.int32)
        y1 = y0 + 1

        x0 = torch.clip(x0, zero, max_x)
        x1 = torch.clip(x1, zero, max_x)
        y0 = torch.clip(y0, zero, max_y)
        y1 = torch.clip(y1, zero, max_y)
        dim2 = width
        dim1 = width * height
        # print("dim2:", dim2)
        # print("dim1:", dim1)
        base = _repeat(torch.range(0,num_batch-1) * dim1, out_height * out_width).to(im.device)
        # print("base:", base.shape)
        # print("y0:", y0.shape)
        base_y0 = base + y0 * dim2
        base_y1 = base + y1 * dim2
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        # use indices to lookup pixels in the flat image and restore
        # channels dim
        im_flat = im.contiguous().view(-1, channels)
        im_flat = im_flat.to(dtype= torch.float32)
        # print("im_flat:",im_flat.shape)
        # print("idx_a:",idx_a.shape)
        device = im_flat.device
        Ia = torch.gather(im_flat, 0, idx_a.unsqueeze(1).repeat(1,channels).to(device,dtype=torch.int64))
        Ib = torch.gather(im_flat, 0, idx_b.unsqueeze(1).repeat(1,channels).to(device,dtype=torch.int64))
        Ic = torch.gather(im_flat, 0, idx_c.unsqueeze(1).repeat(1,channels).to(device,dtype=torch.int64))
        Id = torch.gather(im_flat, 0, idx_d.unsqueeze(1).repeat(1,channels).to(device,dtype=torch.int64))
        # print("IA:",Ia.shape)
        # and finally calculate interpolated values
        x0_f = x0.to(dtype= torch.float32)
        x1_f = x1.to(dtype= torch.float32)
        y0_f = y0.to(dtype= torch.float32)
        y1_f = y1.to(dtype= torch.float32)
        wa = ((x1_f - x) * (y1_f - y)).unsqueeze(1).to(device)
        wb = ((x1_f - x) * (y - y0_f)).unsqueeze(1).to(device)
        wc = ((x - x0_f) * (y1_f - y)).unsqueeze(1).to(device)
        wd = ((x - x0_f) * (y - y0_f)).unsqueeze(1).to(device)
        output = wa * Ia + wb * Ib + wc * Ic + wd * Id
        return output

    #input:  batch_size*(grid_h+1)*(grid_w+1)*2
    #output: batch_size*grid_h*grid_w*9
    def get_Hs(theta, width, height):
        # print("theta:",theta.shape)
        num_batch = theta.shape[0]
        h = height / grid_h
        w = width / grid_w
        Hs = []
        for i in range(grid_h):
            for j in range(grid_w):
                hh = i * h
                ww = j * w
                ori = torch.tile(
                    torch.FloatTensor([ww, hh, ww + w, hh, ww, hh + h, ww + w, hh + h]),
                    dims=(num_batch, 1)).to(theta.device)
                # id = i * (grid_w + 1) + grid_w
                tar = torch.cat(
                    [
                        theta[0:,i:i+1,j:j+1,0:],theta[0:,i:i+1,(j+1):(j+1)+1,0:],
                        theta[0:, (i+1):(i+1) + 1, j:j + 1, 0:],theta[0:,(i+1):(i+1)+1,(j+1):(j+1)+1,0:],
                    ], dim=1).to(theta.device)
                # print("ori:",ori.shape)
                # print("tar:", tar.shape)
                tar = tar.view(num_batch, 8)
                # tar = tf.Print(tar, [tf.slice(ori, [0, 0], [1, -1])],message="[ori--i:"+str(i)+",j:"+str(j)+"]:", summarize=100,first_n=5)
                # tar = tf.Print(tar, [tf.slice(tar, [0, 0], [1, -1])],message="[tar--i:"+str(i)+",j:"+str(j)+"]:", summarize=100,first_n=5)
                Hs.append(tensorDLT_local.solve_DLT(ori, tar).view(num_batch, 1, 9))
        Hs = torch.cat(Hs, dim=1).view(num_batch, grid_h, grid_w, 9)
        return Hs

        
    def _meshgrid(height, width):
        #x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
        #                        tf.transpose(tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
        #y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
        #                        tf.ones(shape=tf.stack([1, width])))
        x_t = torch.matmul(torch.ones(height, 1),
                                torch.transpose(torch.linspace(0., float(width)-1.001, width).unsqueeze(1), 0,1))
        y_t = torch.matmul(torch.linspace(0., float(height)-1.001, height).unsqueeze(1),
                                torch.ones(1, width))

        x_t_flat = x_t.view(1, -1)
        y_t_flat = y_t.view(1, -1)

        ones = torch.ones_like(x_t_flat)
        grid = torch.cat([x_t_flat, y_t_flat, ones], dim=0)
            
        return grid

    def _transform3(theta, input_dim, mask):
        device = theta.device
        num_batch, height, width, num_channels  = input_dim.shape

        # the width/height should be an an integral multiple of grid_w/grid_h
        width_float = width
        height_float = height

        theta = theta.to(dtype= torch.float32)
        Hs = get_Hs(theta, width_float, height_float)
        ##########################################
        # print("Hs")
        # print(Hs.shape)
        Hs = Hs.permute(0,3,1,2)
        H_array = Upsample(size=(height, width), mode='nearest')(Hs)
        H_array = H_array.permute(0,2,3,1)
        # print("H_array:",H_array.shape)
        H_array = H_array.view(-1, 3, 3)
        ##########################################

        out_height = height
        out_width = width
        grid = _meshgrid(out_height, out_width)
        # print("out_height:", out_height)
        # print("out_width:", out_width)
        # print("_meshgrid:", grid.shape)
        grid = grid.unsqueeze(0).to(device)
        grid = grid.view(-1)
        grid = torch.tile(grid, [num_batch])  # stack num_batch grids
        grid = grid.view(num_batch, 3, -1)
        # print("grid")
        # print(grid.shape)
        ### [bs, 3, N]

        grid = torch.transpose(grid, 1,2).unsqueeze(3)
        # print("grid:",grid.shape)
        ### [bs, 3, N] -> [bs, N, 3] -> [bs, N, 3, 1]
        grid = grid.contiguous().view(-1, 3, 1)
        ### [bs*N, 3, 1]

        grid_row = grid.view(-1, 3)
        # print("grid_row")
        # print(grid_row.shape)
        # print(H_array[:, 0, :].shape)
        x_s = torch.sum(torch.multiply(H_array[:, 0, :], grid_row), 1)
        y_s = torch.sum(torch.multiply(H_array[:, 1, :], grid_row), 1)
        t_s = torch.sum(torch.multiply(H_array[:, 2, :], grid_row), 1)

        # The problem may be here as a general homo does not preserve the parallelism
        # while an affine transformation preserves it.
        # while an affine transformation preserves it.
        t_s_flat =t_s.view(-1)
        # print("x_s:", x_s.shape)
        # print("y_s:", y_s.shape)
        # print("t_s_flat:", t_s_flat.shape)
        t_1 = torch.ones_like(t_s_flat)
        t_0 = torch.zeros_like(t_s_flat)
        sign_t = torch.where(t_s_flat >= 0, t_1, t_0) * 2 - 1
        t_s_flat = t_s_flat + sign_t * 1e-8

        x_s_flat = x_s.view(-1) / t_s_flat
        y_s_flat = y_s.view(-1) / t_s_flat

        out_size = (height, width)
        input_transformed = _interpolate(input_dim, x_s_flat, y_s_flat, out_size)
        mask_transformed = _interpolate(mask, x_s_flat, y_s_flat, out_size)
        # print("input_transformed:",input_transformed.shape)
        warp_image = input_transformed.view(num_batch, height, width, num_channels)
        warp_mask = mask_transformed.view(num_batch, height, width, num_channels)

        return warp_image, warp_mask


    # output = _transform(theta, U, out_size)
    U, mask =  U.permute(0,2,3,1),mask.permute(0,2,3,1)
    U = U - 1.
    warp_image, warp_mask = _transform3(theta, U, mask)
    warp_image = warp_image + 1.
    warp_image = torch.clip_(warp_image, -1, 1)
    return warp_image, warp_mask



