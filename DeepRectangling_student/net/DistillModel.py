import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import utils.tf_spatial_transform_local as tf_spatial_transform_local
import utils.torch_tps_transform as torch_tps_transform
import utils.constant as constant

grid_w = constant.GRID_W
grid_h = constant.GRID_H
gpu_device = constant.GPU_DEVICE

def shift2mesh0(mesh_shift, height,width):
    device = mesh_shift.device
    batch_size = mesh_shift.shape[0]
    h = height / grid_h
    w = width / grid_w
    ori_pt = []
    for i in range(grid_h + 1):
        for j in range(grid_w + 1):
            ww = j * w
            hh = i * h
            p = torch.FloatTensor([ww, hh])
            ori_pt.append(p.unsqueeze(0))
    ori_pt = torch.cat(ori_pt,dim=0)
    # print(ori_pt.shape)
    ori_pt = ori_pt.view(grid_h + 1, grid_w + 1, 2)
    # print(ori_pt)
    ori_pt = torch.tile(ori_pt.unsqueeze(0), [batch_size, 1, 1, 1])
    ori_pt = ori_pt.to(gpu_device)
    # print("ori_pt:",ori_pt.shape)
    # print("mesh_shift:", mesh_shift.shape)
    tar_pt = ori_pt + mesh_shift
    return tar_pt

def get_rigid_mesh(batch_size, height, width):

    ww = torch.matmul(torch.ones([grid_h+1, 1]), torch.unsqueeze(torch.linspace(0., float(width), grid_w+1), 0))
    hh = torch.matmul(torch.unsqueeze(torch.linspace(0.0, float(height), grid_h+1), 1), torch.ones([1, grid_w+1]))
    ww = ww.to(gpu_device)
    hh = hh.to(gpu_device)

    ori_pt = torch.cat((ww.unsqueeze(2), hh.unsqueeze(2)),2) # (grid_h+1)*(grid_w+1)*2
    ori_pt = ori_pt.unsqueeze(0).expand(batch_size, -1, -1, -1)

    return ori_pt

def get_norm_mesh(mesh, height, width):
    batch_size = mesh.size()[0]
    mesh_w = mesh[...,0]*2./float(width) - 1.
    mesh_h = mesh[...,1]*2./float(height) - 1.
    norm_mesh = torch.stack([mesh_w, mesh_h], 3) # bs*(grid_h+1)*(grid_w+1)*2
    # norm_mesh = torch.stack([mesh_h, mesh_w], 3)  # bs*(grid_h+1)*(grid_w+1)*2
    # print("norm_mesh:",norm_mesh.shape)
    return norm_mesh.reshape([batch_size, -1, 2]) # bs*-1*2

def H2Mesh(H, rigid_mesh):

    H_inv = torch.inverse(H)
    ori_pt = rigid_mesh.reshape(rigid_mesh.size()[0], -1, 2)
    ones = torch.ones(rigid_mesh.size()[0], (grid_h+1)*(grid_w+1),1)
    ori_pt = ori_pt.to(gpu_device)
    ones = ones.to(gpu_device)

    ori_pt = torch.cat((ori_pt, ones), 2) # bs*(grid_h+1)*(grid_w+1)*3
    tar_pt = torch.matmul(H_inv, ori_pt.permute(0,2,1)) # bs*3*(grid_h+1)*(grid_w+1)

    mesh_x = torch.unsqueeze(tar_pt[:,0,:]/tar_pt[:,2,:], 2)
    mesh_y = torch.unsqueeze(tar_pt[:,1,:]/tar_pt[:,2,:], 2)
    mesh = torch.cat((mesh_x, mesh_y), 2).reshape([rigid_mesh.size()[0], grid_h+1, grid_w+1, 2])

    return mesh


def autopad2(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad2(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class RepConv(nn.Module):
    # Represented convolution
    # https://arxiv.org/abs/2101.03697
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=nn.SiLU(), deploy=False):
        super(RepConv, self).__init__()
        self.deploy = deploy
        self.groups = g
        self.in_channels = c1
        self.out_channels = c2

        assert k == 3
        assert autopad(k, p) == 1

        padding_11 = autopad(k, p) - k // 2
        self.act = nn.LeakyReLU(0.1, inplace=True) if act is True else (
            act if isinstance(act, nn.Module) else nn.Identity())

        if deploy:
            self.rbr_reparam = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=True)
        else:
            self.rbr_identity = (
                nn.BatchNorm2d(num_features=c1, eps=0.001, momentum=0.03) if c2 == c1 and s == 1 else None)
            self.rbr_dense = nn.Sequential(
                nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False),
                nn.BatchNorm2d(num_features=c2, eps=0.001, momentum=0.03),
            )
            self.rbr_1x1 = nn.Sequential(
                nn.Conv2d(c1, c2, 1, s, padding_11, groups=g, bias=False),
                nn.BatchNorm2d(num_features=c2, eps=0.001, momentum=0.03),
            )

    def forward(self, inputs):
        if hasattr(self, "rbr_reparam"):
            return self.act(self.rbr_reparam(inputs))
        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
        return self.act(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return (
            kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid,
            bias3x3 + bias1x1 + biasid,
        )

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch[0].weight
            running_mean = branch[1].running_mean
            running_var = branch[1].running_var
            gamma = branch[1].weight
            beta = branch[1].bias
            eps = branch[1].eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros(
                    (self.in_channels, input_dim, 3, 3), dtype=np.float32
                )
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def repvgg_convert(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        return (
            kernel.detach().cpu().numpy(),
            bias.detach().cpu().numpy(),
        )

    def fuse_conv_bn(self, conv, bn):
        std = (bn.running_var + bn.eps).sqrt()
        bias = bn.bias - bn.running_mean * bn.weight / std

        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        weights = conv.weight * t

        bn = nn.Identity()
        conv = nn.Conv2d(in_channels=conv.in_channels,
                         out_channels=conv.out_channels,
                         kernel_size=conv.kernel_size,
                         stride=conv.stride,
                         padding=conv.padding,
                         dilation=conv.dilation,
                         groups=conv.groups,
                         bias=True,
                         padding_mode=conv.padding_mode)

        conv.weight = torch.nn.Parameter(weights)
        conv.bias = torch.nn.Parameter(bias)
        return conv

    def fuse_repvgg_block(self):
        if self.deploy:
            return
        print(f"RepConv.fuse_repvgg_block")
        self.rbr_dense = self.fuse_conv_bn(self.rbr_dense[0], self.rbr_dense[1])

        self.rbr_1x1 = self.fuse_conv_bn(self.rbr_1x1[0], self.rbr_1x1[1])
        rbr_1x1_bias = self.rbr_1x1.bias
        weight_1x1_expanded = torch.nn.functional.pad(self.rbr_1x1.weight, [1, 1, 1, 1])

        # Fuse self.rbr_identity
        if (isinstance(self.rbr_identity, nn.BatchNorm2d) or isinstance(self.rbr_identity,
                                                                        nn.modules.batchnorm.SyncBatchNorm)):
            identity_conv_1x1 = nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=self.groups,
                bias=False)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.to(self.rbr_1x1.weight.data.device)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.squeeze().squeeze()
            identity_conv_1x1.weight.data.fill_(0.0)
            identity_conv_1x1.weight.data.fill_diagonal_(1.0)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.unsqueeze(2).unsqueeze(3)

            identity_conv_1x1 = self.fuse_conv_bn(identity_conv_1x1, self.rbr_identity)
            bias_identity_expanded = identity_conv_1x1.bias
            weight_identity_expanded = torch.nn.functional.pad(identity_conv_1x1.weight, [1, 1, 1, 1])
        else:
            bias_identity_expanded = torch.nn.Parameter(torch.zeros_like(rbr_1x1_bias))
            weight_identity_expanded = torch.nn.Parameter(torch.zeros_like(weight_1x1_expanded))

        self.rbr_dense.weight = torch.nn.Parameter(
            self.rbr_dense.weight + weight_1x1_expanded + weight_identity_expanded)
        self.rbr_dense.bias = torch.nn.Parameter(self.rbr_dense.bias + rbr_1x1_bias + bias_identity_expanded)

        self.rbr_reparam = self.rbr_dense
        self.deploy = True

        if self.rbr_identity is not None:
            del self.rbr_identity
            self.rbr_identity = None

        if self.rbr_1x1 is not None:
            del self.rbr_1x1
            self.rbr_1x1 = None

        if self.rbr_dense is not None:
            del self.rbr_dense
            self.rbr_dense = None


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C2f(nn.Module):
    # CSP Bottleneck with 2 convolutions
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class MeshRegressionNetwork(nn.Module):
    def __init__(self,inchannels=3):
        super(MeshRegressionNetwork, self).__init__()
        self.patch_height = (grid_h + 1)
        self.patch_width = (grid_w + 1)
        # Conv
        self.featureExtractor = nn.Sequential(
            Conv(inchannels, 64,3,2), # 512 -> 256
            Conv(64, 64,3,2), # 256 -> 128
            C2f(64,64),
            Conv(64, 64,3,2), # 128 -> 64
            C2f(64,64),
            Conv(64, 64, 3, 2), # 64 -> 32
            C2f(64,64),
            Conv(64, 64, 3, 2), # 32 -> 16
            C2f(64,64),
            Conv(64, 64, 3, 2), # 16 -> 8
            C2f(64,64),
            Conv(64, 64, 3, 2) # 8 -> 4
        )

        self.head512 = nn.Sequential(
            nn.Conv2d(64,1024,(3,4),2),
            nn.Flatten(),
            # nn.Linear(768, 1024),
            nn.SiLU(inplace=True),
            #nn.GELU(),
            nn.Linear(1024, self.patch_height * self.patch_width * 2)
        )

    def fuse(self):
        print('Fusing layers... ')
        for m in self.modules():
            if isinstance(m, RepConv):
                m.fuse_repvgg_block()
        return self

    def forward(self,x1):
        x1 = self.featureExtractor(x1)
        x512 = self.head512(x1)
        return x512.view(-1,self.patch_height, self.patch_width, 2)



class RectanglingNetwork(nn.Module):
    def __init__(self):
        super(RectanglingNetwork, self).__init__()
        self.ShareFeature = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16, eps=0.001, momentum=0.03),
            nn.SiLU(inplace=True),
            #  nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, eps=0.001, momentum=0.03),
            nn.SiLU(inplace=True),
            # nn.GELU(),

            nn.Conv2d(32, 3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(3, eps=0.001, momentum=0.03),
            nn.SiLU(inplace=True),
            #nn.GELU(),
        )
        self.meshRegression = MeshRegressionNetwork(3)

    # using multi-mesh strategy to warp the input image
    # def forward(self,input_img,mask_img):
    #     _,_,height,width = input_img.shape
    #     # method 1
    #     f_input_img = self.ShareFeature(input_img)
    #     feature = torch.mul(f_input_img, mask_img)
    #     # method 2
    #     #feature = torch.cat([input_img, mask_img],dim=1)
    #     mesh_shift_finnal = self.meshRegression(feature)
    #
    #     mesh_final = shift2mesh0(mesh_shift_finnal,height,width)
    #     warp_image_final, warp_mask_final = tf_spatial_transform_local.transformer(input_img, mask_img, mesh_final)
    #     warp_image_final, warp_mask_final = warp_image_final.permute(0, 3, 1, 2), warp_mask_final.permute(0, 3, 1, 2)
    #     # method 1, Options in inference
    #     warp_image_final = torch.mul(warp_image_final, warp_mask_final)
    #
    #     return mesh_final, warp_image_final, warp_mask_final

    # using tps strategy to warp the input image
    def forward(self, input_img, mask_img):
        batch_size, _, height, width = input_img.shape
        # method 1
        f_input_img = self.ShareFeature(input_img)
        feature = torch.mul(f_input_img, mask_img)
        # method 2
        # feature = torch.cat([input_img, mask_img],dim=1)
        mesh_motion = self.meshRegression(feature)

        rigid_mesh = get_rigid_mesh(batch_size, height, width)
        H_one = torch.eye(3)
        H = torch.tile(H_one.unsqueeze(0), [batch_size, 1, 1]).to(gpu_device)

        ini_mesh = H2Mesh(H, rigid_mesh)
        mesh_final = ini_mesh + mesh_motion
        # print("rigid_mesh:",rigid_mesh.shape)
        # print("mesh_final:",mesh_final.shape)

        norm_rigid_mesh = get_norm_mesh(rigid_mesh, height, width)
        norm_mesh = get_norm_mesh(mesh_final, height, width)

        output_tps = torch_tps_transform.transformer(torch.cat((input_img, mask_img), 1), norm_rigid_mesh,  norm_mesh, (height, width))
        warp_image_final = output_tps[:, 0:3, ...]
        warp_mask_final = output_tps[:, 3:6, ...]
        # method 1, Options in inference
        warp_image_final = torch.mul(warp_image_final, warp_mask_final)

        return mesh_final, warp_image_final, warp_mask_final

        # warp_image_final, warp_mask_final = tf_spatial_transform_local.transformer(input_img, mask_img, mesh_final)
        # warp_image_final, warp_mask_final = warp_image_final.permute(0, 3, 1, 2), warp_mask_final.permute(0, 3, 1, 2)
        # # method 1, Options in inference
        # warp_image_final = torch.mul(warp_image_final, warp_mask_final)
        #
        # return mesh_final, warp_image_final, warp_mask_final


def tensor_DLT(src_p, dst_p):
    bs, _, _ = src_p.shape

    ones = torch.ones(bs, 4, 1)
    if torch.cuda.is_available():
        ones = ones.to(gpu_device)
    xy1 = torch.cat((src_p, ones), 2)
    zeros = torch.zeros_like(xy1)
    if torch.cuda.is_available():
        zeros = zeros.to(gpu_device)

    xyu, xyd = torch.cat((xy1, zeros), 2), torch.cat((zeros, xy1), 2)
    M1 = torch.cat((xyu, xyd), 2).reshape(bs, -1, 6)
    M2 = torch.matmul(
        dst_p.reshape(-1, 2, 1),
        src_p.reshape(-1, 1, 2),
    ).reshape(bs, -1, 2)

    # Ah = b
    A = torch.cat((M1, -M2), 2)
    b = dst_p.reshape(bs, -1, 1)

    # h = A^{-1}b
    Ainv = torch.inverse(A)
    h8 = torch.matmul(Ainv, b).reshape(bs, 8)

    H = torch.cat((h8, ones[:, 0, :]), 1).reshape(bs, 3, 3)
    return H
