import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
import argparse
import cv2
from net.DistillModel import RectanglingNetwork
import torchvision.transforms as transforms

import utils.constant as constant
os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

grid_w = constant.GRID_W
grid_h = constant.GRID_H
    
def draw_mesh_on_warp(warp, f_local):
    
    #f_local[3,0,0] = f_local[3,0,0] - 2
    #f_local[4,0,0] = f_local[4,0,0] - 4
    #f_local[5,0,0] = f_local[5,0,0] - 6
    #f_local[6,0,0] = f_local[6,0,0] - 8
    #f_local[6,0,1] = f_local[6,0,1] + 7
    # print("f_local:",f_local.shape)
    min_w = np.minimum(np.min(f_local[:,:,0]), 0).astype(np.int32)
    max_w = np.maximum(np.max(f_local[:,:,0]), 512).astype(np.int32)
    min_h = np.minimum(np.min(f_local[:,:,1]), 0).astype(np.int32)
    max_h = np.maximum(np.max(f_local[:,:,1]), 384).astype(np.int32)
    cw = max_w - min_w
    ch = max_h - min_h
    # print("f_local[:,:,1]:",np.max(f_local[:,:,1]))
    print("np.maximum(np.max(f_local[:,:,1]), 384):", np.maximum(np.max(f_local[:, :, 1]), 384))
    # print("max_h:", max_h)

    pic = np.ones([ch+10, cw+10, 3], np.int32)*255
    # print("pic:", pic.shape)
    # print("warp:", warp.shape)
    #x = warp[:,:,0]
    #y = warp[:,:,2]
    #warp[:,:,0] = y
    #warp[:,:,2] = x
    pic[0-min_h+5:0-min_h+384+5, 0-min_w+5:0-min_w+512+5, :] = warp
    
    warp = pic
    # print("warp:",warp.shape)
    f_local[:,:,0] = f_local[:,:,0] - min_w+5
    f_local[:,:,1] = f_local[:,:,1] - min_h+5
    f_local = f_local.astype(int)
    # print("f_local:", f_local)
    
    
    point_color = (0, 255, 0) # BGR
    thickness = 2
    lineType = 8
    #cv.circle(warp, (60, 0), 60, point_color, 0)
    #cv.circle(warp, (f_local[0,0,0], f_local[0,0,1]), 5, point_color, 0)
    num = 1
    for i in range(grid_h+1):
        for j in range(grid_w+1):
            #cv.putText(warp, str(num), (f_local[i,j,0], f_local[i,j,1]), cv.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            num = num + 1
            if j == grid_w and i == grid_h:
                continue
            elif j == grid_w:
                cv2.line(warp, (f_local[i,j,0], f_local[i,j,1]), (f_local[i+1,j,0], f_local[i+1,j,1]), point_color, thickness, lineType)
            elif i == grid_h:
                cv2.line(warp, (f_local[i,j,0], f_local[i,j,1]), (f_local[i,j+1,0], f_local[i,j+1,1]), point_color, thickness, lineType)
            else :
                cv2.line(warp, (f_local[i,j,0], f_local[i,j,1]), (f_local[i+1,j,0], f_local[i+1,j,1]), point_color, thickness, lineType)
                cv2.line(warp, (f_local[i,j,0], f_local[i,j,1]), (f_local[i,j+1,0], f_local[i,j+1,1]), point_color, thickness, lineType)
              
    return warp

def inference_func(pathInput2,pathMask2,pathGT2,model_path):
    _origin_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    resize_w, resize_h = args.img_w,args.img_h
    index_all = list(sorted([x.split('.')[0] for x in os.listdir(pathInput2)]))
    # load model
    model = RectanglingNetwork()
    # model.load_state_dict(torch.load(model_path))

    pretrain_model = torch.load(model_path, map_location='cpu')
    # Extract K,V from the existing model
    model_dict = model.state_dict()
    # Create a new weight dictionary and update it
    state_dict = {k: v for k, v in pretrain_model.items() if k in model_dict.keys()}
    # Update the weight dictionary of the existing model
    model_dict.update(state_dict)
    # Load the updated weight dictionary
    model.load_state_dict(model_dict)
    # loading model to device 0
    model = model.cuda(device=args.device_ids[0])
    # model.featureExtrator.fuse()
    model.meshRegression.fuse()
    # print(model)
    model.eval()
    length = 519
    for i in range(0, length):
        idx = index_all[i]
        input_img = cv2.imread(os.path.join(pathInput2, str(idx) + '.jpg')) / 255.
        input_img = cv2.resize(input_img, (resize_w, resize_h))

        mask_img = cv2.imread(os.path.join(pathMask2, str(idx) + '.jpg')) / 255.
        mask_img = cv2.resize(mask_img, (resize_w, resize_h))

        gt_img = cv2.imread(os.path.join(pathGT2, str(idx) + '.jpg')) / 255.
        gt_img = cv2.resize(gt_img, (resize_w, resize_h))

        test_gt = _origin_transform(gt_img).unsqueeze(0).float().to(args.device_ids[0])
        test_input = _origin_transform(input_img).unsqueeze(0).float().to(args.device_ids[0])
        test_mask = _origin_transform(mask_img).unsqueeze(0).float().to(args.device_ids[0])

        test_mesh_primary, test_warp_image_primary, test_warp_mask_primary = model.forward(test_input, test_mask)
        # print("test_mesh_primary:",test_mesh_primary)
        # print("test_warp_image_primary:", test_warp_image_primary.shape)
        # print("test_warp_mask_primary:", test_warp_mask_primary.shape)
        # mask = test_warp_mask_primary[0].permute(1,2,0).cpu().detach().numpy()

        mesh = test_mesh_primary[0].cpu().detach().numpy()
        # mesh = generatorRandomMesh(args.img_h,args.img_w)[0].cpu().detach().numpy()
        input_image = cv2.imread(os.path.join(pathInput2, str(i + 1).zfill(5) + '.jpg'))
        input_image = cv2.resize(input_image, (resize_w, resize_h))
        # input_image = cv2.resize(input_image, (resize_w, resize_h))

        input_image = draw_mesh_on_warp(input_image, mesh)

        path = "final_mesh/" + str(i + 1).zfill(5) + ".jpg"
        cv2.imwrite(path, input_image)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='E:\DataSets\DIR-D')
    parser.add_argument('--device_ids', type=list, default=[0])
    parser.add_argument('--img_h', type=int, default=384)
    parser.add_argument('--img_w', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_epoch', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--model_path', type=str, default='model/distill_model_epoch200.pkl')
    parser.add_argument('--lam_perception', type=float, default=5e-6)
    parser.add_argument('--lam_mask', type=float, default=1)
    parser.add_argument('--lam_mesh', type=float, default=1)
    parser.add_argument('--lam_appearance', type=float, default=1)
    print('<==================== Loading data ===================>\n')

    args = parser.parse_args()
    print(args)
    ##############
    pathGT2 = os.path.join(args.path, 'testing\gt')
    pathInput2 = os.path.join(args.path, 'testing\input')
    pathMask2 = os.path.join(args.path, 'testing\mask')

    ##########testing###############
    inference_func(pathInput2,pathMask2,pathGT2,args.model_path)





                






