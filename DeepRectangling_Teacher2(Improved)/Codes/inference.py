import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import os
import time
import numpy as np
import pickle
import cv2

from PIL import Image
from model import RectanglingNetwork
from utils import load, save, DataLoader
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

import constant

os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = constant.GPU

test_folder = constant.TEST_FOLDER
batch_size = constant.TEST_BATCH_SIZE

# snapshot_dir = constant.SNAPSHOT_DIR + '/pretrained_model/model.ckpt-100000'
snapshot_dir = './checkpoints_16_f1(2)/model.ckpt-100000'

batch_size = 1

# define dataset
with tf.name_scope('dataset'):
    ##########testing###############
    test_inputs_clips_tensor = tf.placeholder(shape=[batch_size, None, None, 3 * 3], dtype=tf.float32)

    test_input = test_inputs_clips_tensor[..., 0:3]
    test_mask = test_inputs_clips_tensor[..., 3:6]
    test_gt = test_inputs_clips_tensor[..., 6:9]

    print('test input = {}'.format(test_input))
    print('test mask = {}'.format(test_mask))
    print('test gt = {}'.format(test_gt))

# define testing generator function
with tf.variable_scope('generator', reuse=None):
    print('testing = {}'.format(tf.get_variable_scope().name))
    # one level model
    # test_mesh_primary, test_warp_image_primary, test_warp_mask_primary = RectanglingNetwork(test_input, test_mask)

    # two level model
    test_mesh_primary, test_warp_image_primary, test_warp_mask_primary, test_mesh_mid, test_warp_image_mid, test_warp_mask_mid = RectanglingNetwork(test_input, test_mask)

    # three level model
    # test_mesh_primary, test_warp_image_primary,  test_warp_mask_primary, test_mesh_mid, test_warp_image_mid,  test_warp_mask_mid, \
    # test_mesh_final, test_warp_image_final, test_warp_mask_final = RectanglingNetwork(test_input, test_mask)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    # dataset
    input_loader = DataLoader(test_folder)

    # initialize weights
    sess.run(tf.global_variables_initializer())
    print('Init global successfully!')

    # tf saver
    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=None)

    restore_var = [v for v in tf.global_variables()]
    loader = tf.train.Saver(var_list=restore_var)


    def inference_func(ckpt):
        print("============")
        print(ckpt)
        load(loader, sess, ckpt)
        print("============")
        length = len(os.listdir(test_folder+"/input")) #519#
        psnr_list = []
        ssim_list = []

        for i in range(0, length):
            input_clip = np.expand_dims(input_loader.get_data_clips(i), axis=0)
            # inference one level model
            # mesh_primary, warp_image_primary, warp_mask_primary = sess.run(
            #     [test_mesh_primary, test_warp_image_primary, test_warp_mask_primary],
            #     feed_dict={test_inputs_clips_tensor: input_clip})

            # inference two level model
            mesh_primary, warp_image_primary, warp_mask_primary, mesh_mid, warp_image_mid, warp_mask_mid = sess.run(
                [test_mesh_primary, test_warp_image_primary, test_warp_mask_primary, test_mesh_mid, test_warp_image_mid,
                 test_warp_mask_mid],
                feed_dict={test_inputs_clips_tensor: input_clip})

            # inference three level model
            # mesh_primary, warp_image_primary, warp_mask_primary, mesh_mid, warp_image_mid, warp_mask_mid, \
            # mesh_final, warp_image_final,  warp_mask_final = sess.run(
            #     [test_mesh_primary, test_warp_image_primary,test_warp_mask_primary, test_mesh_mid, test_warp_image_mid,
            #      test_warp_mask_mid, test_mesh_final, test_warp_image_final, test_warp_mask_final],
            #     feed_dict={test_inputs_clips_tensor: input_clip})

            #################################
            # one level
            # warp_image2 = (warp_image_primary[0] + 1) * 127.5
            # warp_gt = (input_clip[0, :, :, 6:9] + 1) * 127.5

            # two level
            warp_image1 = (warp_image_primary[0] + 1) * 127.5
            warp_image2 = (warp_image_mid[0] + 1) * 127.5
            warp_gt = (input_clip[0, :, :, 6:9] + 1) * 127.5

            # Three level
            # warp_image1 = (warp_image_primary[0] + 1) * 127.5
            # warp_image2 = (warp_image_mid[0] + 1) * 127.5
            # warp_image3 = (warp_image_final[0] + 1) * 127.5
            # warp_gt = (input_clip[0, :, :, 6:9] + 1) * 127.5
            # warp_mask3 = (warp_mask_final[0] + 1) * 127.5
            # warp_image2 = warp_image3

            # warped_image1 = (warped_image_primary[0] + 1) * 127.5
            # warped_image2 = (warped_image_mid[0] + 1) * 127.5
            # warped_image3 = (warped_image_final[0] + 1) * 127.5
            # warp_mask1 = (warp_mask_primary[0] + 1) * 127.5
            # warp_mask2 = (warp_mask_mid[0] + 1) * 127.5

            # path = "./result/final_image/" + str(i + 1).zfill(5) + ".jpg"
            # cv2.imwrite(path, warped_image3)
            # path = "./result/primary_image/" + str(i + 1).zfill(5) + ".jpg"
            # cv2.imwrite(path, warped_image1)
            # path = "./result/middle_image/" + str(i + 1).zfill(5) + ".jpg"
            # cv2.imwrite(path, warped_image2)
            # path = "./result/primary_mask/" + str(i + 1).zfill(5) + ".jpg"
            # cv2.imwrite(path, warp_mask1)
            # path = "./result/middle_mask/" + str(i + 1).zfill(5) + ".jpg"
            # cv2.imwrite(path, warp_mask2)

            psnr = compare_psnr(warp_image2, warp_gt, data_range=255)
            ssim = compare_ssim(warp_image2, warp_gt, data_range=255, channel_axis=2)

            # path_mesh = "./result/train_final_mesh/" + str(i + 1).zfill(5) + ".npy"
            # np.save(path_mesh, mesh_final)
            #
            # path = "./result/final_rectangling/" + str(i + 1).zfill(5) + ".jpg"
            # cv2.imwrite(path, warp_image3)
            # path = "./result/final_mask/" + str(i + 1).zfill(5) + ".jpg"
            # cv2.imwrite(path, warp_mask3)
            # path = "./result/primary_rectangling/" + str(i + 1).zfill(5) + ".jpg"
            # cv2.imwrite(path, warp_image1)
            #
            # path = "./result/middle_rectangling/" + str(i + 1).zfill(5) + ".jpg"
            # cv2.imwrite(path, warp_image2)

            fusion = np.zeros_like(warp_gt, dtype=np.float64)
            fusion[..., 0] = warp_image2[..., 0]
            fusion[..., 1] = warp_gt[..., 1] * 0.5 + warp_image2[..., 1] * 0.5
            fusion[..., 2] = warp_gt[..., 2]
            fusion = np.clip(fusion, 0, 255.)
            path = "./result/teacher_2level_fusion/" + str(i + 1).zfill(5) + ".jpg"
            cv2.imwrite(path, fusion)

            print('i = {} / {}, psnr = {:.6f}'.format(i + 1, length, psnr))

            psnr_list.append(psnr)
            ssim_list.append(ssim)

        print("===================Results Analysis==================")
        print('average psnr:', np.mean(psnr_list))
        print('average ssim:', np.mean(ssim_list))
        # as for FID, we use the CODE from https://github.com/bioinf-jku/TTUR to evaluate


    inference_func(snapshot_dir)








