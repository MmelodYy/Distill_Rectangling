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
# import skimage
# import imageio
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr


import constant
os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = constant.GPU

# !!!! using TRAIN_FOLDER instead of TEST_FOLDER
test_folder = constant.TRAIN_FOLDER
batch_size = constant.TEST_BATCH_SIZE



snapshot_dir = constant.SNAPSHOT_DIR + '/pretrained_model/model.ckpt-100000'


batch_size = 1

# define dataset
with tf.name_scope('dataset'):
    ##########testing###############
    test_inputs_clips_tensor = tf.placeholder(shape=[batch_size, None, None, 3 * 3], dtype=tf.float32)
    
    test_input = test_inputs_clips_tensor[...,0:3]
    test_mask = test_inputs_clips_tensor[...,3:6]
    test_gt = test_inputs_clips_tensor[...,6:9]
    
    print('test input = {}'.format(test_input))
    print('test mask = {}'.format(test_mask))
    print('test gt = {}'.format(test_gt))



# define testing generator function 
with tf.variable_scope('generator', reuse=None):
    print('testing = {}'.format(tf.get_variable_scope().name))
    test_mesh_primary, test_warp_image_primary, test_warp_mask_primary, test_mesh_final, test_warp_image_final, test_warp_mask_final = RectanglingNetwork(test_input, test_mask)
   


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
        length = len(os.listdir(test_folder+"/input"))
        psnr_list = []
        ssim_list = []

        for i in range(0, length):
            input_clip = np.expand_dims(input_loader.get_data_clips(i), axis=0)
            

            mesh_primary, warp_image_primary, warp_mask_primary, mesh_final, warp_image_final, warp_mask_final  = sess.run([test_mesh_primary, test_warp_image_primary, test_warp_mask_primary, test_mesh_final, test_warp_image_final, test_warp_mask_final], feed_dict={test_inputs_clips_tensor: input_clip})
            
            
            warp_image = (warp_image_final[0]+1) * 127.5
            warp_gt = (input_clip[0,:,:,6:9]+1) * 127.5
            
            psnr = compare_psnr(warp_image, warp_gt, data_range=255)
            ssim = compare_ssim(warp_image, warp_gt, data_range=255, channel_axis=2)


            path = "../train_final_mesh/" + str(i + 1).zfill(5) + ".npy"
            np.save(path, mesh_final[0,:,:,:])

            # path = "../train_final_img/" + str(i + 1).zfill(5) + ".jpg"
            # cv2.imwrite(path, warp_image)

            arrResult = np.concatenate([[psnr], [ssim]], axis=0).reshape(2, 1)
            # print("arrResult:", arrResult.shape)
            weight_path = "../result/weight/" + str(i + 1).zfill(5) + ".npy"
            np.save(weight_path, arrResult)
            
            print('i = {} / {}, psnr = {:.6f}, ssim = {:.6f}'.format( i+1, length, psnr, ssim))
            
            psnr_list.append(psnr)
            ssim_list.append(ssim)
            
            
        print("=================== Mesh data generation completed ==================")

    inference_func(snapshot_dir)

    






