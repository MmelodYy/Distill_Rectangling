import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import numpy as np
# import tensorflow.contrib.slim as slim
import tf_slim as slim
# from tensorflow.contrib.layers import conv2d
import tf_spatial_transform_local
import tf_spatial_transform_local_feature

import constant

grid_w = constant.GRID_W
grid_h = constant.GRID_H

def shift2mesh(mesh_shift, width, height):
    batch_size = tf.shape(mesh_shift)[0]
    h = height / grid_h
    w = width / grid_w
    ori_pt = []
    for i in range(grid_h + 1):
        for j in range(grid_w + 1):
            ww = j * w
            hh = i * h
            p = tf.constant([ww, hh], shape=[2], dtype=tf.float32)
            ori_pt.append(tf.expand_dims(p, 0))
    ori_pt = tf.concat(ori_pt, axis=0)
    ori_pt = tf.reshape(ori_pt, [grid_h + 1, grid_w + 1, 2])
    ori_pt = tf.tile(tf.expand_dims(ori_pt, 0), [batch_size, 1, 1, 1])  #

    tar_pt = ori_pt + mesh_shift
    # tar_pt = tf.reshape(tar_pt, [batch_size, grid_h+1, grid_w+1, 2])

    return tar_pt

def imageToFeatures(train_input):
    conv1 = slim.conv2d(inputs=train_input, num_outputs=32, kernel_size=3, rate=1, activation_fn=tf.nn.relu)
    conv2 = slim.conv2d(inputs=conv1, num_outputs=64, kernel_size=3, rate=1, activation_fn=tf.nn.relu)
    conv3 = slim.conv2d(inputs=conv2, num_outputs=3, kernel_size=3, rate=1, activation_fn=tf.nn.relu)
    return conv3

# feature extraction module
def feature_extractor(image_tf):
    feature = []
    # 512*384
    with tf.variable_scope('conv_block1'):
        conv1 = slim.conv2d(inputs=image_tf, num_outputs=64, kernel_size=7,stride=2, rate=1, activation_fn=tf.nn.relu)
        maxpool1 = slim.max_pool2d(conv1, 2, stride=2, padding='SAME')
    # 256*192
    with tf.variable_scope('conv_block2'):
        short_cut2_1 = maxpool1
        conv2_1 = slim.conv2d(inputs=maxpool1, num_outputs=64, kernel_size=3, activation_fn=tf.nn.relu)
        conv2_1 = slim.conv2d(inputs=conv2_1, num_outputs=64, kernel_size=3, activation_fn=tf.nn.relu)
        conv2_1 = conv2_1 + short_cut2_1
        feature.append(conv2_1)
        # maxpool2 = slim.max_pool2d(conv2, 2, stride=2, padding='SAME')
    # 128*96
    with tf.variable_scope('conv_block3'):
        short_cut3_1 = slim.conv2d(inputs=conv2_1, num_outputs=128, kernel_size=1, stride=2, activation_fn=tf.nn.relu)
        conv3_1 = slim.conv2d(inputs=conv2_1, num_outputs=128, kernel_size=3, stride=2, activation_fn=tf.nn.relu)
        conv3_1 = slim.conv2d(inputs=conv3_1, num_outputs=128, kernel_size=3, activation_fn=tf.nn.relu)
        conv3_1 = conv3_1 + short_cut3_1
        feature.append(conv3_1)
        # maxpool3 = slim.max_pool2d(conv3, 2, stride=2, padding='SAME')
    # 64*48
    with tf.variable_scope('conv_block4'):
        short_cut4_1 = slim.conv2d(inputs=conv3_1, num_outputs=256, kernel_size=1, stride=2, activation_fn=tf.nn.relu)
        conv4_1 = slim.conv2d(inputs=conv3_1, num_outputs=256, kernel_size=3, stride=2, activation_fn=tf.nn.relu)
        conv4_1 = slim.conv2d(inputs=conv4_1, num_outputs=256, kernel_size=3, activation_fn=tf.nn.relu)
        conv4_1 = conv4_1 + short_cut4_1
        feature.append(conv4_1)
        # maxpool4 = slim.max_pool2d(conv4, 2, stride=2, padding='SAME')
    return feature


# mesh motion regression module
def regression_Net(correlation):
    short_cut1 = slim.conv2d(inputs=correlation, num_outputs=256, kernel_size=1, stride=1, activation_fn=tf.nn.relu)
    conv1 = slim.conv2d(inputs=correlation, num_outputs=256, kernel_size=3, activation_fn=tf.nn.relu)
    conv1 = slim.conv2d(inputs=conv1, num_outputs=256, kernel_size=3, activation_fn=tf.nn.relu)
    conv1 = conv1 + short_cut1

    short_cut2 = slim.conv2d(inputs=conv1, num_outputs=256, kernel_size=1, stride=2, activation_fn=tf.nn.relu)
    conv2 = slim.conv2d(inputs=conv1, num_outputs=256, kernel_size=3, stride=2, activation_fn=tf.nn.relu)
    conv2 = slim.conv2d(inputs=conv2, num_outputs=256, kernel_size=3, activation_fn=tf.nn.relu)
    conv2 = conv2 + short_cut2

    short_cut3 = slim.conv2d(inputs=conv2, num_outputs=512, kernel_size=1, stride=2, activation_fn=tf.nn.relu)
    conv3 = slim.conv2d(inputs=conv2, num_outputs=512, kernel_size=3, stride=2, activation_fn=tf.nn.relu)
    conv3 = slim.conv2d(inputs=conv3, num_outputs=512, kernel_size=3, activation_fn=tf.nn.relu)
    conv3 = conv3 + short_cut3

    short_cut4 = slim.conv2d(inputs=conv3, num_outputs=512, kernel_size=1, stride=2, activation_fn=tf.nn.relu)
    conv4 = slim.conv2d(inputs=conv3, num_outputs=512, kernel_size=3, stride=2, activation_fn=tf.nn.relu)
    conv4 = slim.conv2d(inputs=conv4, num_outputs=512, kernel_size=3, activation_fn=tf.nn.relu)
    conv4 = conv4 + short_cut4

    fc1 = slim.conv2d(inputs=conv4, num_outputs=2048, kernel_size=[3, 4], activation_fn=tf.nn.relu, padding="VALID")
    fc2 = slim.conv2d(inputs=fc1, num_outputs=1024, kernel_size=1, activation_fn=tf.nn.relu)
    fc3 = slim.conv2d(inputs=fc2, num_outputs=(grid_w + 1) * (grid_h + 1) * 2, kernel_size=1, activation_fn=None)
    # net3_f = tf.expand_dims(tf.squeeze(tf.squeeze(fc3,1),1), [2])
    net3_f_local = tf.reshape(fc3, (-1, grid_h + 1, grid_w + 1, 2))

    return net3_f_local


#####################################################
# # two level prediction
def build_model(train_input, train_mask):
    with tf.variable_scope('model'):
        batch_size = tf.shape(train_input)[0]

        with tf.variable_scope('image_to_feature', reuse=None):
            train_input_f = imageToFeatures(train_input)
            train_input_f_m = tf.multiply(train_input_f, tf.nn.relu(train_mask))

        # train_input_f_m = tf.concat([train_input, train_mask],axis=-1)
        with tf.variable_scope('feature_extract', reuse=None):
            features = feature_extractor(train_input_f_m)

        w,h = 24,32
        feature0 = tf.image.resize_images(features[-1], [w,h], method=0)
        with tf.variable_scope('regression_primary', reuse=None):
            mesh_shift_primary = regression_Net(feature0)

        w,h = 24,32
        s1 = 16
        feature1 = tf.image.resize_images(features[-1], [w,h], method=0)
        with tf.variable_scope('regression_mid', reuse=None):
            mesh_primary = shift2mesh(mesh_shift_primary / s1, float(h), float(w))
            feature_warp = tf_spatial_transform_local_feature.transformer(feature1, mesh_primary,float(h),float(w))
            mesh_shift_mid = regression_Net(feature_warp)

        return mesh_shift_primary, mesh_shift_mid

def RectanglingNetwork(train_input, train_mask, width=512., height=384.):
    batch_size = tf.shape(train_input)[0]

    mesh_shift_primary, mesh_shift_mid = build_model(train_input, train_mask)

    mesh_primary = shift2mesh(mesh_shift_primary, width, height)  # 标准矩形网格和传进来的模型预测的网格动量相加,得到最终偏移量
    mesh_mid = shift2mesh(mesh_shift_mid + mesh_shift_primary, width, height)

    warp_image_primary, warp_mask_primary = tf_spatial_transform_local.transformer(train_input, train_mask, mesh_primary)
    warp_image_mid, warp_mask_mid = tf_spatial_transform_local.transformer(train_input, train_mask, mesh_mid)

    warp_image_primary = tf.multiply(warp_image_primary, tf.nn.relu(warp_mask_primary))
    warp_image_mid = tf.multiply(warp_image_mid, tf.nn.relu(warp_mask_mid))

    return mesh_primary, warp_image_primary, warp_mask_primary, mesh_mid, warp_image_mid, warp_mask_mid
