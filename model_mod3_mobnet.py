#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

from keras.models import Model
from keras.layers.merge import Concatenate
from keras.layers import Activation, Input, Lambda, BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import Multiply
from keras.regularizers import l2
from keras.initializers import random_normal,constant

import keras.backend as K

import keras.applications.mobilenet as mobnet
from keras.applications.mobilenet import DepthwiseConv2D, _depthwise_conv_block, _conv_block

def relu(x): return Activation('relu')(x)

def conv(x, nf, ks, name, weight_decay):
    kernel_reg = l2(weight_decay[0]) if weight_decay else None
    bias_reg = l2(weight_decay[1]) if weight_decay else None

    x = Conv2D(nf, (ks, ks), padding='same', name=name,
               kernel_regularizer=kernel_reg,
               bias_regularizer=bias_reg,
               kernel_initializer=random_normal(stddev=0.01),
               bias_initializer=constant(0.0))(x)
    return x

def pooling(x, ks, st, name):
    x = MaxPooling2D((ks, ks), strides=(st, st), name=name)(x)
    return x

def vgg_block(x, weight_decay):
    # Block 1
    x = conv(x, 32, 3, "conv1_1", (weight_decay, 0))
    x = relu(x)
    x = conv(x, 32, 3, "conv1_2", (weight_decay, 0))
    x = relu(x)
    x = pooling(x, 2, 2, "pool1_1")
    # Block 2
    x = conv(x, 64, 3, "conv2_1", (weight_decay, 0))
    x = relu(x)
    x = conv(x, 64, 3, "conv2_2", (weight_decay, 0))
    x = relu(x)
    x = pooling(x, 2, 2, "pool2_1")
    # Block 3
    x = conv(x, 128, 3, "conv3_1", (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 3, "conv3_2", (weight_decay, 0))
    x = relu(x)
    x = pooling(x, 2, 2, "pool3_1")
    # Block 4
    x = conv(x, 256, 3, "conv4_1", (weight_decay, 0))
    x = relu(x)
    x = conv(x, 256, 3, "conv4_2", (weight_decay, 0))
    x = relu(x)
    # Additional non vgg layers
    x = conv(x, 256, 3, "conv4_3_CPM", (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 3, "conv4_4_CPM", (weight_decay, 0))
    x = relu(x)
    return x

def stage1_block(x, num_p, branch, weight_decay):
    # Block 1
    x = conv(x, 128, 3, "Mconv1_stage1_L%d" % branch, (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 3, "Mconv2_stage1_L%d" % branch, (weight_decay, 0))
    x = relu(x)
    # x = conv(x, 128, 3, "Mconv3_stage1_L%d" % branch, (weight_decay, 0))
    # x = relu(x)
    # x = conv(x, 512, 1, "Mconv4_stage1_L%d" % branch, (weight_decay, 0))
    # x = relu(x)
    x = conv(x, num_p, 1, "Mconv5_stage1_L%d" % branch, (weight_decay, 0))
    return x

def stageT_block(x, num_p, stage, branch, weight_decay):
    # Block 1
    x = conv(x, 128, 5, "Mconv1_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 3, "Mconv2_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = relu(x)
    # x = conv(x, 128, 7, "Mconv1_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    # x = relu(x)
    # x = conv(x, 128, 7, "Mconv2_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    # x = relu(x)
    # x = conv(x, 128, 7, "Mconv3_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    # x = relu(x)
    # x = conv(x, 128, 7, "Mconv4_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    # x = relu(x)
    # x = conv(x, 128, 7, "Mconv5_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    # x = relu(x)
    # x = conv(x, 128, 1, "Mconv6_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    # x = relu(x)
    x = conv(x, num_p, 1, "Mconv7_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    return x

def mobilenet_block(x, weight_decay, palpha=1.0, depth_multiplier=1):
    x = _conv_block(x, 32, palpha, strides=(2, 2))
    x = _depthwise_conv_block_mod(x, 64, palpha, depth_multiplier, block_id=1)

    x = _depthwise_conv_block_mod(x, 128, palpha, depth_multiplier, strides=(2, 2), block_id=2)
    x = _depthwise_conv_block_mod(x, 128, palpha, depth_multiplier, block_id=3)

    x = _depthwise_conv_block_mod(x, 256, palpha, depth_multiplier, strides=(2, 2), block_id=4)
    x = _depthwise_conv_block_mod(x, 256, palpha, depth_multiplier, block_id=5)

    # x = _depthwise_conv_block_mod(x, 512, palpha, depth_multiplier, strides=(2, 2), block_id=6)
    x = _depthwise_conv_block_mod(x, 512, palpha, depth_multiplier, strides=(1, 1), block_id=6)
    x = _depthwise_conv_block_mod(x, 512, palpha, depth_multiplier, block_id=7)
    x = _depthwise_conv_block_mod(x, 512, palpha, depth_multiplier, block_id=8)
    # x = _depthwise_conv_block_mod(x, 512, palpha, depth_multiplier, block_id=9)
    # x = _depthwise_conv_block_mod(x, 512, palpha, depth_multiplier, block_id=10)
    # x = _depthwise_conv_block_mod(x, 512, palpha, depth_multiplier, block_id=11)

    # x = _depthwise_conv_block_mod(x, 1024, palpha, depth_multiplier, strides=(2, 2), block_id=12)
    # x = _depthwise_conv_block_mod(x, 1024, palpha, depth_multiplier, block_id=13)
    return x

def stage1_block_mobilenet(x, num_p, branch, weight_decay, palpha = 1.0, depth_multiplier=1):
    # Block 1
    x = _depthwise_conv_block_mod(x, 192, palpha, depth_multiplier, block_name="Mconv1_stage1_L%d" % branch)
    x = _depthwise_conv_block_mod(x, 96, palpha, depth_multiplier, block_name="Mconv2_stage1_L%d" % branch)
    # x = conv(x, 128, 3, "Mconv1_stage1_L%d" % branch, (weight_decay, 0))
    # x = relu(x)
    # x = conv(x, 128, 3, "Mconv2_stage1_L%d" % branch, (weight_decay, 0))
    # x = relu(x)
    # x = conv(x, 128, 3, "Mconv3_stage1_L%d" % branch, (weight_decay, 0))
    # x = relu(x)
    # x = conv(x, 512, 1, "Mconv4_stage1_L%d" % branch, (weight_decay, 0))
    # x = relu(x)
    x = conv(x, num_p, 1, "Mconv3_stage1_L%d" % branch, (weight_decay, 0))
    return x


def stageT_block_mobilenet(x, num_p, stage, branch, weight_decay, palpha = 1.0, depth_multiplier=1):
    x = _depthwise_conv_block_mod(x, 192, palpha, depth_multiplier, block_name="Mconv1_stage%d_L%d" % (stage, branch))
    x = _depthwise_conv_block_mod(x, 96, palpha, depth_multiplier, block_name="Mconv2_stage%d_L%d" % (stage, branch))
    # Block 1
    # x = conv(x, 128, 5, "Mconv1_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    # x = relu(x)
    # x = conv(x, 128, 3, "Mconv2_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    # x = relu(x)
    # x = conv(x, 128, 7, "Mconv1_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    # x = relu(x)
    # x = conv(x, 128, 7, "Mconv2_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    # x = relu(x)
    # x = conv(x, 128, 7, "Mconv3_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    # x = relu(x)
    # x = conv(x, 128, 7, "Mconv4_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    # x = relu(x)
    # x = conv(x, 128, 7, "Mconv5_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    # x = relu(x)
    # x = conv(x, 128, 1, "Mconv6_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    # x = relu(x)
    x = conv(x, num_p, 1, "Mconv3_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    return x

def apply_mask(x, mask1, mask2, num_p, stage, branch):
    w_name = "weight_stage%d_L%d" % (stage, branch)
    if num_p == 38:
        w = Multiply(name=w_name)([x, mask1]) # vec_weight

    else:
        w = Multiply(name=w_name)([x, mask2])  # vec_heat
    return w


def get_training_model(weight_decay, pinpShape = None, palpha=1.0, pstages = 6):

    stages = pstages
    np_branch1 = 38
    np_branch2 = 19

    if pinpShape is None:
        img_input_shape = (None, None, 3)
    else:
        img_input_shape = pinpShape
    vec_input_shape = (None, None, 38)
    heat_input_shape = (None, None, 19)

    inputs = []
    outputs = []

    img_input = Input(shape=img_input_shape)
    vec_weight_input = Input(shape=vec_input_shape)
    heat_weight_input = Input(shape=heat_input_shape)

    inputs.append(img_input)
    inputs.append(vec_weight_input)
    inputs.append(heat_weight_input)

    img_normalized = Lambda(lambda x: x / 256 - 0.5)(img_input) # [-0.5, 0.5]

    # VGG
    # stage0_out = vgg_block(img_normalized, weight_decay)

    # MobileNet
    stage0_out = mobilenet_block(img_normalized, weight_decay, palpha=palpha)

    # stage 1 - branch 1 (PAF)
    # stage1_branch1_out = stage1_block(stage0_out, np_branch1, 1, weight_decay)
    stage1_branch1_out = stage1_block_mobilenet(stage0_out, np_branch1, 1, weight_decay, palpha=palpha)
    w1 = apply_mask(stage1_branch1_out, vec_weight_input, heat_weight_input, np_branch1, 1, 1)

    # stage 1 - branch 2 (confidence maps)
    # stage1_branch2_out = stage1_block(stage0_out, np_branch2, 2, weight_decay)
    stage1_branch2_out = stage1_block_mobilenet(stage0_out, np_branch2, 2, weight_decay, palpha=palpha)
    w2 = apply_mask(stage1_branch2_out, vec_weight_input, heat_weight_input, np_branch2, 1, 2)

    x = Concatenate()([stage1_branch1_out, stage1_branch2_out, stage0_out])

    outputs.append(w1)
    outputs.append(w2)

    # stage sn >= 2
    for sn in range(2, stages + 1):
        # stage SN - branch 1 (PAF)
        # stageT_branch1_out = stageT_block(x, np_branch1, sn, 1, weight_decay)
        stageT_branch1_out = stageT_block_mobilenet(x, np_branch1, sn, 1, weight_decay, palpha=palpha)
        w1 = apply_mask(stageT_branch1_out, vec_weight_input, heat_weight_input, np_branch1, sn, 1)

        # stage SN - branch 2 (confidence maps)
        # stageT_branch2_out = stageT_block(x, np_branch2, sn, 2, weight_decay)
        stageT_branch2_out = stageT_block_mobilenet(x, np_branch2, sn, 2, weight_decay, palpha=palpha)
        w2 = apply_mask(stageT_branch2_out, vec_weight_input, heat_weight_input, np_branch2, sn, 2)

        outputs.append(w1)
        outputs.append(w2)

        if (sn < stages):
            x = Concatenate()([stageT_branch1_out, stageT_branch2_out, stage0_out])

    model = Model(inputs=inputs, outputs=outputs)
    return model


def get_testing_model(pinpShape = None, palpha=1.0, pstages=6):
    stages = pstages
    np_branch1 = 38
    np_branch2 = 19

    if pinpShape is None:
        img_input_shape = (None, None, 3)
    else:
        img_input_shape = pinpShape

    img_input = Input(shape=img_input_shape)

    img_normalized = Lambda(lambda x: x / 256 - 0.5)(img_input) # [-0.5, 0.5]

    # VGG
    # stage0_out = vgg_block(img_normalized, None)
    stage0_out = mobilenet_block(img_normalized, None, palpha=palpha)

    # stage 1 - branch 1 (PAF)
    # stage1_branch1_out = stage1_block(stage0_out, np_branch1, 1, None)
    stage1_branch1_out = stage1_block_mobilenet(stage0_out, np_branch1, 1, None, palpha=palpha)

    # stage 1 - branch 2 (confidence maps)
    # stage1_branch2_out = stage1_block(stage0_out, np_branch2, 2, None)
    stage1_branch2_out = stage1_block_mobilenet(stage0_out, np_branch2, 2, None, palpha=palpha)

    x = Concatenate()([stage1_branch1_out, stage1_branch2_out, stage0_out])

    # stage t >= 2
    stageT_branch1_out = None
    stageT_branch2_out = None
    for sn in range(2, stages + 1):
        # stageT_branch1_out = stageT_block(x, np_branch1, sn, 1, None)
        # stageT_branch2_out = stageT_block(x, np_branch2, sn, 2, None)
        stageT_branch1_out = stageT_block_mobilenet(x, np_branch1, sn, 1, None, palpha=palpha)
        stageT_branch2_out = stageT_block_mobilenet(x, np_branch2, sn, 2, None, palpha=palpha)

        if (sn < stages):
            x = Concatenate()([stageT_branch1_out, stageT_branch2_out, stage0_out])

    model = Model(inputs=[img_input], outputs=[stageT_branch1_out, stageT_branch2_out])

    return model


def _depthwise_conv_block_mod(inputs, pointwise_conv_filters, alpha,
                          depth_multiplier=1, strides=(1, 1), block_id=1, block_name = None):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)
    #
    if block_name is None:
        tname_convd = 'conv_dw_%d'      % block_id
        tname_relud = 'conv_dw_%d_relu' % block_id
        tname_bnd   = 'conv_dw_%d_bn'   % block_id
        tname_convp = 'conv_pw_%d'      % block_id
        tname_relup = 'conv_pw_%d_relu' % block_id
        tname_bnp   = 'conv_pw_%d_bn'   % block_id
    else:
        tname_convd = 'conv_dw_%s'      % block_name
        tname_relud = 'conv_dw_%s_relu' % block_name
        tname_bnd   = 'conv_dw_%s_bn'   % block_name
        tname_convp = 'conv_pw_%s'      % block_name
        tname_relup = 'conv_pw_%s_relu' % block_name
        tname_bnp   = 'conv_pw_%s_bn'   % block_name
    #
    x = DepthwiseConv2D((3, 3),
                        padding='same',
                        depth_multiplier=depth_multiplier,
                        strides=strides,
                        use_bias=False,
                        name=tname_convd)(inputs)
    x = BatchNormalization(axis=channel_axis, name=tname_bnd)(x)
    x = Activation(relu6, name=tname_relud)(x)
    x = Conv2D(pointwise_conv_filters, (1, 1),
               padding='same',
               use_bias=False,
               strides=(1, 1),
               name=tname_convp)(x)
    x = BatchNormalization(axis=channel_axis, name=tname_bnp)(x)
    return Activation(relu6, name=tname_relup)(x)

def relu6(x):
    return K.relu(x, max_value=6)