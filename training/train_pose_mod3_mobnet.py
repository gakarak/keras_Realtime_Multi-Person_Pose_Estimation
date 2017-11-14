#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import sys
import os
import pandas
import re
import math

import argparse

import numpy as np
import matplotlib.pyplot as plt

sys.path.append("..")
from model_mod2_mobnet import get_training_model, get_testing_model

try:
    from .ds_iterator import DataIterator
    from .ds_generator_client import DataGeneratorClient
    from .optimizers import MultiSGD
except:
    from ds_iterator import DataIterator
    from ds_generator_client import DataGeneratorClient
    from optimizers import MultiSGD

from keras.callbacks import LearningRateScheduler, ModelCheckpoint, CSVLogger, TensorBoard
from keras.layers.convolutional import Conv2D
from keras.applications.vgg19 import VGG19
import keras.applications.mobilenet as mobnet
import keras.backend as K
import tensorflow as tf

import keras

parser = argparse.ArgumentParser()
parser.add_argument('--batch',   required=False, help='batch size', default=16, type=int)
parser.add_argument('--gpumem',  required=False, help='gpu memory usage fraction', default=None, type=float)
parser.add_argument('--palpha',  required=False, help='model parameter: alpha', default=1.0, type=float)
parser.add_argument('--pstages', required=False, help='model parameter: stages', default=6, type=int)
parser.add_argument('--weights', required=False, help='model file for weiths initialization', default=None)
parser.add_argument('--port',    required=False, help='img-augmentation service port', default=5557, type=int)
pargs = parser.parse_args()

paramAlpha = pargs.palpha
# paramNumStages = 6
paramNumStages = pargs.pstages


WEIGHTS_BEST = "weights_mobilenet_best_a{}_s{}.h5".format(paramAlpha, paramNumStages)
TRAINING_LOG = "log_trai_mobilenet_a{}_s{}.csv".format(paramAlpha, paramNumStages)
LOGS_DIR = "./logs/log_mobilenet_a{}_s{}".format(paramAlpha, paramNumStages)

print("""
[*] parameters
--------------------
    batch-size:  {},
    gpumem:      {},
    net-alpha:   {},
    net-stages:  {},
    net-weights: {},
    PORT(augm):  {}
    -
    log-dir:     {},
    log-csv:     {},
    best_model:  {}
""".format(pargs.batch, pargs.gpumem, pargs.palpha,
           pargs.pstages, pargs.weights, pargs.port,
           WEIGHTS_BEST, TRAINING_LOG, LOGS_DIR))

###############################
if pargs.gpumem is not None:
    configGPU = tf.ConfigProto()
    configGPU.gpu_options.per_process_gpu_memory_fraction = pargs.gpumem
    keras.backend.tensorflow_backend.set_session(tf.Session(config=configGPU))

batch_size = pargs.batch
base_lr = 4e-5 # 2e-5
momentum = 0.9
weight_decay = 5e-4
lr_policy =  "step"
gamma = 0.333
stepsize = 136106 #68053   // after each stepsize iterations update learning rate: lr=lr*gamma
max_iter = 200000 # 600000

# True = start data generator client, False = use augmented dataset file (deprecated)
use_client_gen = True

# euclidean loss as implemented in caffe https://github.com/BVLC/caffe/blob/master/src/caffe/layers/euclidean_loss_layer.cpp
def eucl_loss(x, y):
    return K.sum(K.square(x - y)) / batch_size / 2

def get_last_epoch():
    data = pandas.read_csv(TRAINING_LOG)
    return max(data['epoch'].values)


model = get_training_model(weight_decay, pstages=paramNumStages, palpha=paramAlpha)
# model = get_testing_model(pstages=paramNumStages, pinpShape=(512, 512, 3), palpha=paramAlpha)
# model = get_testing_model(pstages=paramNumStages, palpha=paramAlpha)
model.summary()

# load previous weights or vgg19 if this is the first run
if os.path.exists(WEIGHTS_BEST):
    print("Loading the best weights file... [{}]".format(WEIGHTS_BEST))
    model.load_weights(WEIGHTS_BEST)
    last_epoch = get_last_epoch() + 1
else:
    if paramAlpha == 1.0:
        alpha_text = '1_0'
    elif paramAlpha == 0.75:
        alpha_text = '7_5'
    elif paramAlpha == 0.50:
        alpha_text = '5_0'
    else:
        alpha_text = '2_5'
    model_name = 'mobilenet_%s_%d_tf_no_top.h5' % (alpha_text, 224)
    if pargs.weights is None:
        weights_url = mobnet.BASE_WEIGHT_PATH + model_name
        weights_path = keras.utils.data_utils.get_file(model_name, weights_url, cache_subdir='models')
        print("No pretrained model found, loading ImageNet-pretrained model weights from [{}]".format(weights_path))
    else:
        pathWeightsP = pargs.weights
        print("!!! WARNING !!! Loading pretrained model weights from [{}]".format(pathWeightsP))
    model.load_weights(pathWeightsP, by_name=True)

    # print("Loading vgg19 weights...")
    # vgg_model = VGG19(include_top=False, weights='imagenet')
    # for layer in model.layers:
    #     if layer.name in from_vgg:
    #         vgg_layer_name = from_vgg[layer.name]
    #         layer.set_weights(vgg_model.get_layer(vgg_layer_name).get_weights())
    #         print("Loaded VGG19 layer: " + vgg_layer_name)
    last_epoch = 0

# prepare generators

if use_client_gen:
    train_client = DataGeneratorClient(port=pargs.port, host="localhost", hwm=160,
                                       batch_size=batch_size,
                                       pstages=paramNumStages)
    train_client.start()
    train_di = train_client.gen()
    train_samples = 52597

    val_client = DataGeneratorClient(port=pargs.port + 1, host="localhost", hwm=160,
                                     batch_size=batch_size,
                                     pstages=paramNumStages)
    val_client.start()
    val_di = val_client.gen()
    val_samples = 2645
else:
    train_di = DataIterator("../dataset/train_dataset.h5", data_shape=(3, 368, 368),
                      mask_shape=(1, 46, 46),
                      label_shape=(57, 46, 46),
                      vec_num=38, heat_num=19, batch_size=batch_size, shuffle=True)
    train_samples=train_di.N
    val_di = DataIterator("../dataset/val_dataset.h5", data_shape=(3, 368, 368),
                      mask_shape=(1, 46, 46),
                      label_shape=(57, 46, 46),
                      vec_num=38, heat_num=19, batch_size=batch_size, shuffle=True)
    val_samples=val_di.N

# setup lr multipliers for conv layers
lr_mult=dict()
# for layer in model.layers:
#     if isinstance(layer, Conv2D) or isinstance(layer, mobnet.DepthwiseConv2D):
#         # stage = 1
#         if re.match(".*Mconv\d_stage1.*", layer.name):
#             kernel_name = layer.weights[0].name
#             lr_mult[kernel_name] = 1
#             if len(layer.weights) > 1:
#                 bias_name = layer.weights[1].name
#                 lr_mult[bias_name] = 2
#         # stage > 1
#         elif re.match(".*Mconv\d_stage.*", layer.name):
#             kernel_name = layer.weights[0].name
#             lr_mult[kernel_name] = 4
#             if len(layer.weights) > 1:
#                 bias_name = layer.weights[1].name
#                 lr_mult[bias_name] = 8
#         # vgg
#         else:
#            kernel_name = layer.weights[0].name
#            lr_mult[kernel_name] = 1
#            if len(layer.weights)>1:
#                bias_name = layer.weights[1].name
#                lr_mult[bias_name] = 2

# configure loss functions
# losses = {}
# losses["weight_stage1_L1"] = eucl_loss
# losses["weight_stage1_L2"] = eucl_loss
# losses["weight_stage2_L1"] = eucl_loss
# losses["weight_stage2_L2"] = eucl_loss
# losses["weight_stage3_L1"] = eucl_loss
# losses["weight_stage3_L2"] = eucl_loss
# losses["weight_stage4_L1"] = eucl_loss
# losses["weight_stage4_L2"] = eucl_loss
# losses["weight_stage5_L1"] = eucl_loss
# losses["weight_stage5_L2"] = eucl_loss
# losses["weight_stage6_L1"] = eucl_loss
# losses["weight_stage6_L2"] = eucl_loss
losses = {xx.name: eucl_loss for xx in model.output_layers}


# learning rate schedule - equivalent of caffe lr_policy =  "step"
iterations_per_epoch = train_samples // batch_size
def step_decay(epoch):
    initial_lrate = base_lr
    steps = epoch * iterations_per_epoch
    lrate = initial_lrate * math.pow(gamma, math.floor(steps/stepsize))
    return lrate

# configure callbacks
lrate = LearningRateScheduler(step_decay)
checkpoint = ModelCheckpoint(WEIGHTS_BEST, monitor='loss', verbose=0, save_best_only=False, save_weights_only=True, mode='min', period=1)
csv_logger = CSVLogger(TRAINING_LOG, append=True)
tb = TensorBoard(log_dir=LOGS_DIR, histogram_freq=0, write_graph=True, write_images=False)

callbacks_list = [lrate, checkpoint, csv_logger, tb]

# sgd optimizer with lr multipliers
multisgd = MultiSGD(lr=base_lr, momentum=momentum, decay=0.0, nesterov=False, lr_mult=lr_mult)

# start training
model.compile(loss=losses, optimizer=multisgd, metrics=["accuracy"])

model.fit_generator(train_di,
                    steps_per_epoch=train_samples // batch_size,
                    epochs=max_iter,
                    callbacks=callbacks_list,
                    #validation_data=val_di,
                    #validation_steps=val_samples // batch_size,
                    use_multiprocessing=False,
                    initial_epoch=last_epoch
                    )

