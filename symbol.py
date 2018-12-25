#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""                                   ————————————————————————————————————————————
               ,%%%%%%%%,             Created Time:  -- --th 2018
             ,%%/\%%%%/\%%            Revised Time:  -- --th 2018 
            ,%%%\c "" J/%%%           Contact     :  ------------
   %.       %%%%/ o  o \%%%           Environment :  python2.7
   `%%.     %%%%    _  |%%%           Description :  /
    `%%     `%%%%(__Y__)%%'           Author      :  ___           __ ____   ____
    //       ;%%%%`\-/%%%'                          | | \   ____  / /| |\ \ | |\ \
   ((       /  `%%%%%%%'                            | |\ \ / / | / / | |\ \ | |\ \
    \\    .'          |                             | | \ V /| |/ /  | |\ \| | \ \
     \\  /       \  | |                           _ | |  \_/ |_|_/   |_|\_\|_| \_\
      \\/         ) | |                          \\ | |
       \         /_ | |__                         \ | |
       (___________))))))) 攻城湿                  \_| 
                                     ——————————————————————————————————————————————
"""

import mxnet as mx
import numpy as np

def conv2d(data, num_filter, name):
    conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=(3,3), pad=(1,1), name='conv'+name)
    relu = mx.sym.Activation(data=conv, act_type='relu', name='relu'+name)
    return relu

def decode():
    data = mx.sym.Variable('data')
    data = conv2d(data, 256, '4_1')
    data = mx.sym.UpSampling(data, scale=2, sample_type='nearest')
    data = conv2d(data, 256, '3_4')
    data = conv2d(data, 256, '3_3')
    data = conv2d(data, 256, '3_2')
    data = conv2d(data, 128, '3_1')
    data = mx.sym.UpSampling(data, scale=2, sample_type='nearest')
    data = conv2d(data, 128, '2_2')
    data = conv2d(data, 64, '2_1')
    data = mx.sym.UpSampling(data, scale=2, sample_type='nearest')
    data = conv2d(data, 64, '1_2')
    data = mx.sym.Convolution(data=data, num_filter=3, kernel=(3,3), pad=(1,1), name='conv1_1')
    return data 


def encode(style_layers=['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1'], content_layer='relu4_2'):
    data = mx.symbol.Variable('data')
    conv1_1 = mx.symbol.Convolution(name='conv1_1', data=data , num_filter=64, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False, workspace=1024)
    relu1_1 = mx.symbol.Activation(name='relu1_1', data=conv1_1 , act_type='relu')
    conv1_2 = mx.symbol.Convolution(name='conv1_2', data=relu1_1 , num_filter=64, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False, workspace=1024)
    relu1_2 = mx.symbol.Activation(name='relu1_2', data=conv1_2 , act_type='relu')
    pool1 = mx.symbol.Pooling(name='pool1', data=relu1_2 , pad=(0,0), kernel=(2,2), stride=(2,2), pool_type='avg')
    conv2_1 = mx.symbol.Convolution(name='conv2_1', data=pool1 , num_filter=128, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False, workspace=1024)
    relu2_1 = mx.symbol.Activation(name='relu2_1', data=conv2_1 , act_type='relu')
    conv2_2 = mx.symbol.Convolution(name='conv2_2', data=relu2_1 , num_filter=128, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False, workspace=1024)
    relu2_2 = mx.symbol.Activation(name='relu2_2', data=conv2_2 , act_type='relu')
    pool2 = mx.symbol.Pooling(name='pool2', data=relu2_2 , pad=(0,0), kernel=(2,2), stride=(2,2), pool_type='avg')
    conv3_1 = mx.symbol.Convolution(name='conv3_1', data=pool2 , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False, workspace=1024)
    relu3_1 = mx.symbol.Activation(name='relu3_1', data=conv3_1 , act_type='relu')
    conv3_2 = mx.symbol.Convolution(name='conv3_2', data=relu3_1 , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False, workspace=1024)
    relu3_2 = mx.symbol.Activation(name='relu3_2', data=conv3_2 , act_type='relu')
    conv3_3 = mx.symbol.Convolution(name='conv3_3', data=relu3_2 , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False, workspace=1024)
    relu3_3 = mx.symbol.Activation(name='relu3_3', data=conv3_3 , act_type='relu')
    conv3_4 = mx.symbol.Convolution(name='conv3_4', data=relu3_3 , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False, workspace=1024)
    relu3_4 = mx.symbol.Activation(name='relu3_4', data=conv3_4 , act_type='relu')
    pool3 = mx.symbol.Pooling(name='pool3', data=relu3_4 , pad=(0,0), kernel=(2,2), stride=(2,2), pool_type='avg')
    conv4_1 = mx.symbol.Convolution(name='conv4_1', data=pool3 , num_filter=512, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False, workspace=1024)
    relu4_1 = mx.symbol.Activation(name='relu4_1', data=conv4_1 , act_type='relu')
    conv4_2 = mx.symbol.Convolution(name='conv4_2', data=relu4_1 , num_filter=512, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False, workspace=1024)
    relu4_2 = mx.symbol.Activation(name='relu4_2', data=conv4_2 , act_type='relu')
    conv4_3 = mx.symbol.Convolution(name='conv4_3', data=relu4_2 , num_filter=512, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False, workspace=1024)
    relu4_3 = mx.symbol.Activation(name='relu4_3', data=conv4_3 , act_type='relu')
    conv4_4 = mx.symbol.Convolution(name='conv4_4', data=relu4_3 , num_filter=512, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False, workspace=1024)
    relu4_4 = mx.symbol.Activation(name='relu4_4', data=conv4_4 , act_type='relu')
    pool4 = mx.symbol.Pooling(name='pool4', data=relu4_4 , pad=(0,0), kernel=(2,2), stride=(2,2), pool_type='avg')
    conv5_1 = mx.symbol.Convolution(name='conv5_1', data=pool4 , num_filter=512, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False, workspace=1024)
    relu5_1 = mx.symbol.Activation(name='relu5_1', data=conv5_1 , act_type='relu')
    style_out = mx.sym.Group([x for x in map(eval, style_layers)])
    return mx.sym.Group([style_out, eval(content_layer)])


if __name__ == '__main__':
    feat_sym = decode()
    arg_shape, out_shape, aux_shape = feat_sym.infer_shape(data=(1,512,32,32))
    print out_shape
    feat_sym = encode(style_layers=['relu1_1','relu2_1','relu3_1','relu4_1'], content_layer='relu4_1')
    arg_shapes, out_shapes, aux_shapes = feat_sym.infer_shape(data=(1,3,256,256))
    print out_shapes
    
