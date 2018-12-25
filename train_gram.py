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

import time
import random
import os
import mxnet as mx
import numpy as np
np.set_printoptions(precision=2)
import symbol
from dataset import DataLoader

VGGPATH = '/media/jwm/DATA/work/project/neural_style/vgg19.params'
CONTENTPATH = '/media/jwm/DATA/work/data/COCO/train2014'
STYLEPATH = '/media/jwm/DATA/work/data/WikiArt'

def postprocess_img(im):
#     im = im[0]
    im[0,:] += 123.68
    im[1,:] += 116.779
    im[2,:] += 103.939
    im[im<0] = 0
    im[im>255] = 255
    return im
    
def preprocess_img(im):
    im[0,:] -= 123.68
    im[1,:] -= 116.779
    im[2,:] -= 103.939
    return im

def get_mean_var(data):
    mean_data = data.mean(axis=(2,3), keepdims=True)
    var_data = ((data - mean_data)**2).mean(axis=(2,3), keepdims=True)
    return mean_data, var_data
    
def get_sigma(data, eps=1e-5):
    return mx.nd.sqrt(mx.nd.add(data, eps))
    
def adaInstanceNorm(content, style, eps=1e-5):
    mean_content, var_content = get_mean_var(content)
    mean_style, var_style = get_mean_var(style)
    sigma_content = get_sigma(var_content, eps)
    sigma_style = get_sigma(var_style, eps)
    return (content - mean_content) * sigma_style / sigma_content + mean_style
    
def get_gram_executor(out_shapes, weights=[1,1,1,1]):
    gram_executors = []
    for i in range(len(weights)):
        shape = out_shapes[i]
        data = mx.sym.Variable('gram_data')
        data = mx.sym.SliceChannel(data, axis=0, num_outputs=shape[0])
        norms = []
        for j in range(shape[0]):
            flat = mx.sym.Reshape(data[j], shape=(int(shape[1]), int(np.prod(shape[2:]))))
            gram = mx.sym.FullyConnected(flat, flat, no_bias=True, num_hidden=shape[1]) # data shape: batchsize*n_in, weight shape: n_out*n_in
            normed = gram/np.prod(shape[1:])/shape[1]*np.sqrt(weights[i])
            norms.append(mx.sym.expand_dims(normed, axis=0))
        norms_sym = mx.sym.concat(*norms)
        gram_executors.append(norms_sym.bind(ctx=mx.gpu(), args={'gram_data':mx.nd.zeros(shape, mx.gpu())}, args_grad={'gram_data':mx.nd.zeros(shape, mx.gpu())}, grad_req='write'))
    return gram_executors

def get_tv_grad_executor(img, ctx, tv_weight):
    nchannel = img.shape[1]
    simg = mx.sym.Variable("img")
    skernel = mx.sym.Variable("kernel")
    channels = mx.sym.SliceChannel(simg, num_outputs=nchannel)
    out = mx.sym.Concat(*[
        mx.sym.Convolution(data=channels[i], weight=skernel,
                           num_filter=1,
                           kernel=(3, 3), pad=(1,1),
                           no_bias=True, stride=(1,1))
        for i in range(nchannel)])
    kernel = mx.nd.array(np.array([[0, -1, 0],
                                   [-1, 4, -1],
                                   [0, -1, 0]])
                         .reshape((1, 1, 3, 3)),
                         ctx) / 8.0
    out = out * tv_weight
    return out.bind(ctx, args={"img": img, "kernel": kernel})


def init_executor(batch_size, style_layers=['relu1_1','relu2_1','relu3_1','relu4_1'], content_layer='relu4_1'):
    size = 256 
    initializer = mx.init.Xavier(rnd_type='gaussian')
    encoder = symbol.encode(content_layer=content_layer, style_layers=style_layers)
    arg_shapes, output_shapes, aux_shapes = encoder.infer_shape(data=(batch_size, 3, size, size))
    arg_names = encoder.list_arguments()
    arg_dict = dict(zip(arg_names, [mx.nd.zeros(shape, ctx=mx.gpu()) for shape in arg_shapes])) # allocate memory in gpu
    grad_dict = {"data": arg_dict["data"].copyto(mx.gpu())}
    pretrained = mx.nd.load(VGGPATH)
    for name in arg_names:
        if name == "data":
            continue
        key = "arg:" + name
        if key in pretrained:
            pretrained[key].copyto(arg_dict[name])
    encode_executor = encoder.bind(ctx=mx.gpu(), args=arg_dict, args_grad=grad_dict, grad_req='write')
    
    gram_executors = get_gram_executor(output_shapes)
    
    decoder = symbol.decode()
    arg_shapes, output_shapes, aux_shapes = decoder.infer_shape(data=output_shapes[-1]) #content
    arg_names = decoder.list_arguments()
    arg_dict = dict(zip(arg_names, [mx.nd.zeros(shape, ctx=mx.gpu()) for shape in arg_shapes]))
    aux_names = decoder.list_auxiliary_states()
    aux_dict = dict(zip(aux_names, [mx.nd.zeros(shape, ctx=mx.gpu()) for shape in aux_shapes]))
    grad_dict = {}
    for k in arg_dict:
        if k != 'data':
            grad_dict[k] = arg_dict[k].copyto(mx.gpu())
    for name in arg_names:
        if name != 'data':
            initializer(name, arg_dict[name])
    decode_executor = decoder.bind(ctx=mx.gpu(), args=arg_dict, args_grad=grad_dict, aux_states=aux_dict, grad_req='write')
    return encode_executor, decode_executor, gram_executors


def train_style(model_prefix, alpha=0.5, size=256, batch_size=4, tv_weight=1e-4, max_epoch=1000, lr=1e-4,
                style_layer = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1'], content_layer = 'relu4_1'):
    desc_executor, gene_executor, gram_executors = init_executor(batch_size, style_layer, content_layer)
    tv_grad_executor = get_tv_grad_executor(desc_executor.arg_dict['data'], mx.gpu(), tv_weight) 
    optimizer = mx.optimizer.Adam(learning_rate=lr, wd=5e-5)
    optim_states = []
    for i, var in enumerate(gene_executor.grad_dict):
        if var != 'data':
            optim_states.append(optimizer.create_state(i, gene_executor.arg_dict[var])) # allocate memory
        else:
            optim_states.append([])
    target_grams = [mx.nd.zeros(x.outputs[0].shape, mx.gpu()) for x in gram_executors] #[64*64, 128*128, 256*256, 512*512]
    gram_diff = [mx.nd.empty(x.outputs[0].shape, mx.gpu()) for x in gram_executors]
    gram_grad = [mx.nd.empty(x.shape, mx.gpu()) for x in desc_executor.outputs[:len(style_layer)]]
    
    content_content = mx.nd.empty(desc_executor.outputs[-1].shape, mx.gpu())
    style_content = mx.nd.empty(content_content.shape, mx.gpu())
    content_grad = mx.nd.empty(content_content.shape, mx.gpu())
    
#    target_style = [mx.nd.empty(desc_executor.outputs[i].shape, mx.gpu()) for i in range(len(style_layer))]
    style_grad = [mx.nd.empty(x.shape, mx.gpu()) for x in desc_executor.outputs[:len(style_layer)]]
    
    content_loader = DataLoader(CONTENTPATH, batch_size=batch_size)
    style_loader = DataLoader(STYLEPATH, batch_size=batch_size)

    for epoch in range(max_epoch):
        content_loader.next().copyto(desc_executor.arg_dict['data'])
        desc_executor.forward()
        content_content[:] = desc_executor.outputs[-1]
        
        style_loader.next().copyto(desc_executor.arg_dict['data'])
        desc_executor.forward()
        style_content[:] = desc_executor.outputs[-1]
        
        target_content = adaInstanceNorm(content_content, style_content)
        for j in range(len(style_layer)):
            desc_executor.outputs[j].copyto(gram_executors[j].arg_dict['gram_data'])
            gram_executors[j].forward()
            target_grams[j][:] = gram_executors[j].outputs[0]
            target_grams[j][:] /= 1
        
        target_content.copyto(gene_executor.arg_dict['data'])
        gene_executor.forward(is_train=True)
        generate_imgs = [postprocess_img(img.asnumpy()) for img in gene_executor.outputs[0]]
        generate_imgs = [preprocess_img(img) for img in generate_imgs]
        for i in range(batch_size):
            desc_executor.arg_dict['data'][i:i+1] = generate_imgs[i]  #copy numpy to mxnet.ndarray
        tv_grad_executor.forward()
        desc_executor.forward(is_train=True)
        
        loss = [0] * len(desc_executor.outputs)
        for j in range(len(style_layer)):
            desc_executor.outputs[j].copyto(gram_executors[j].arg_dict['gram_data'])
            gram_executors[j].forward(is_train=True)
            gram_diff[j] = gram_executors[j].outputs[0] - target_grams[j]
            gram_executors[j].backward(gram_diff[j])
            gram_grad[j][:] = gram_executors[j].grad_dict['gram_data'] / batch_size
            loss[j] += np.sum(np.square(gram_diff[j].asnumpy())) / batch_size
        
        dec_gen_feat = desc_executor.outputs[-1]
        layer_size = np.prod(dec_gen_feat.shape)
        content_diff = dec_gen_feat - target_content
#        loss[-1] += alpha * np.sum(np.mean(np.square(content_diff.asnumpy()), axis=(2,3))) / batch_size
        loss[-1] += alpha * np.sum(np.square(content_diff.asnumpy()/np.sqrt(layer_size))) / batch_size
        content_grad = alpha * content_diff / batch_size / layer_size
        if epoch % 20 == 0:
            print epoch, 'loss', sum(loss), np.array(loss)
        desc_executor.backward(style_grad+[content_grad])
        gene_executor.backward(desc_executor.grad_dict['data']+tv_grad_executor.outputs[0])
        for i, var in enumerate(gene_executor.grad_dict):
            if var != 'data':
                optimizer.update(i, gene_executor.arg_dict[var], gene_executor.grad_dict[var], optim_states[i]) #update parameter
        if epoch % 500 == 499:
            mx.nd.save('%s_args.nd'%model_prefix, gene_executor.arg_dict)
            mx.nd.save('%s_auxs.nd'%model_prefix, gene_executor.aux_dict)
    mx.nd.save('%s_args.nd'%model_prefix, gene_executor.arg_dict)
    mx.nd.save('%s_auxs.nd'%model_prefix, gene_executor.aux_dict)

if __name__ == '__main__':
    train_style(model_prefix='models/adainstyle_gram', alpha=0.5, max_epoch=200000)
