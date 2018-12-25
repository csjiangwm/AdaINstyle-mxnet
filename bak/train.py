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
import symbol
from dataset import DataLoader

VGGPATH = '/media/jwm/DATA/work/project/neural_style/vgg19.params'
CONTENTPATH = '/media/jwm/DATA/work/data/COCO/train2014'
STYLEPATH = 'images/style'

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
    
def get_style_loss_executor(output_shapes, ctx, eps=1e-5):
    styleloss_executors = []
    for shape in output_shapes:
        gen = mx.sym.Variable('gen')
        style = mx.sym.Variable('style')
        meanS = mx.sym.mean(data=style, axis=(2,3), keepdims=True)
        diffS = mx.sym.broadcast_sub(style, meanS)
        squareS = mx.sym.square(data=diffS)
        varS = mx.sym.mean(squareS, axis=(2,3), keepdims=True)
        sigmaS = mx.sym.sqrt(data=varS+eps)
        
        meanG = mx.sym.mean(data=gen, axis=(2,3), keepdims=True)
        diffG = mx.sym.broadcast_sub(gen, meanG)
        squareG = mx.sym.square(data=diffG)
        varG = mx.sym.mean(squareG, axis=(2,3), keepdims=True)
        sigmaG = mx.sym.sqrt(data=varG+eps)
        
        mean_diff = mx.sym.square(data=meanG-meanS)
        l2_mean = mx.sym.sum(data=mean_diff)
        
        sigma_diff = mx.sym.square(data=sigmaG-sigmaS)
        l2_sigma = mx.sym.sum(data=sigma_diff)
        
        loss = mx.sym.sum(data=l2_mean + l2_sigma)
        styleloss_executor = loss.bind(ctx, args={'gen':mx.nd.zeros(shape, ctx), 'style':mx.nd.zeros(shape, ctx)},
                                            args_grad={'gen':mx.nd.zeros(shape, ctx)}, 
                                            grad_req='write')
        styleloss_executors.append(styleloss_executor)
    return styleloss_executors


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
#    initializer = mx.init.Normal()
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
    
    styleloss_executors = get_style_loss_executor(output_shapes[:len(style_layers)], ctx=mx.gpu())
    
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
    return encode_executor, decode_executor, styleloss_executors


def train_style(model_prefix, alpha=0.5, size=256, batch_size=4, tv_weight=1e-4, max_epoch=1000, lr=1e-4,
                style_layer = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1'], content_layer = 'relu4_1'):
    desc_executor, gene_executor, styleloss_executors = init_executor(batch_size, style_layer, content_layer)
#    tv_grad_executor = get_tv_grad_executor(desc_executor.arg_dict['data'], mx.gpu(), tv_weight) 
    optimizer = mx.optimizer.Adam(learning_rate=lr, wd=5e-5)
    optim_states = []
    for i, var in enumerate(gene_executor.grad_dict):
        if var != 'data':
            optim_states.append(optimizer.create_state(i, gene_executor.arg_dict[var])) # allocate memory
        else:
            optim_states.append([])
    
    content_content = mx.nd.empty(desc_executor.outputs[-1].shape, mx.gpu())
    style_content = mx.nd.empty(content_content.shape, mx.gpu())
    content_grad = mx.nd.empty(content_content.shape, mx.gpu())
    target_content = mx.nd.empty(content_content.shape, mx.gpu())
    
    target_style = [mx.nd.empty(desc_executor.outputs[i].shape, mx.gpu()) for i in range(len(style_layer))]
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
        for i in range(len(style_layer)):
            target_style[i][:] = desc_executor.outputs[i]
        target_content[:] = adaInstanceNorm(content_content, style_content)
        
        target_content.copyto(gene_executor.arg_dict['data'])
        gene_executor.forward(is_train=True)
        generate_imgs = [postprocess_img(img.asnumpy()) for img in gene_executor.outputs[0]]
        generate_imgs = [preprocess_img(img) for img in generate_imgs]
        for i in range(batch_size):
            desc_executor.arg_dict['data'][i:i+1] = generate_imgs[i]  #copy numpy to mxnet.ndarray
#        tv_grad_executor.forward()
        desc_executor.forward(is_train=True)
        
        loss = [0] * len(desc_executor.outputs)
        for j in range(len(style_layer)):
            target_style[j].copyto(styleloss_executors[j].arg_dict['style'])
            desc_executor.outputs[j].copyto(styleloss_executors[j].arg_dict['gen'])
            styleloss_executors[j].forward(is_train=True)
            loss[j] += styleloss_executors[j].outputs[0].asnumpy()[0] / batch_size
            styleloss_executors[j].backward(styleloss_executors[j].outputs[0])
            style_grad[j] = styleloss_executors[j].grad_dict['gen'] / batch_size
        
        dec_gen_feat = desc_executor.outputs[-1]
        layer_size = np.prod(dec_gen_feat.shape)
        content_diff = dec_gen_feat - target_content
        loss[-1] += alpha * np.sum(np.square(content_diff.asnumpy()/np.sqrt(layer_size))) / batch_size
        content_grad = alpha * content_diff / batch_size / layer_size
        if epoch % 20 == 0:
            print epoch, 'loss', sum(loss), np.array(loss)
        desc_executor.backward(style_grad+[content_grad])
#        gene_executor.backward(desc_executor.grad_dict['data']+tv_grad_executor.outputs[0])
        gene_executor.backward(desc_executor.grad_dict['data'])
        for i, var in enumerate(gene_executor.grad_dict):
            if var != 'data':
                optimizer.update(i, gene_executor.arg_dict[var], gene_executor.grad_dict[var], optim_states[i]) #update parameter
        if epoch % 500 == 499:
            mx.nd.save('%s_args.nd'%model_prefix, gene_executor.arg_dict)
            mx.nd.save('%s_auxs.nd'%model_prefix, gene_executor.aux_dict)
    mx.nd.save('%s_args.nd'%model_prefix, gene_executor.arg_dict)
    mx.nd.save('%s_auxs.nd'%model_prefix, gene_executor.aux_dict)

if __name__ == '__main__':
    train_style(model_prefix='models/adainstyle', alpha=0.5, max_epoch=2000000)
    