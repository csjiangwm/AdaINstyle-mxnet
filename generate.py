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
import symbol
from skimage import io, transform
import numpy as np
import dataset

VGGPATH = '/media/jwm/DATA/work/project/neural_style/vgg19.params'
MODELPATH = 'models/adainstyle_gram'

def postprocess_img(im):
    im = im[0]
    im[0,:] += 123.68
    im[1,:] += 116.779
    im[2,:] += 103.939
    im = np.swapaxes(im, 0, 2)
    im = np.swapaxes(im, 0, 1)
    im[im<0] = 0
    im[im>255] = 255
    return im[64:-64,64:-64,:].astype(np.uint8)    
    
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

def stylize(style_layers=['relu1_1','relu2_1','relu3_1','relu4_1'], content_layer='relu4_1', size=512, 
            content_path='images/content/karya.jpg', style_path='images/style/cat.jpg', save_path='images/output.jpg'):
    content_img = dataset.preprocess_img(content_path, size)
    style_img = dataset.preprocess_img(style_path, size)
    
    encoder = symbol.encode(content_layer=content_layer, style_layers=style_layers)
    arg_shapes, output_shapes, aux_shapes = encoder.infer_shape(data=content_img.shape)
    arg_names = encoder.list_arguments()
    arg_dict = dict(zip(arg_names, [mx.nd.zeros(shape, ctx=mx.gpu()) for shape in arg_shapes])) # allocate memory in gpu
    pretrained = mx.nd.load(VGGPATH)
    for name in arg_names:
        if name == "data":
            continue
        key = "arg:" + name
        if key in pretrained:
            pretrained[key].copyto(arg_dict[name])
    encode_executor = encoder.bind(ctx=mx.gpu(), args=arg_dict)
    content_content = mx.nd.zeros(output_shapes[-1], mx.gpu())
    style_content = mx.nd.zeros(output_shapes[-1], mx.gpu())
    target_content = mx.nd.empty(content_content.shape, mx.gpu())
    
    decoder = symbol.decode()
    args = mx.nd.load('%s_args.nd'%MODELPATH)
    auxs = mx.nd.load('%s_auxs.nd'%MODELPATH)
    args['data'] = mx.nd.zeros(output_shapes[-1], mx.gpu())
    gene_executor = decoder.bind(ctx=mx.gpu(), args=args, aux_states=auxs)
    
    encode_executor.arg_dict['data'][:] = content_img
    encode_executor.forward()
    content_content[:] = encode_executor.outputs[-1]
    
    encode_executor.arg_dict['data'][:] = style_img
    encode_executor.forward()
    style_content[:] = encode_executor.outputs[-1]
    
    target_content[:] = adaInstanceNorm(content_content, style_content)
    target_content.copyto(gene_executor.arg_dict['data'])
    gene_executor.forward()
    out = gene_executor.outputs[0].asnumpy()
    im = postprocess_img(out)
    io.imsave(save_path, im)
    print 'done!'
    
if __name__ == '__main__':
    stylize()