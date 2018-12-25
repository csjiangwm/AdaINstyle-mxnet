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

import os
import numpy as np
import random
import mxnet as mx
from skimage import io, transform

def postprocess_img(im):
    im[0,:] += 123.68
    im[1,:] += 116.779
    im[2,:] += 103.939
    im = np.swapaxes(im, 0, 2)
    im = np.swapaxes(im, 0, 1)
    im[im<0] = 0
    im[im>255] = 255
    return im.astype(np.uint8)
    
def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret

def crop_img(im, size):
    im = io.imread(im)
    if size !=None:
        if len(im.shape) < 3:
            im = to_rgb(im)
        if im.shape[0]*size[1] > im.shape[1]*size[0]:
            c = (im.shape[0]-1.*im.shape[1]/size[1]*size[0]) / 2
            c = int(c)
            im = im[c:-(1+c),:,:]
        else:
            c = (im.shape[1]-1.*im.shape[0]/size[0]*size[1]) / 2
            c = int(c)
            im = im[:,c:-(1+c),:]
        im = transform.resize(im, size)
        im *= 255
    return im


def preprocess_img(im, size=256):
    if type(size) == int:
        size = (size, size)
    im = crop_img(im, size)
    im = im.astype(np.float32)
    im = np.swapaxes(im, 0, 2)
    im = np.swapaxes(im, 1, 2)
    im[0,:] -= 123.68
    im[1,:] -= 116.779
    im[2,:] -= 103.939
    im = np.expand_dims(im, 0)
    return im

class DataLoader():
    def __init__(self, data_path, batch_size=1, shape=(256, 256), shuffle=False):
        self.batch_size = batch_size
        self.shape = shape
        self.get_images(data_path)
        self.shuffle = shuffle
        self.size = len(self.images)
        self.cur = 0
        self.data = None
        self.reset()
        
    def get_images(self, data_path):
        images = []
        for f in os.listdir(data_path):
            name = f.lower()
            if name.endswith('.png'):
                images.append(os.path.join(data_path, f))
            elif name.endswith('.jpg'):
                images.append(os.path.join(data_path, f))
            elif name.endswith('.jpeg'):
                images.append(os.path.join(data_path, f))
        self.images = images

    def reset(self):
        self.cur = 0
        if self.shuffle:
            random.shuffle(self.images)

    def iter_next(self):
        return self.cur + self.batch_size <= self.size

    def next(self):
        if self.iter_next():
            self.get_batch()
            self.cur += self.batch_size
            return self.data
        else:
            self.reset()
            self.get_batch()
            return self.data

    def get_batch(self):
        cur_from = self.cur
        cur_to = min(cur_from + self.batch_size, self.size)
        data = np.concatenate([preprocess_img(self.images[i], self.shape) for i in range(cur_from, cur_to)])
        self.data = mx.nd.array(data)
        
if __name__ == '__main__':
    data_path = 'images/content/'
    data_loader = DataLoader(data_path, batch_size=2)
    data = data_loader.next()
    print data.shape
