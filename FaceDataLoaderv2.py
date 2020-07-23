import cv2
import matplotlib.pyplot as plt
import paddle
import numpy as np
from PIL import Image, ImageEnhance
import random
import os
import json
from typing import *
import functools
from matplotlib.image import imread
from random import random, randint, uniform
from scipy.ndimage.filters import gaussian_filter

def gaussian_blur(img, sigma):
    # 高斯平滑
    gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
    gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
    gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)

def cv2_jpg(img, compress_val):
    # jpeg压缩
    img_cv2 = img[:,:,::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:,:,::-1]

def data_augment_deepfake(img, randnum, sigma):
    # print("NOW RANDNUM IS ", randnum)
    # 根据论文设定的相关数据增广方法
    img = np.array(img)
    blur_prob = 0.1
    jpg_prob = 0.1
    
    if randnum < blur_prob:
        # sig = np.random.uniform(0, 3)
        gaussian_blur(img, sigma)

    if randnum < jpg_prob:
        img = cv2_jpg(img, 75)
    return img

def random_distort(img, randnum):
    """
    随机改变亮度，对比度，颜色
    :param img:
    :param randnum:随机数，在0.6-1.4这个范围内
    :return:
    """

    # 随机改变亮度
    def random_brightness(img):
        
        return ImageEnhance.Brightness(img).enhance(randnum)

    # 随机改变对比度
    def random_contrast(img):

        return ImageEnhance.Contrast(img).enhance(randnum)

    # 随机改变颜色
    def random_color(img):
        
        return ImageEnhance.Color(img).enhance(randnum)

    ops = [random_brightness, random_contrast, random_color]
    # np.random.shuffle(ops)

    img = Image.fromarray(img)
    img = ops[0](img)
    # img = ops[1](img)
    # img = ops[2](img)
    img = np.asarray(img)

    return img

def random_flip(img, thresh=0.5):
    randnum = randint(0, 100) / 100
    if randnum > thresh:
        img = img[:, ::-1, :]
        
    return img


def Normalize_Img(img):
    """
    对图片数据做归一化
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    mean = np.array(mean).reshape((1, 1, -1))
    std = np.array(std).reshape((1, 1, -1))
    img = (img / 255.0 - mean) / std
    img = img.astype('float32').transpose((2, 0, 1))
    return img
 
def Image_augumentation(img, resolution, use_augumentation, distort_randnum, sig, deepfake_randnum):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # 转成RGB先
    if not use_augumentation:
        # 如果不使用augumentation，则都置为0
        distort_randnum, sig, deepfake_randnum = 0, 0, 0
        
    # print("DISTORT RANDNUM", distort_randnum)
    # print("SIGMA IS ", sig)
    # print("DEEPFAKE RANDNUM IS", deepfake_randnum)
    
    if use_augumentation:
        img = random_distort(img, distort_randnum)
        # img = random_flip(img)
        img = data_augment_deepfake(img, deepfake_randnum, sig)
    
    img = cv2.resize(img, (resolution, resolution))
    img = Normalize_Img(img)
    
    return img
    
def TrainDataLoader(data_dir, resolution, frame_length, lower=0.8, upper=1.2, smooth_weight=0.1, use_label_smooth=False, use_augumentation=True):
    """
    数据装载器
    相较于v2增加了label_smooth选项
    :param data_dir: metadata.json 路径
    :param video_dir: 视频文件路径
    :param frame_length:每个文件取多少帧
    :return: reader闭包函数
    yield 标签，帧数据
    """
    # datadir is ./train_face/
    framfileDirs = os.listdir(data_dir)
    # kzrjkklsin_7 xpgnlphhug_1
    
    # 随机打乱文件顺序
    np.random.shuffle(framfileDirs)
    def reader():
        for framefileDir in framfileDirs:
            # frameDir is dhoqofwoxa
            if framefileDir == ".ipynb_checkpoints":
                continue
                
            # /faceimage/dhoqofwoxa    
            perVideoFrameDir = os.path.join(data_dir, framefileDir)
            # print(perVideoFrameDir)
            FAKE_OR_REAL = ['FAKE', 'REAL']
            
            FAKE_OR_REAL_NUM = 0 # 每个文件夹都有FAKE和REAL，以这个变量计数
            videoArray = np.zeros((2, frame_length, 3, resolution, resolution), dtype='float32') # [2, 20, 3, 224, 224]
            if use_label_smooth:
                # [[0.1, 0.9], 
                #  [0.9, 0.1]]
                labelArray = np.zeros((2, 2), dtype='float32')
            else:
                # [[1], 
                #  [0]]
                labelArray = np.zeros((2, 1), dtype='int32')
            
            # 用于设置常规的增广
            distort_randnum = round(uniform(lower, upper), 2)
            # 用于设置deepfake数据增广
            sig = np.random.uniform(0, 3)
            deepfake_randnum = round(uniform(0, 1), 2)
                
            for fake_or_real in FAKE_OR_REAL:
                
                # /faceimage/dhoqofwoxa/REAL  /faceimage/dhoqofwoxa/FAKE   
                perLabelDir = os.path.join(perVideoFrameDir+'/', fake_or_real)
            
                perFaceDirList = os.listdir(perLabelDir)
                # face_0
                    
                for perFaceDir in perFaceDirList:
                    # /faceimage/dhoqofwoxa/REAL/face_0
                    fullFaceDir = os.path.join(perLabelDir+'/', perFaceDir)
                    perImageDirList = os.listdir(fullFaceDir)
                    # 0_79.jpg 0_103.jpg
                    perImageDirList.sort(key=lambda x: int(x.split('_')[1][:-4])) # 按文件名后面的帧序号排序
                    
                    frame_cnt = 0
                    for perImage in perImageDirList:
                        if frame_cnt == frame_length:
                            # 如果读取的数据读满至指定的frame_length
                            break
                        # /faceimage/dhoqofwoxa/REAL/face_0/0_79.jpg
                        fullImageDir = os.path.join(fullFaceDir+'/', perImage)
                        # print(fullImageDir)
                        # 使用opencv读取数据
                        img = cv2.imread(fullImageDir)
                        img = Image_augumentation(img, resolution, use_augumentation, distort_randnum, sig, deepfake_randnum)
                        
                        videoArray[FAKE_OR_REAL_NUM][frame_cnt] = img    
                        frame_cnt += 1
                        
                if fake_or_real == 'FAKE':
                    # FAKE标签                    
                    if use_label_smooth:
                        # [[0.1, 0.9], 
                        #  [0.9, 0.1]]
                        labelArray[FAKE_OR_REAL_NUM] = [1-smooth_weight, smooth_weight]
                    else:
                        labelArray[FAKE_OR_REAL_NUM] = [0]
                        
                else:
                    # REAL标签
                    if use_label_smooth:
                        # [[0.1, 0.9], 
                        #  [0.9, 0.1]]
                        labelArray[FAKE_OR_REAL_NUM] = [smooth_weight, 1-smooth_weight]
                    else:
                        labelArray[FAKE_OR_REAL_NUM] = [1]
                        
                FAKE_OR_REAL_NUM += 1
            yield (videoArray, labelArray)    
            
    return reader

def BatchedTrainDataLoader(data_dir, resolution, batchsize, frame_length, lower=0.85, upper=1.15, smooth_weight=0.1, use_label_smooth=False, use_augumentation=True):
    """
    与dataloader配合使用
    """
    train_loader = TrainDataLoader(data_dir, resolution, frame_length, lower, upper, smooth_weight, use_label_smooth, use_augumentation)
    
    def reader():
        videolist = []
        labellist = []
        for data in train_loader():
            videolist.append(data[0])
            labellist.append(data[1])
            if len(labellist) == batchsize:
                # print("Success Load Image!")
                videoarray = np.concatenate(videolist, axis=0)
                labelarray = np.concatenate(labellist, axis=0)
                yield (videoarray, labelarray)
                videolist = []
                labellist = []
                
    # def get_data(samples):
    #     return samples

    # mapper = functools.partial(get_data, )

    # return paddle.reader.xmap_readers(mapper, reader, 8, 20)
    return reader

def BatchedTestDataLoader(data_dir, resolution, batchsize, frame_length, lower=0.85, upper=1.15, smooth_weight=0.1, use_label_smooth=False, use_augumentation=False):
    """
    与dataloader配合使用
    """
    test_loader = DataLoader(data_dir, resolution, frame_length, lower, upper, smooth_weight, use_label_smooth, use_augumentation)
    
    def reader():
        videolist = []
        labellist = []
        for data in test_loader():
            videolist.append(data[0])
            labellist.append(data[1])
            if len(labellist) == batchsize:
                # print("Success Load Image!")
                videoarray = np.concatenate(videolist, axis=0)
                labelarray = np.concatenate(labellist, axis=0)
                yield (videoarray, labelarray)
                videolist = []
                labellist = []
                
    # def get_data(samples):
    #     return samples

    # mapper = functools.partial(get_data, )

    # return paddle.reader.xmap_readers(mapper, reader, 8, 20)
    return reader
  
if __name__ == "__main__":
    data_dir = '/home/aistudio/work/train_face'
    lower = 0.6
    upper = 1.4
    # reader = DataLoader(data_dir, 224, 10, lower, upper)
    reader = BatchedTrainDataLoader(data_dir, 224, 4, 5, lower, upper, use_label_smooth=True, use_augumentation=False)
    
    videoarray, labelarray = next(reader())
    print("video array shape is ", videoarray.shape)
    print("label Array shape is", labelarray.shape)
    print(labelarray)
    