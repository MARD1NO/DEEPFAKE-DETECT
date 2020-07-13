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

"""
训练集读取做了 平滑，jpeg压缩，亮度对比度变化增广
验证集则不做任何增广操作

并且保证同一视频下的各个帧数据所作的增广操作，及系数一致，以防出现视频帧变化过大的情况
"""
def gaussian_blur(img, sigma):
    # 高斯平滑
    gaussian_filter(img[:, :, 0], output=img[:, :, 0], sigma=sigma)
    gaussian_filter(img[:, :, 1], output=img[:, :, 1], sigma=sigma)
    gaussian_filter(img[:, :, 2], output=img[:, :, 2], sigma=sigma)


def cv2_jpg(img, compress_val):
    # jpeg压缩
    img_cv2 = img[:, :, ::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:, :, ::-1]


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
    img = ops[1](img)
    img = ops[2](img)
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


def Image_augumentation(img, resolution, distort_randnum, sig, deepfake_randnum):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转成RGB先

    # print("DISTORT RANDNUM", distort_randnum)
    # print("SIGMA IS ", sig)
    # print("DEEPFAKE RANDNUM IS", deepfake_randnum)

    img = random_distort(img, distort_randnum)
    # img = random_flip(img)
    img = data_augment_deepfake(img, deepfake_randnum, sig)
    img = cv2.resize(img, (resolution, resolution))

    img = Normalize_Img(img)
    return img


def Data_Loaderv2(data_dir, resolution, frame_length, lower=0.6, upper=1.4):
    """
    数据装载器
    :param data_dir: metadata.json 路径
    :param video_dir: 视频文件路径
    :param frame_length:每个文件取多少帧
    :return: reader闭包函数
    yield 标签，帧数据
    """
    # datadir is faceimage/
    framfileDirs = os.listdir(data_dir)
    # 随机打乱文件顺序
    np.random.shuffle(framfileDirs)

    def reader():
        for framefileDir in framfileDirs:
            # frameDir is dhoqofwoxa
            if framefileDir == ".ipynb_checkpoints":
                continue
            perVideoFrameDir = os.path.join(data_dir, framefileDir)  # /faceimage/dhoqofwoxa
            # print(perVideoFrameDir)
            perImageDirList = os.listdir(perVideoFrameDir)
            perImageDirList.sort(key=lambda x: int(x.split('_')[1][:-4]))
            # print(perImageDirList)
            videoArray = np.zeros((frame_length, 3, resolution, resolution), dtype='float32')  # [10, 3, 224, 224]

            cnt = 0

            # 用于设置常规的增广
            distort_randnum = round(uniform(lower, upper), 2)
            # 用于设置deepfake数据增广
            sig = np.random.uniform(0, 3)
            deepfake_randnum = round(uniform(0, 1), 2)

            for perImageDir in perImageDirList:
                if cnt == frame_length:
                    break

                fullImagedir = os.path.join(perVideoFrameDir, perImageDir)
                # print("now it is ", fullImagedir)
                img = cv2.imread(fullImagedir)
                # print(img.shape)
                img = Image_augumentation(img, resolution, distort_randnum, sig, deepfake_randnum)
                # print("img shape is ", img.shape)
                videoArray[cnt] = img
                label = perImageDir.split('_')[0]
                cnt += 1

            if label == '0':
                labelArray = np.array([0])
                # labelArray = np.array([1, 0])
            else:
                labelArray = np.array([1])
                # labelArray = np.array([0, 1])

            yield (videoArray, labelArray)

    def get_data(samples):
        return samples

    mapper = functools.partial(get_data, )

    return paddle.reader.xmap_readers(mapper, reader, 8, 20)


def Data_Loaderv3(data_dir, resolution, frame_length, lower=0.6, upper=1.4, smooth_weight=0.1, use_label_smooth=False):
    """
    数据装载器
    相较于v2增加了label_smooth选项
    :param data_dir: metadata.json 路径
    :param video_dir: 视频文件路径
    :param frame_length:每个文件取多少帧
    :return: reader闭包函数
    yield 标签，帧数据
    """
    # datadir is faceimage/
    framfileDirs = os.listdir(data_dir)
    # 随机打乱文件顺序
    np.random.shuffle(framfileDirs)

    def reader():
        for framefileDir in framfileDirs:
            # frameDir is dhoqofwoxa
            if framefileDir == ".ipynb_checkpoints":
                continue
            perVideoFrameDir = os.path.join(data_dir, framefileDir)  # /faceimage/dhoqofwoxa
            # print(perVideoFrameDir)
            perImageDirList = os.listdir(perVideoFrameDir)
            perImageDirList.sort(key=lambda x: int(x.split('_')[1][:-4]))
            # print(perImageDirList)
            videoArray = np.zeros((frame_length, 3, resolution, resolution), dtype='float32')  # [10, 3, 224, 224]

            cnt = 0

            # 用于设置常规的增广
            distort_randnum = round(uniform(lower, upper), 2)
            # 用于设置deepfake数据增广
            sig = np.random.uniform(0, 3)
            deepfake_randnum = round(uniform(0, 1), 2)

            for perImageDir in perImageDirList:
                if cnt == frame_length:
                    break

                fullImagedir = os.path.join(perVideoFrameDir, perImageDir)
                # print("now it is ", fullImagedir)
                img = cv2.imread(fullImagedir)
                # print(img.shape)
                img = Image_augumentation(img, resolution, distort_randnum, sig, deepfake_randnum)
                # print("img shape is ", img.shape)
                videoArray[cnt] = img
                label = perImageDir.split('_')[0]
                cnt += 1

            if label == '0':
                if not use_label_smooth:
                    labelArray = np.array([0])
                else:
                    labelArray = np.array([1 - smooth_weight, smooth_weight])
            else:
                if not use_label_smooth:
                    labelArray = np.array([1])
                else:
                    labelArray = np.array([smooth_weight, 1 - smooth_weight])

            yield (videoArray, labelArray)

    def get_data(samples):
        return samples

    mapper = functools.partial(get_data, )

    return paddle.reader.xmap_readers(mapper, reader, 8, 20)


def Test_Loaderv3(data_dir, resolution, frame_length, smooth_weight=0.1, use_label_smooth=False):
    """
    测试数据装载器
    不打乱
    相较于v2增加了label_smooth选项
    :param data_dir: metadata.json 路径
    :param video_dir: 视频文件路径
    :param frame_length:每个文件取多少帧
    :return: reader闭包函数
    yield 标签，帧数据
    """
    # datadir is faceimage/
    framfileDirs = os.listdir(data_dir)

    # print(framfileDirs)

    def reader():
        for framefileDir in framfileDirs:
            # frameDir is dhoqofwoxa
            if framefileDir == ".ipynb_checkpoints":
                continue
            perVideoFrameDir = os.path.join(data_dir, framefileDir)  # /faceimage/dhoqofwoxa
            # print(perVideoFrameDir)
            perImageDirList = os.listdir(perVideoFrameDir)
            perImageDirList.sort(key=lambda x: int(x.split('_')[1][:-4]))
            # print(perImageDirList)
            videoArray = np.zeros((frame_length, 3, resolution, resolution), dtype='float32')  # [10, 3, 224, 224]

            cnt = 0

            for perImageDir in perImageDirList:
                # print(perImageDir)
                if cnt == frame_length:
                    break

                fullImagedir = os.path.join(perVideoFrameDir, perImageDir)
                img = cv2.imread(fullImagedir)
                # 验证集只做转RGB 和 resize 归一化处理
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转成RGB先
                img = cv2.resize(img, (resolution, resolution))
                img = Normalize_Img(img)

                videoArray[cnt] = img
                label = perImageDir.split('_')[0]
                cnt += 1

            if label == '0':
                if not use_label_smooth:
                    labelArray = np.array([0])
                else:
                    labelArray = np.array([1 - smooth_weight, smooth_weight])
            else:
                if not use_label_smooth:
                    labelArray = np.array([1])
                else:
                    labelArray = np.array([smooth_weight, 1 - smooth_weight])

            yield (videoArray, labelArray)

    return reader


# def BatchedDataLoaderv2(data_dir, resolution, batchsize, frame_length):
#     train_loader = Data_Loaderv2(data_dir, resolution, frame_length)
#     # video_file_count = len(os.listdir(data_dir))
#     # max_epoch = video_file_count // batchsize
#     def reader():
#         videolist = []
#         labellist = []
#         for data in train_loader():
#             videolist.append(data[0])
#             labellist.append(data[1])
#             if len(labellist) == batchsize:
#                 # print("Success Load Image!")
#                 videoarray = np.array(videolist)
#                 labelarray = np.array(labellist)
#                 yield (videoarray, labelarray)
#                 videolist = []
#                 labellist = []
#     return reader

def BatchedDataLoaderv2(data_dir, resolution, batchsize, frame_length, lower=0.6, upper=1.4):
    train_loader = Data_Loaderv2(data_dir, resolution, frame_length, lower, upper)

    # video_file_count = len(os.listdir(data_dir))
    # max_epoch = video_file_count // batchsize
    def reader():
        videolist = []
        labellist = []
        for data in train_loader():
            videolist.append(data[0])
            labellist.append(data[1])
            if len(labellist) == batchsize:
                # print("Success Load Image!")
                videoarray = np.array(videolist)
                labelarray = np.array(labellist)
                yield (videoarray, labelarray)
                videolist = []
                labellist = []

    def get_data(samples):
        return samples

    mapper = functools.partial(get_data, )

    return paddle.reader.xmap_readers(mapper, reader, 8, 20)


def BatchedDataLoaderv3(data_dir, resolution, batchsize, frame_length, lower=0.6, upper=1.4, smooth_weight=0.1,
                        use_label_smooth=False):
    """
    与dataloaderv3配合使用
    """
    train_loader = Data_Loaderv3(data_dir, resolution, frame_length, lower, upper, smooth_weight, use_label_smooth)

    # video_file_count = len(os.listdir(data_dir))
    # max_epoch = video_file_count // batchsize
    def reader():
        videolist = []
        labellist = []
        for data in train_loader():
            videolist.append(data[0])
            labellist.append(data[1])
            if len(labellist) == batchsize:
                # print("Success Load Image!")
                videoarray = np.array(videolist)
                labelarray = np.array(labellist)
                yield (videoarray, labelarray)
                videolist = []
                labellist = []

    def get_data(samples):
        return samples

    mapper = functools.partial(get_data, )

    return paddle.reader.xmap_readers(mapper, reader, 8, 20)


def BatchedTestLoaderv3(data_dir, resolution, batchsize, frame_length, smooth_weight=0.1, use_label_smooth=False):
    """
    与Test_Loaderv3配合使用
    """
    train_loader = Test_Loaderv3(data_dir, resolution, frame_length, smooth_weight, use_label_smooth)

    # video_file_count = len(os.listdir(data_dir))
    # max_epoch = video_file_count // batchsize
    def reader():
        videolist = []
        labellist = []
        for data in train_loader():
            videolist.append(data[0])
            labellist.append(data[1])
            if len(labellist) == batchsize:
                # print("Success Load Image!")
                videoarray = np.array(videolist)
                labelarray = np.array(labellist)
                yield (videoarray, labelarray)
                videolist = []
                labellist = []

    return reader


if __name__ == "__main__":
    data_dir = '/home/aistudio/work/train_face14'
    lower = 0.6
    upper = 1.4
    reader = BatchedDataLoaderv2(data_dir, 224, 5, 10, lower, upper)
    # reader = Data_Loaderv2(data_dir, 224, 10)
    videoarray, labelarray = next(reader())
    print("video array shape is ", videoarray.shape)
    print("label Array shape is", labelarray.shape)
    print(labelarray[0])
    img = videoarray[0, 1, :, :, :]
    img = img.transpose(2, 1, 0)
    print(img)
    print(img.shape)
    plt.imshow(img)
    plt.show()
