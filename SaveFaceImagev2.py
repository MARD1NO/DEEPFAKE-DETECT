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
import paddlehub as hub

def DetectFace(face_detector_big, imgPath):
    """
    传入图片路径，进行人脸识别
    resolution是缩放的大小
    返回缩放后的图片
    """
    margin_ratio = 0.045
    input_dict = {"image": [imgPath]}
    face_results = face_detector_big.face_detection(data=input_dict, confs_threshold=0.75, visualization=False)
    # results.sort(key=lambda x:x['data'][0]['left'], reverse=False) # 按置信度大小排序
    # print(face_results)
    results = sorted(face_results[0]['data'], key=lambda x:x['left']) # 根据人脸从右往左排序
    
    # print(results)
    face_nums = len(results)
    # print("Face num is ", face_nums)
    
    face_list = []
    for face_num in range(face_nums):
        faceSituation = results[face_num]
        left, right, top, bottom = int(faceSituation['left']), int(faceSituation['right']), int(faceSituation['top']), int(faceSituation['bottom']) 
        img = cv2.imread(imgPath)
        
        # 稍微扩大框
        left = int(max(left*(1-margin_ratio), 0))
        right = int(min(right*(1+margin_ratio), img.shape[1]))
        top = int(max(top*(1-margin_ratio), 0))
        bottom = int(min(bottom*(1+margin_ratio), img.shape[0]))
    
        print("left is {}, right is {}, bottom is {}, top is {}".format(left, right, bottom, top))
        
        face_list.append(img[top:bottom, left:right, :])
    return face_list

def Saver(face_detector_big, frameImageDir, train_faceImageDir, validate_faceImageDir, threshold=0.9):
    """
    人脸图片保存
    face_detetor_big 人脸检测模型路径
    frameImageDir 帧数据路径 './Frame_data/'
    train_faceImageDir '/train_face/' 训练集保存路径
    validate_faceImageDir '/validate_face/' 验证集保存路径
    threshold 训练集，验证集分割比例
    """
    # filuudleua_0， filuudleua_1
    
    frameImageDirLists = os.listdir(frameImageDir)
    # frameImageDirLists = ['filuudleua_0', 'filuudleua_1', 'hdunuumyxa_0', 'hdunuumyxa_1', 'hdunuumyxa_2']
    
    
    FAKE_REAL = ['FAKE', 'REAL']
    for frameImageDirList in frameImageDirLists:
        # ./Frame_data/filuudleua_0
        frameFullDir = os.path.join(frameImageDir, frameImageDirList)
        
        randnum = random.randint(0, 100) / 100
        if randnum > threshold:
            # 训练集，验证集分割
            faceImageDir = validate_faceImageDir
        else:
            faceImageDir = train_faceImageDir
            
        faceFullDir = os.path.join(faceImageDir, frameImageDirList)
        for fake_or_real in FAKE_REAL:
            # ./Frame_data/filuudleua_0/FAKE  ./Frame_data/filuudleua_0/REAL
            frameFileFullDir = os.path.join(frameFullDir+'/', fake_or_real)
            # ./train_face/filuudleua_0/FAKE  ./train_face/filuudleua_0/REAL
            faceFileFullDir = os.path.join(faceFullDir+'/', fake_or_real)
            
            for frameFile in os.listdir(frameFileFullDir):
                # ./Frame_data/filuudleua_0/FAKE/0_123.jpg
                frameFullName = os.path.join(frameFileFullDir, frameFile)
                try:
                    # 得到人脸矩阵列表
                    facelist = DetectFace(face_detector_big, frameFullName)
                except Exception as e:
                    continue
                # 获取人脸数目
                face_nums = len(facelist)
                for face_num in range(face_nums):
                    # face_0
                    dirname = 'face_{}'.format(face_num) 
                    # ./train_face/filuudleua_0/FAKE/face_0 ./train_face/filuudleua_0/FAKE/face_1    
                    facedirname = os.path.join(faceFileFullDir+'/', dirname)
                    # 创建对应的目录
                    if not os.path.isdir(facedirname):
                        # 如果不存在该目录，则创建目录
                        os.makedirs(facedirname)
                    face = facelist[face_num]
                    faceFullName = os.path.join(facedirname+'/', frameFile)
                    print(faceFullName)
                    cv2.imwrite(faceFullName, face)
                    
if __name__ == '__main__':
    face_detector_big = hub.Module(name="ultra_light_fast_generic_face_detector_1mb_640")
    # frameImageDir = "/home/aistudio/work/Frame_data/filuudleua_0/FAKE/0_123.jpg"
    # face_list = DetectFace(face_detector_big, frameImageDir)
    
    # print(face_list[0].shape)
    # print(type(face_list[0]))
    frameImageDir = '/home/aistudio/work/Frame_data/'
    train_faceImageDir = '/home/aistudio/work/train_face/'
    validate_faceImageDir = '/home/aistudio/work/validate_face/'
    Saver(face_detector_big, frameImageDir, train_faceImageDir, validate_faceImageDir, threshold=0.9)