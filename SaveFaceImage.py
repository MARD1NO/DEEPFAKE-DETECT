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

"""
需要先安装paddlhub
将帧数据进行人脸检测并截取

并以9：1的比例划分到训练集和验证集当中

"""


def DetectFace(face_detector_big, imgPath, resolution):
    """
    传入图片路径，进行人脸识别
    resolution是缩放的大小
    返回缩放后的图片
    """
    margin_ratio = 0.02
    input_dict = {"image": [imgPath]}
    results = face_detector_big.face_detection(data=input_dict, confs_threshold=0.45, visualization=False)
    
    results.sort(key=lambda x:x['data'][0]['confidence']) # 按置信度大小排序
    
    faceSituation = results[0]['data'][0]
    left, right, top, bottom = int(faceSituation['left']), int(faceSituation['right']), int(faceSituation['top']), int(faceSituation['bottom']) 
    img = cv2.imread(imgPath)
    
    left = int(max(left*(1-margin_ratio), 0))
    right = int(min(right*(1+margin_ratio), img.shape[1]))
    top = int(max(top*(1-margin_ratio), 0))
    bottom = int(min(bottom*(1+margin_ratio), img.shape[0]))
    
    print("left is{}, right is{}, bottom is{}, top is{}".format(left, right, bottom, top))
    
    face = img[top:bottom, left:right, :]
    face = cv2.resize(face, (resolution, resolution))
    return face

def Saver(face_detector_big, frameImageDir, train_faceImageDir, validate_faceImageDir, threshold=0.9):
    frameImageDirLists = os.listdir(frameImageDir)

    for frameImageDirList in frameImageDirLists:
        frameFullDir = os.path.join(frameImageDir, frameImageDirList)
        randnum = random.randint(0, 100) / 100
        if randnum > threshold:
            # 训练集，验证集分割
            faceImageDir = validate_faceImageDir
        else:
            faceImageDir = train_faceImageDir

        faceFullDir = os.path.join(faceImageDir, frameImageDirList)
        os.mkdir(faceFullDir)
        
        for frameFile in os.listdir(frameFullDir):
            frameFullName = os.path.join(frameFullDir, frameFile)
            faceFullName = os.path.join(faceFullDir, frameFile)
            print("face dir is ", faceFullName)
            try:
                face = DetectFace(face_detector_big, frameFullName, 300)
            except Exception as e:
                continue
            cv2.imwrite(faceFullName, face)
            
if __name__ == "__main__":
    face_detector_big = hub.Module(name="ultra_light_fast_generic_face_detector_1mb_640")
    frameImageDir = '/home/aistudio/work/frame_image_part14'
    trainfaceImageDir = '/home/aistudio/work/train_face14'
    validatefaceImageDir = '/home/aistudio/work/validate_face14'
    Saver(face_detector_big, frameImageDir, trainfaceImageDir, validatefaceImageDir)    