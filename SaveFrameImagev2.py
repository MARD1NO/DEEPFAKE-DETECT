import cv2
import numpy as np
import random
import os
import json
from typing import *

def CaptureVideoImage(
        videoFile: str,
        savedir: str, 
        label:int,
        totalFrame=25):
    """
    截取视频帧
    :param videoFile: 视频文件
    """
    # print(videoFile)
    vidcap = cv2.VideoCapture(videoFile)  # 视频流截图
    # print(vidcap.isOpened())
    frame_all = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))  # 总帧数
    frame_start = random.randint(0, frame_all // 2)  # 起始帧

    frame_interval = 2

    if vidcap.isOpened():
        for i in range(frame_start, frame_start + totalFrame * frame_interval, frame_interval):
            vidcap.set(cv2.CAP_PROP_POS_FRAMES, i)  # set方法获取指定帧
            success, img = vidcap.read()
            if success:
                img_save_dir = savedir+"{}_{}.jpg".format(label, i)
                print(img_save_dir)
                cv2.imwrite(img_save_dir, img)
                
def SaveFrameImage(hashlistDir, saveDir, videoDir, totalFrame=25):
    """
    保存帧图片
    哈希表里存储的是 
    {'REAL视频Dir':['FAKE视频Dir1', 'FAKE视频Dir2'......]}
    
    最终文件目录如下
    REAL视频Dir_{index}
        |FAKE
            |FAKE视频帧
        |REAL
            |REAL视频帧
    """
    with open(hashlistDir, 'r') as f:
        hashlist = json.load(f)
    
    for key in list(hashlist.keys()):
        fake_num = len(hashlist[key])

        for i in range(fake_num):
            # REAL视频Dir_{index}/
            framedir = os.path.join(saveDir, '{}_{}/'.format(key[:-4], i))
            # print("framedir is", framedir)
            # REAL视频Dir_{index}/REAL/
            real_dir = os.path.join(framedir, 'REAL/')
            # print("realdir is", real_dir)
            
            # FAKE视频Dir_{index}/FAKE/
            fake_dir = os.path.join(framedir, 'FAKE/')
            # print("fakedir is", fake_dir)
            
            os.makedirs(real_dir)
            os.makedirs(fake_dir)
            
            realVideoDir = os.path.join(videoDir, key)
            fakeVideoDir = os.path.join(videoDir, hashlist[key][i])
            # print("realVideoDir is", realVideoDir)
            # print("fakeVideoDir is", fakeVideoDir)
            
            # REAL的label设置为1
            video_frame_data = CaptureVideoImage(realVideoDir, real_dir, 1, totalFrame)
            # FAKE的label设置为0
            video_frame_data = CaptureVideoImage(fakeVideoDir, fake_dir, 0, totalFrame)
            
if __name__ == "__main__":
    hashlistDir = '/home/aistudio/work/video_hashlist.json'
    saveDir = '/home/aistudio/work/Frame_data/'
    videoDir = '/home/aistudio/work/dfdc_train_part_14/'
    totalFrame = 25
    SaveFrameImage(hashlistDir, saveDir, videoDir, totalFrame)
    