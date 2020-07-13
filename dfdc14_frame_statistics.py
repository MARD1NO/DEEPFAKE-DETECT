import os

"""
对筛选过后的帧数据做一个统计
"""
videoDir = '/home/aistudio/work/train_face14'

videoDirlist = os.listdir(videoDir)

realcnt = 0
fakecnt = 1
for video in videoDirlist:
    label = video.split('_')[-1]
    if label == '1':
        realcnt += 1
    else:
        fakecnt += 1

print("REAL 帧一共有{}条".format(realcnt))
print("FAKE 帧一共有{}条".format(fakecnt))
