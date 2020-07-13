import os
import shutil

"""
后续的模型采用20帧

因此需要删去帧数据不足的文件夹
"""

if __name__ == "__main__":
    """
    删除少于20张图片的文件夹
    """
    frameImageDir = '/home/aistudio/work/validate_face'
    print("Full cnt is ", len(os.listdir(frameImageDir)))
    frameImageDirLists = os.listdir(frameImageDir)
    cnt = 0
    framelen = 20
    for frameImageDirList in frameImageDirLists:
        frameFullDir = os.path.join(frameImageDir, frameImageDirList)
        # print(frameFullDir)
        label = frameFullDir.split('_')[-1]

        if len(os.listdir(frameFullDir)) < framelen:
            print("???")
            print(frameFullDir)

            cnt += 1
            shutil.rmtree(frameFullDir)
            
    print("总共有{}条数据".format(len(frameImageDirLists)))
    print("其中人脸图片不到20的有{}".format(cnt))