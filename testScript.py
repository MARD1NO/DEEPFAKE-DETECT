import os
import shutil

"""
暂时废弃不用

"""
if __name__ == "__main__":
    """
    删除少于20张图片的文件夹
    """
    frameImageDir = '/home/aistudio/work/face_image_big'
    print("Full cnt is ", len(os.listdir(frameImageDir)))
    frameImageDirLists = os.listdir(frameImageDir)
    cnt = 0

    realcnt = 0
    for frameImageDirList in frameImageDirLists:
        frameFullDir = os.path.join(frameImageDir, frameImageDirList)
        # print(frameFullDir)
        label = frameFullDir.split('_')[-1]
        if label == '1':
            realcnt += 1

        if len(os.listdir(frameFullDir)) < 10:
            print("???")
            print(frameFullDir)

            # cnt += 1
            shutil.rmtree(frameFullDir)
            
    # print(cnt)
    print("REAL CNT is ", realcnt)
    print("Real 占比为", realcnt/len(os.listdir(frameImageDir)))