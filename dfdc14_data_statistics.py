import os
import shutil
import json

"""
对视频数据做一个统计
"""

if __name__ == "__main__":
    videoDir = '/home/aistudio/work/dfdc_train_part_14'
    video_cnt = len(os.listdir(videoDir))-1
    print("一共有{} 条视频".format(video_cnt))
    label_path = '/home/aistudio/work/dfdc_train_part_14/metadata.json'
    
    true_cnt = 0
    with open(label_path) as f:
        file = f.read()
        jsonfile = json.loads(file)

        for val in jsonfile.values():
            if val['label'] == 'REAL':
                true_cnt += 1
    
    print("Real 视频数量为：",true_cnt)
    print("Fake 视频数量为：", video_cnt - true_cnt)
    print("Real 视频占比为：", true_cnt/video_cnt)
    print("Fake 视频占比为: ", 1 - (true_cnt/video_cnt))


    videoDir = '/home/aistudio/work/train_sample_videos'
    video_cnt = len(os.listdir(videoDir))-1
    print("验证集 一共有{} 条视频".format(video_cnt))
    label_path = '/home/aistudio/work/train_sample_videos/metadata.json'
    true_cnt = 0
    with open(label_path) as f:
        file = f.read()
        jsonfile = json.loads(file)

        for val in jsonfile.values():
            if val['label'] == 'REAL':
                true_cnt += 1
    
    print("Real 视频数量为：",true_cnt)
    print("Fake 视频数量为：", video_cnt - true_cnt)
    print("Real 视频占比为：", true_cnt/video_cnt)
    print("Fake 视频占比为: ", 1 - (true_cnt/video_cnt))