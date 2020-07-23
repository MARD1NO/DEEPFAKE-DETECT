import os
import json

hashlistDir = 'F:/计算机相关论文/DEEPFAKE DETECT/DataProcess/hashlist2.json'

with open(hashlistDir, 'r') as f:
    hashlist = json.load(f)

print(hashlist)

framedata = 'F:/计算机相关论文/DEEPFAKE DETECT/DataProcess/framedata/'

for key in list(hashlist.keys()):
    fake_num = len(hashlist[key])

    for i in range(fake_num):
        framedir = os.path.join(framedata, '{}_{}/'.format(key[:-4], i))
        real_dir = os.path.join(framedir, 'REAL/')
        fake_dir = os.path.join(framedir, 'FAKE/')
        # os.makedirs(real_dir)
        # os.makedirs(fake_dir)

        print(hashlist[key][i])