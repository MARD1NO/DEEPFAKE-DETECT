import os

"""
用于对验证帧数据的统计

暂时废弃不用
"""
if __name__ =="__main__":
    framedir = "/home/aistudio/work/frameimage"
    dirlist = os.listdir(framedir)
    
    fakenum = 0
    realnum = 0
    
    # print(dirlist[0].split('_')[2])

    for dir in dirlist:
        # print(dir)
        try:
            a = dir.split('_')[2]
            if dir.split('_')[2] == '0':
                fakenum += 1
            else:
                realnum += 1
        except:
            print(dir)
        
    
    print("FAKE IMAGE COUNT ", fakenum)
    print("REAL IMAGE COUNT ", realnum)
