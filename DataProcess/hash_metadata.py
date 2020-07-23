import json

metadataDir = 'F:/计算机相关论文/DeepfakeData/metadata.json'
# metadataDir = 'F:/计算机相关论文/DeepFakeCode/deepfakedata/train_sample_videos/metadata.json'

hashlist = {}
with open(metadataDir) as f:
    file = f.read()
    jsonfiles = json.loads(file)
    print(len(jsonfiles))
    for jsonfile in jsonfiles:

        # print(jsonfiles[jsonfile]['label'])
        if jsonfiles[jsonfile]['label'] == 'REAL':
            hashlist[jsonfile] = []
        elif jsonfiles[jsonfile]['label'] == 'FAKE':
            if jsonfiles[jsonfile]['original'] in jsonfiles:
                if jsonfiles[jsonfile]['original'] in hashlist:
                    hashlist[jsonfiles[jsonfile]['original']].append(jsonfile)
                else:
                    hashlist[jsonfiles[jsonfile]['original']] = [jsonfile]
            # else:
            #     hashlist[jsonfiles[jsonfile]['original']] = [jsonfile]

cnt = 0
cnt_len = 450
# cnt_len = 1

for key in list(hashlist.keys()):
    if cnt > cnt_len:
        del hashlist[key]
        continue
    if hashlist[key] == []:
        del hashlist[key]
    else:
        cnt += len(hashlist[key])*2

print(hashlist)
print(len(hashlist))
print(sum([len(hashlist[key]) for key in hashlist.keys()]))

# jsonfileOutputDir = 'F:/计算机相关论文/DEEPFAKE DETECT/DataProcess/hashlist.json'
#
# with open(jsonfileOutputDir, 'w+') as jsonfile:
#     jsonfile.write(json.dumps(hashlist))