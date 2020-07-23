import shutil
import os
train_face_dir = '/home/aistudio/work/train_face'
validate_face_dir = '/home/aistudio/work/validate_face'

# shutil.rmtree("/home/aistudio/work/train_face/.ipynb_checkpoints")

dirlists = os.listdir(train_face_dir)
for dirlist in dirlists:
    perdir = os.path.join(train_face_dir+'/', dirlist)
    fulldir = os.path.join(perdir+'/', 'REAL')
    if fulldir == '/home/aistudio/work/train_face/.ipynb_checkpoints/FAKE':
        continue
    if len(os.listdir(fulldir)) > 1:
        print(os.listdir(fulldir))
        print(fulldir)
        backdirs = os.listdir(fulldir)
        for backdir in backdirs:
            if backdir == '.ipynb_checkpoints':
                fullbackdir = os.path.join(fulldir+'/', backdir)
                print(fullbackdir)
                # shutil.rmtree(fullbackdir)
                # os.remove(fullbackdir)
    # print(len(os.listdir(fulldir)))
    
print("train face nums is ", len(os.listdir(train_face_dir)))
print("validate face nums is ", len(os.listdir(validate_face_dir)))

