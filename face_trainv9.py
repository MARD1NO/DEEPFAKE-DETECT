import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.regularizer import L2Decay
import numpy as np
from paddle.fluid.dygraph.base import to_variable
from visualdl import LogWriter
import math
from FaceDataLoader import *
from CNNMOGGRUModel import CNNGRUModel

"""
CNN + MogGRU模型
加入了Label Smooth功能
"""
def eval_net(test_reader, model, use_label_smooth=False):
    """
    进行验证集验证
    """
    accs = []
    for i, data in enumerate(test_reader()):
        frame_data, frame_label= data
        
        frame_data = np.array(frame_data).astype('float32')
        if not use_label_smooth:
            # 如果是硬标签则类型为int64
            frame_label = np.array(frame_label).astype('int64')
        else:
            # 如果是软标签则类型为float32
            frame_label = np.array(frame_label).astype('float32')
        
        img = fluid.dygraph.to_variable(frame_data)
        label = fluid.dygraph.to_variable(frame_label)
        label.stop_gradient = True
        output, acc = model(img, label)
        # print(output.numpy())
        accs.append(acc.numpy()[0])
    
    final_acc = np.array(accs).mean()
    print("Eval ACC is:", final_acc)
    return final_acc


if __name__ == "__main__":
    train_video_path = "/home/aistudio/work/train_sample_videos"
    meta_data_path = "/home/aistudio/work/train_sample_videos/metadata.json"
    test_video_path = "/home/aistudio/work/test_videos"
    # threshold = 0.3 # 用于对过多的fake video采样
    face_data_dir = '/home/aistudio/work/train_face14'
    # face_data_dir = '/home/aistudio/work/validate_face14'
    

    val_face_data_dir = '/home/aistudio/work/validate_face14'
    
    # 日志记录可视化
    # trainloss_writer = LogWriter("./log/trainloss")
    
    resolution = 224
    batchsize = 14
    frame_length = 20
    base_lr = 0.0001
    use_label_smooth = True
    
    with fluid.dygraph.guard():
        model = CNNGRUModel(use_label_smooth=True)
        
        opt = fluid.optimizer.AdamOptimizer(learning_rate=fluid.dygraph.ExponentialDecay(
              learning_rate=base_lr,
              decay_steps=2000,
              decay_rate=0.1,
              staircase=True),  regularization=fluid.regularizer.L2Decay(0.00005),  parameter_list=model.parameters())  #创建优化器
        
        # opt = fluid.optimizer.AdamOptimizer(learning_rate=base_lr,  regularization=fluid.regularizer.L2Decay(0.00005),  parameter_list=model.parameters())  #创建优化器

        # 训练数据读取器
        train_loader = BatchedDataLoaderv3(face_data_dir, resolution, batchsize, frame_length, use_label_smooth=use_label_smooth)
        # 验证数据读取器，不打乱顺序
        validate_loader = BatchedTestLoaderv3(val_face_data_dir, resolution, 1, frame_length, use_label_smooth=use_label_smooth)
        # 初始验证集准确率设置为0
        val_acc = 0

        video_file_count = len(os.listdir(face_data_dir))
        max_per_epoch = video_file_count // batchsize
        print("max per epoch is", max_per_epoch)
        MAX_EPOCH = 400

        for epoch in range(MAX_EPOCH):
            smallepoch = 1
            total_train_video_sample = 0
            accurate_count = 0
            total_loss = 0
            total_acc = 0
            for i, data in enumerate(train_loader()):
                if smallepoch > max_per_epoch - 1:
                    break
                
                frame_data, frame_label= data
                
                # 转化为合适的数据类型
                frame_data = np.array(frame_data).astype('float32')
                if not use_label_smooth:
                    # 如果是硬标签则类型为int64
                    frame_label = np.array(frame_label).astype('int64')
                else:
                    # 如果是软标签则类型为float32
                    frame_label = np.array(frame_label).astype('float32')
            
                        
                img = fluid.dygraph.to_variable(frame_data, name="img")
                label = fluid.dygraph.to_variable(frame_label, name="label")
                label.stop_gradient = True 
                output, acc = model(img, label)
                # 计算交叉熵loss，可选labelsmooth选项
                loss = fluid.layers.cross_entropy(output, label, soft_label=use_label_smooth)
                mean_loss = fluid.layers.reduce_mean(loss)
                mean_loss.backward()
                total_loss += mean_loss
                opt.minimize(mean_loss)
                model.clear_gradients()
                total_acc += acc.numpy()[0]
                
                smallepoch += 1
                total_train_video_sample += batchsize
            
            print("EPOCH ", epoch)
            print("total loss is ", sum(total_loss.numpy()))
            print("Accuracy is ", total_acc/smallepoch)
            
            # trainloss_writer.add_scalar(tag='trainloss', step=epoch, value=total_loss.numpy())
            
            if epoch % 20 == 0:
                model.eval()
                cur_val_acc = eval_net(validate_loader, model, use_label_smooth)
                if cur_val_acc > val_acc:
                    print("validation at epoch {} current acc {} is better than last acc {}".format(epoch, cur_val_acc, val_acc))
                    val_acc = cur_val_acc

            model.train()
            if epoch % 50 == 0 or epoch == MAX_EPOCH-1:
                fluid.save_dygraph(model.state_dict(), 'DpDcModel_BiGRU_epoch{}'.format(epoch))
            
          