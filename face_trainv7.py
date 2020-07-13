import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.regularizer import L2Decay
import numpy as np
from paddle.fluid.dygraph.base import to_variable
import math
from FaceDataLoader import *
from CNN_RNN_3D_modelv2 import CNN_RNN_Model4, Resnet3dSub

"""
CNN 双向LSTM + Resnet3D帧间差模型
添加了训练过程中的评估程序
添加了双路网络分配的权重

可以在weight3D参数那里设置，默认为0.5:0.5

"""


def eval_net(test_reader, model, model3d, weight3d=0.5):
    """
    进行验证集验证
    """
    accs = []

    binary_tensor1 = np.array([[(1-weight3d) for i in range(2)] for i in range(1)]).astype("float32")
    binary_tensor1 = fluid.dygraph.to_variable(binary_tensor1)
    binary_tensor1.stop_gradient = True

    binary_tensor2 = np.array([[weight3d for i in range(2)] for i in range(1)]).astype("float32")
    binary_tensor2 = fluid.dygraph.to_variable(binary_tensor2)
    binary_tensor2.stop_gradient = True

    for i, data in enumerate(test_reader()):
        frame_data, frame_label= data
        
        frame_data = np.array(frame_data).astype('float32')
        frame_label = np.array(frame_label).astype('int64')
        
        img = fluid.dygraph.to_variable(frame_data)
        label = fluid.dygraph.to_variable(frame_label)
        label.stop_gradient = True
        output1, acc1 = model(img, label)
        output3d, acc3d = model3d(img, label)

        final_output = fluid.layers.elementwise_mul(binary_tensor1, output1) + fluid.layers.elementwise_mul(binary_tensor2, output3d)
        acc = fluid.layers.accuracy(final_output, label)

        
        accs.append(acc.numpy()[0])
    
    final_acc = np.array(accs).mean()
    print("Eval ACC is:", final_acc)
    return final_acc


if __name__ == "__main__":
    train_video_path = "/home/aistudio/work/train_sample_videos"
    meta_data_path = "/home/aistudio/work/train_sample_videos/metadata.json"
    test_video_path = "/home/aistudio/work/test_videos"
    face_data_dir = '/home/aistudio/work/train_face14'
    val_face_data_dir = '/home/aistudio/work/validate_face14'
    
    resolution = 224
    batchsize = 4
    frame_length = 20
    base_lr = 0.0001
    base_lr_3d = 0.003

    # 双路网络最终输出预测的分配权重，weight_3d指代的是3D网络的权重
    weight_3d = 0.5

    with fluid.dygraph.guard():
        # CNN+LSTM
        model = CNN_RNN_Model4()
        opt = fluid.optimizer.AdamOptimizer(learning_rate=fluid.dygraph.ExponentialDecay(
              learning_rate=base_lr,
              decay_steps=2000,
              decay_rate=0.5,
              staircase=True),  regularization=fluid.regularizer.L2Decay(0.00005),  parameter_list=model.parameters())  #创建优化器
        
        # Resnet3D
        model3d = Resnet3dSub(10, frame_length)
        opt_3d = fluid.optimizer.AdamOptimizer(learning_rate=fluid.dygraph.ExponentialDecay(
              learning_rate=base_lr_3d,
              decay_steps=2500,
              decay_rate=0.5,
              staircase=True),  regularization=fluid.regularizer.L2Decay(0.00005),  parameter_list=model3d.parameters())  #创建优化器
        
        # 训练数据读取器
        train_loader = BatchedDataLoaderv2(face_data_dir, resolution, batchsize, frame_length)
        # 验证数据读取器
        validate_loader = BatchedDataLoaderv2(val_face_data_dir, resolution, batchsize, frame_length)
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
                frame_label = np.array(frame_label).astype('int64')

                img = fluid.dygraph.to_variable(frame_data)
                label = fluid.dygraph.to_variable(frame_label)
                label.stop_gradient = True
                
                # CONV_LSTM loss 计算 
                output, acc = model(img, label)
                # print("CNN+LSTM out is \n", output.numpy())
                # print("ACC is", acc.numpy())
                loss = fluid.layers.cross_entropy(output, label)
                mean_loss = fluid.layers.reduce_mean(loss)
                mean_loss.backward()
                total_loss += mean_loss
                opt.minimize(mean_loss)
                model.clear_gradients()

                # # 3D网络 loss 计算
                output3d, acc3d = model3d(img, label)
                # print("output3d is \n", output3d.numpy())
                # print(acc3d.numpy())
                loss3d = fluid.layers.cross_entropy(output3d, label) 
                mean_loss3d = fluid.layers.reduce_mean(loss3d)
                mean_loss3d.backward()
                total_loss += mean_loss3d
                opt_3d.minimize(mean_loss3d)
                model3d.clear_gradients()

                binary_tensor1 = np.array([[(1-weight_3d) for i in range(2)] for i in range(batchsize)]).astype("float32")
                binary_tensor1 = fluid.dygraph.to_variable(binary_tensor1)
                binary_tensor1.stop_gradient = True

                binary_tensor2 = np.array([[weight_3d for i in range(2)] for i in range(batchsize)]).astype("float32")
                binary_tensor2 = fluid.dygraph.to_variable(binary_tensor2)
                binary_tensor2.stop_gradient = True

                final_output = fluid.layers.elementwise_mul(binary_tensor1, output) + fluid.layers.elementwise_mul(binary_tensor2, output3d)
                # print("final output is \n", final_output.numpy())
                final_acc = fluid.layers.accuracy(final_output, label)
                # print("final acc is", final_acc.numpy())
                total_acc += final_acc.numpy()[0]
                # print(total_acc/smallepoch)

                smallepoch += 1

            print("EPOCH ", epoch)
            print("total loss is ", sum(total_loss.numpy()))
            print("Accuracy is ", total_acc/smallepoch)

            if epoch % 20 == 0:
                model.eval()
                cur_val_acc = eval_net(validate_loader, model, model3d, weight3d=0.35)
                if cur_val_acc > val_acc:
                    print("validation at epoch {} current acc {} is better than last acc {}".format(epoch, cur_val_acc, val_acc))
                    val_acc = cur_val_acc

            model.train()
            if epoch % 50 == 0 or epoch == MAX_EPOCH-1:
                fluid.save_dygraph(model.state_dict(), 'DpDcModel_BiLSTM_Conv3D_epoch{}'.format(epoch))
                fluid.save_dygraph(model3d.state_dict(), 'DpDcModel_BiLSTM_Conv3D_3DNet_epoch{}'.format(epoch))