import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.regularizer import L2Decay
import numpy as np
from paddle.fluid.dygraph.nn import Conv2D, BatchNorm
from paddle.fluid.dygraph import Linear
from paddle.fluid.dygraph.base import to_variable
import math
from EfficientNet import *
from model_utils import *
from resnet_3D import generate_model
# from resnet_3D_framesub import generate_model
from convlstm import ConvBLSTM
from paddle.fluid.dygraph import Linear


"""
以帧间差的形式
做一个            CONV双向LSTM      + resnet3D帧间差双路网络
调整了一下全连接层
"""



class CNNEnoder(fluid.dygraph.Layer):
    def __init__(self, drop_p=0.5):
        super(CNNEnoder, self).__init__()
        model_name = "efficientnet-b0"
        override_params = {"num_classes": 512}
        blocks_args, global_params = get_model_params(model_name, override_params=override_params)

        self.drop_p = drop_p
        self.BackBone = EfficientNet(blocks_args, global_params)

    def forward(self, x_3d):
        cnn_embed_seq = []
        for time in range(x_3d.shape[1]):
            x = self.BackBone(x_3d[:, time, :, :, :])
            x = fluid.layers.flatten(x, axis=1)
            x = fluid.layers.relu(x)
            x = fluid.layers.dropout(x, self.drop_p)
            cnn_embed_seq.append(x)

        cnn_embed_seq = fluid.layers.stack(cnn_embed_seq, axis=0)
        # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
        # print(cnn_embed_seq.shape)
        cnn_embed_seq = fluid.layers.transpose(cnn_embed_seq, perm=[1, 0, 2])
        cnn_embed_seq = fluid.layers.unsqueeze(input=cnn_embed_seq, axes=[3, 4])
        # cnn_embed_seq: shape=(batch, time_step, input_size)
        # print(cnn_embed_seq.shape)
        # print("cnn shape",  cnn_embed_seq.shape)
        return cnn_embed_seq


class CNN_RNN_Model4(fluid.dygraph.Layer):
    def __init__(self):
        super(CNN_RNN_Model4, self).__init__()
        self.EfficientNet = CNNEnoder()
        self.RNN = ConvBLSTM(in_channels=512, hidden_channels=64, kernel_size=(3, 3), num_layers=1)
        self.fc1 = Linear(1280, 512, param_attr=ParamAttr(initializer=fluid.initializer.XavierInitializer()))
        self.fc2 = Linear(512, 256, param_attr=ParamAttr(initializer=fluid.initializer.XavierInitializer()))
        self.fc3 = Linear(256, 2, param_attr=ParamAttr(initializer=fluid.initializer.XavierInitializer()))

    def forward(self, x, label=None):
        x = self.EfficientNet(x)
        x = self.RNN(x)
        # print(x.shape)
        x = fluid.layers.dropout(x, 0.5)
        x = fluid.layers.flatten(x)
        # print(x.shape)
        x = self.fc1(x)
        x = fluid.layers.relu(x)
        x = self.fc2(x)
        x = fluid.layers.relu(x)
        x = self.fc3(x)
        x = fluid.layers.sigmoid(x)

        y = fluid.layers.softmax(x)
        if label is not None:
            acc = fluid.layers.accuracy(input=y, label=label)
            return y, acc
        else:
            return y


# class Resnet3d(fluid.dygraph.Layer):
#     def __init__(self, model_depth):
#         super(Resnet3d, self).__init__()
#         self.net = generate_model(model_depth, conv1_t_size=10, n_classes=2)

#     def forward(self, x, label=None):
#         x_3d = fluid.layers.transpose(x, perm=[0, 2, 1, 3, 4])
#         x_3d = self.net(x_3d)

#         y = fluid.layers.softmax(x_3d)
#         if label is not None:
#             acc = fluid.layers.accuracy(input=y, label=label)
#             return y, acc
#         else:
#             return y

class Resnet3dSub(fluid.dygraph.Layer):
    def __init__(self, model_depth, frame_length):
        super(Resnet3dSub, self).__init__()
        self.net = generate_model(model_depth, conv1_t_size=frame_length - 1, n_classes=2)

    def forward(self, x, label=None):

        x_3d = fluid.layers.transpose(x, perm=[0, 2, 1, 3, 4])

        timelength = x_3d.shape[2]
        frame_tensor = []
        for timeframe in range(1, timelength):
            frame_sub = fluid.layers.abs(fluid.layers.elementwise_sub(x_3d[:, :, timeframe, :, :],
                                                                      x_3d[:, :, timeframe - 1, :, :]))

            frame_tensor.append(frame_sub)
        x_3d = fluid.layers.stack(frame_tensor, axis=2)

        x_3d = self.net(x_3d)

        y = fluid.layers.softmax(x_3d)
        if label is not None:
            acc = fluid.layers.accuracy(input=y, label=label)
            return y, acc
        else:
            return y


if __name__ == "__main__":
    with fluid.dygraph.guard():
        # batch_size, 20(frames), 3, 240, 240
        x = np.random.randn(2, 20, 3, 224, 224).astype('float32')
        x = to_variable(x)
        # model = CNN_RNN_Model3()
        model = Resnet3dSub(10, 20)

        out = model(x)
        print(out.shape)
        # print(out)
