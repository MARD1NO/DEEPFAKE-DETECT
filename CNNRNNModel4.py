import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.regularizer import L2Decay
import numpy as np
from paddle.fluid.dygraph.nn import Conv2D, BatchNorm
from paddle.fluid.dygraph.base import to_variable
import math
from EfficientNet import *
from model_utils import *
from convlstm import ConvBLSTM
from paddle.fluid.dygraph import Linear

"""
CNN + 双向LSTM
区别在调整了全连接层数量

后面一个CNN_RNN_Model5 则是加入了 标签平滑的功能
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
        self.RNN =ConvBLSTM(in_channels=512, hidden_channels=64, kernel_size=(3, 3), num_layers=1)
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

class CNN_RNN_Model5(fluid.dygraph.Layer):
    """
    带label_smooth版本
    """
    def __init__(self, use_label_smooth=False):
        super(CNN_RNN_Model5, self).__init__()
        self.use_label_smooth = use_label_smooth
        self.EfficientNet = CNNEnoder()
        self.RNN =ConvBLSTM(in_channels=512, hidden_channels=64, kernel_size=(3, 3), num_layers=1)
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
            if self.use_label_smooth:
                label = fluid.layers.argmax(label, axis=1)
                label = fluid.layers.unsqueeze(label, axes=1)
            acc = fluid.layers.accuracy(input=y, label=label)
            return y, acc
        else:
            return y
      
if __name__ == "__main__":        
    with fluid.dygraph.guard():
        # batch_size, 10(frames), 3, 240, 240
        x = np.random.randn(8, 20, 3, 224, 224).astype('float32')
        x = to_variable(x)
        model = CNN_RNN_Model4()

        out = model(x)
        print(out.shape)
        # print(out)