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
稍微调整了下网络结构
还没具体试验
"""


class CNNEnoder(fluid.dygraph.Layer):
    def __init__(self, fc_hidden1=512, fc_hidden2=512, drop_p=0.3, CNN_embed_dim=300):
        super(CNNEnoder, self).__init__()
        model_name = "efficientnet-b0"
        override_params = {"num_classes": 1280}
        blocks_args, global_params = get_model_params(model_name, override_params=override_params)

        self.drop_p = drop_p
        self.BackBone = EfficientNet(blocks_args, global_params)
        self.fc1 = Linear(1280, fc_hidden1)
        self.bn1 = BatchNorm(fc_hidden1)
        self.fc2 = Linear(fc_hidden1, fc_hidden2)
        self.bn2 = BatchNorm(fc_hidden2)
        self.fc3 = Linear(fc_hidden2, CNN_embed_dim)

    def forward(self, x_3d):
        cnn_embed_seq = []
        for time in range(x_3d.shape[1]):
            x = self.BackBone(x_3d[:, time, :, :, :])
            x = fluid.layers.flatten(x, axis=1)
            # print(x.shape)
            x = self.bn1(self.fc1(x))
            x = fluid.layers.relu(x)
            x = self.bn2(self.fc2(x))
            x = fluid.layers.relu(x)
            x = fluid.layers.dropout(x, dropout_prob=self.drop_p)
            x = self.fc3(x)
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


class DecoderRNN(fluid.dygraph.Layer):
    def __init__(self, CNN_embed_dim=300, kernel_size=(3, 3)
                 , frame_length=20, h_RNN_layers=3,
                 h_RNN=128, h_FC_dim=128, drop_p=0.3, num_classes=50):
        super(DecoderRNN, self).__init__()
        self.frame_length = frame_length
        self.RNN_input_size = CNN_embed_dim
        self.h_RNN_layers = h_RNN_layers  # RNN hidden layers
        self.h_RNN = h_RNN  # RNN hidden nodes
        self.h_FC_dim = h_FC_dim
        self.drop_p = drop_p
        self.num_classes = num_classes
        self.BLSTM = ConvBLSTM(CNN_embed_dim, self.h_RNN, kernel_size=kernel_size,
                               num_layers=h_RNN_layers)
        self.fc1 = Linear(self.h_RNN*self.frame_length, self.h_FC_dim)
        self.fc2 = Linear(self.h_FC_dim, self.num_classes)

    def forward(self, x):
        x = self.BLSTM(x)
        print("LSTM SHAPE", x.shape)
        x = fluid.layers.flatten(x)
        # FC layers
        print("flatten shape is", x.shape)
        x = self.fc1(x)  # choose RNN_out at the last time step
        x = fluid.layers.relu(x)
        x = fluid.layers.dropout(x, dropout_prob=self.drop_p)
        x = self.fc2(x)
        return x



class CNN_RNN_Model(fluid.dygraph.Layer):
    """
    带label_smooth版本
    """

    def __init__(self, use_label_smooth=False):
        super(CNN_RNN_Model, self).__init__()
        self.use_label_smooth = use_label_smooth
        self.EfficientNet = CNNEnoder(CNN_embed_dim=300)
        self.RNN = DecoderRNN(CNN_embed_dim=300, num_classes=2)

    def forward(self, x, label=None):
        x = self.EfficientNet(x)
        x = self.RNN(x)
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
        x = np.random.randn(1, 20, 3, 224, 224).astype('float32')
        x = to_variable(x)
        model = CNN_RNN_Model()

        out = model(x)
        print(out.shape)
        # print(out)

