import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.regularizer import L2Decay
import numpy as np
from paddle.fluid.dygraph.nn import Conv2D, BatchNorm
from paddle.fluid.dygraph.base import to_variable
import math
from EfficientNet_without_fc import EfficientNet
from model_utils import *
from moggru import ConvMogBGRU
from paddle.fluid.dygraph import Linear

"""
CNN+MogGRU模型
"""
class CNNEnoder(fluid.dygraph.Layer):
    def __init__(self, drop_p=0.5, is_test=False):
        super(CNNEnoder, self).__init__()
        model_name = "efficientnet-b0"
        blocks_args, global_params = get_model_params(model_name, override_params=None)
        self.is_test = is_test
        self.drop_p = drop_p
        self.BackBone = EfficientNet(blocks_args, global_params)


    def forward(self, x_3d):
        cnn_embed_seq = []
        for time in range(x_3d.shape[1]):
            x = self.BackBone(x_3d[:, time, :, :, :])
            x = fluid.layers.relu(x)
            x = fluid.layers.dropout(x, self.drop_p, is_test=self.is_test)
            cnn_embed_seq.append(x)

        cnn_embed_seq = fluid.layers.stack(cnn_embed_seq, axis=0)
        # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
        # print(cnn_embed_seq.shape)
        
        cnn_embed_seq = fluid.layers.transpose(cnn_embed_seq, perm=[1, 0, 2, 3, 4])
        # cnn_embed_seq = fluid.layers.unsqueeze(input=cnn_embed_seq, axes=[3, 4])
        
        # cnn_embed_seq: shape=(batch, time_step, input_size)
        # print(cnn_embed_seq.shape)
        # print("cnn shape",  cnn_embed_seq.shape)
        return cnn_embed_seq

class CNNGRUModel(fluid.dygraph.Layer):
    """
    EfficientNet 去掉最后全连接层，保留了 batch, 1280, 7, 7
    最后做了个3D池化
    带label_smooth版本
    """
    def __init__(self, use_label_smooth=False, is_test=False):
        super(CNNGRUModel, self).__init__()
        self.use_label_smooth = use_label_smooth
        self.is_test = is_test
        self.EfficientNet = CNNEnoder(is_test=is_test)
        self.RNN = ConvMogBGRU(in_channels=1280, hidden_channels=320, kernel_size=(3, 3), num_layers=3)
        self.fc1 = Linear(320, 256, param_attr=ParamAttr(initializer=fluid.initializer.XavierInitializer()))
        self.fc2 = Linear(256, 128, param_attr=ParamAttr(initializer=fluid.initializer.XavierInitializer()))
        self.fc3 = Linear(128, 2, param_attr=ParamAttr(initializer=fluid.initializer.XavierInitializer()))
        

    def forward(self, x, label=None):
        x = self.EfficientNet(x)
        x = self.RNN(x)
        # print(x.shape)
        x = fluid.layers.transpose(x, perm=[0, 2, 1, 3, 4])
        x = fluid.layers.pool3d(x,  global_pooling=True, pool_type='avg')
        x = fluid.layers.squeeze(x, axes=[2, 3, 4])
        
        # print(x.shape)
        x = self.fc1(x)
        x = fluid.layers.dropout(x, 0.5, is_test=self.is_test)
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
        x = np.random.randn(8, 20, 3, 224, 224).astype('float32')
        x = to_variable(x)
        model = CNNGRUModel()

        out = model(x)
        print(out.shape)
        # print(out)