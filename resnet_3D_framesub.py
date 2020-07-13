import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.regularizer import L2Decay
from paddle.fluid.dygraph import Sequential
from paddle.fluid.dygraph.nn import Conv3D, BatchNorm, Linear, Conv2D
from paddle.fluid.layers import relu, pool3d, adaptive_pool3d, flatten
from paddle.fluid.dygraph.base import to_variable
from paddle.fluid.initializer import Xavier, Constant
import math
from model_utils import *
import collections
import numpy as np

"""
https://github.com/kenshohara/3D-ResNets-PyTorch

在网络输入做了个帧间差操作

具体是 i+1帧  elementwise-sub i帧，并做了个abs处理

"""

def get_inplanes():
    return [64, 128, 256, 512]


def conv3x3x3(in_planes, out_planes, stride=1):
    return Conv3D(in_planes,
                  out_planes,
                  filter_size=3,
                  stride=stride,
                  padding=1,
                  param_attr=ParamAttr(initializer=Xavier()))


def conv1x1x1(in_planes, out_planes, stride=1):
    return Conv3D(in_planes,
                  out_planes,
                  filter_size=1,
                  stride=stride,
                  param_attr=ParamAttr(initializer=Xavier()))

class DepthwiseConv(fluid.dygraph.Layer):
    def __init__(self, input_channel, output_channel, filter_size=3, stride=1, relu=False):
        super(DepthwiseConv, self).__init__()
        self.depthwiseConvBN = fluid.dygraph.Sequential(
            Conv2D(input_channel, output_channel, filter_size=filter_size, stride=stride,
                   groups=input_channel, padding=1),
            BatchNorm(num_channels=output_channel),
        )
        self.relu = relu

    def forward(self, x):
        y = self.depthwiseConvBN(x)
        if self.relu:
            y = fluid.layers.relu(y)
        return y

class FrameSubNet(fluid.dygraph.Layer):
    def __init__(self, input_channel, output_channel):
        super(FrameSubNet, self).__init__()
        self.depthwiseconv1 = DepthwiseConv(input_channel, output_channel, relu=True)
        self.depthwiseconv2 = DepthwiseConv(input_channel, output_channel, relu=True)
        self.pointwiseconv1 = Conv2D(input_channel, output_channel, 1)
    def forward(self, x):
        """
        输入为 batch channel time height width
        :param x: 
        :return: 
        """
        timelength = x.shape[2]
        frame_tensor = []
        for timeframe in range(timelength):
            if timeframe == 0:
                # 为了保证时间维度一致
                frame_sub = x[:, :, timeframe, :, :]
            else:
                frame_sub = fluid.layers.elementwise_sub(x[:, :, timeframe, :, :], 
                                                        x[:, :, timeframe-1, :, :])
            frame_sub = self.depthwiseconv1(frame_sub)
            frame_sub = self.depthwiseconv2(frame_sub)
            frame_sub = self.pointwiseconv1(frame_sub)
            # print(frame_sub.shape)
            frame_tensor.append(frame_sub)
        # return fluid.layers.concat(frame_tensor)
        return fluid.layers.stack(frame_tensor, axis=2)

class BasicBlock(fluid.dygraph.Layer):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3x3(in_planes, planes//2, stride)
        self.bn1 = BatchNorm(planes//2)
        self.conv2 = conv3x3x3(planes//2, planes//2)
        self.bn2 = BatchNorm(planes//2)
        # 另外一半以帧间差方式计算
        self.frameSubLayer = FrameSubNet(planes//2, planes//2)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        # print("out shape is ", out.shape)
        # print("residual shape is ", residual.shape)

        # print("now out shape is", out.shape)
        # print("use frame sub")

        frame_sub_tensor = self.frameSubLayer(out)
        frame_sub_tensor = relu(frame_sub_tensor)

        # print("frame sub tensor shape is ", frame_sub_tensor.shape)
        out = fluid.layers.concat([out, frame_sub_tensor], axis=1)
        out += residual
        out = relu(out)
        


        return out


class Bottleneck(fluid.dygraph.Layer):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = BatchNorm(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = BatchNorm(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = BatchNorm(planes * self.expansion)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = relu(out)

        return out

class ResNet(fluid.dygraph.Layer):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=3,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=400):
        super(ResNet, self).__init__()
        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool
        self.conv1 = Conv3D(n_input_channels,
                            self.in_planes,
                            filter_size=(conv1_t_size, 7, 7),
                            stride=(conv1_t_stride, 2, 2),
                            padding=(conv1_t_size // 2, 3, 3))
        self.bn1 = BatchNorm(self.in_planes)
        # self.maxpool = MaxPool3D(filter_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)

        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)
        

        # self.avgpool = AdaptiveAvgPool3d(1, 1, 1)
        self.fc = Linear(block_inplanes[3] * block.expansion, n_classes,
                         param_attr=ParamAttr(initializer=Xavier()), act='sigmoid')

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                # 这里下采样我是用1x1卷积做，这里先不写
                pass
            else:
                downsample = Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    BatchNorm(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return Sequential(*layers)

    def forward(self, x):
        # print(x[0, :, :, :, :].shape)
        x = self.conv1(x)
        # print("conv1 shape", x.shape)
        x = self.bn1(x)
        x = relu(x)
        if not self.no_max_pool:
            x = pool3d(x, pool_size=3, pool_type='max',
                      pool_stride=2, pool_padding=1,
                      data_format="NCDHW")

        # print("conv1 shape", x.shape)

        x = self.layer1(x)
        # print("layer1 shape", x.shape)

        x = self.layer2(x)
        # print("layer2 shape", x.shape)
        x = fluid.layers.dropout(x, 0.5)
        x = self.layer3(x)
        # print("layer3 shape", x.shape)

        x = self.layer4(x)
        # print("layer4 shape", x.shape)
        
        x = fluid.layers.dropout(x, 0.5)


        x = adaptive_pool3d(x, pool_size=[1, 1, 1],
                               pool_type='avg')

        x = flatten(x)
        x = self.fc(x)

        return x

def generate_model(model_depth, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
    elif model_depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
    elif model_depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
    elif model_depth == 152:
        model = ResNet(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
    elif model_depth == 200:
        model = ResNet(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)

    return model


if __name__ == "__main__":

    with fluid.dygraph.guard():
        """
        输入：
            输入Tensor的维度： [N,Cin,Din,Hin,Win]
        """
        x = np.random.randn(10, 3, 8, 224, 224).astype('float32')
        x = to_variable(x)
        net = generate_model(10, conv1_t_size=8)
        # net = FrameSubNet(3, 3)
        out = net(x)
        print(out.shape)