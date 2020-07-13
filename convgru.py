import paddle.fluid as fluid
from paddle.fluid.layers import sigmoid, relu, tanh, concat
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.dygraph import Layer, Conv2D, Sequential
from paddle.fluid.dygraph.base import to_variable
import numpy as np

"""
卷积版本的GRU单元
"""
class ConvGruCell(Layer):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvGruCell, self).__init__()
        self.input_dim = in_channels
        self.hidden_dim = out_channels
        self.kernel_size = kernel_size
        self.padding = (kernel_size[0] // 2, kernel_size[1] // 2)

        self.update_gate = Conv2D(num_channels=self.input_dim + self.hidden_dim,
                                  num_filters=self.hidden_dim,
                                  filter_size=self.kernel_size,
                                  padding=self.padding)
        self.reset_gate = Conv2D(num_channels=self.input_dim + self.hidden_dim,
                                 num_filters=self.hidden_dim,
                                 filter_size=self.kernel_size,
                                 padding=self.padding)
        self.out_gate = Conv2D(num_channels=self.input_dim + self.hidden_dim,
                               num_filters=self.hidden_dim,
                               filter_size=self.kernel_size,
                               padding=self.padding)

    def forward(self, input_tensor, cur_state):
        h_cur = cur_state
        x_in = concat([input_tensor, h_cur], axis=1)
        update = sigmoid(self.update_gate(x_in))
        reset = sigmoid(self.reset_gate(x_in))
        x_out = tanh(self.out_gate(concat([input_tensor, h_cur * reset], axis=1)))
        h_new = h_cur * (1 - update) + x_out * update
        return h_new

    def init_hidden(self, b, h, w):
        # 初始化
        W = fluid.layers.create_parameter(shape=[b, self.hidden_dim, h, w], dtype='float32',
                                          attr=ParamAttr(initializer=fluid.initializer.XavierInitializer()))

        return W


class ConvGRU(Layer):
    def __init__(self, in_channels, hidden_channels, kernel_size, num_layers,
                 batch_first=True, return_all_layers=False):
        super(ConvGRU, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_channels = self._extend_for_multilayer(hidden_channels, num_layers)

        if not len(kernel_size) == len(hidden_channels) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = in_channels
        self.input_dim = in_channels
        self.hidden_dim = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.return_all_layers = return_all_layers

        self.cell_list = Sequential()
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            self.cell_list.add_sublayer(name='{}'.format(i), sublayer=ConvGruCell(in_channels=cur_input_dim,
                                                                                  out_channels=self.hidden_dim[i],
                                                                                  kernel_size=self.kernel_size[i]))

    def forward(self, input_tensor, hidden_state=None):
        """
                Parameters
                ----------
                input_tensor: todo
                    5-D Tensor (b, t, c, h, w)
                hidden_state: todo
                    None. todo implement stateful
                Returns
                -------
                last_state_list, layer_output
                """
        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            b, _, _, h, w = input_tensor.shape
            hidden_state = self._init_hidden(b, h, w)

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.shape[1]
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h = self.cell_list['{}'.format(layer_idx)](input_tensor=cur_layer_input[:, t, :, :, :],
                                                           cur_state=h)
                output_inner.append(h)

            layer_output = fluid.layers.stack(output_inner, axis=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append(h)

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, b, h, w):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(b, h, w))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class ConvBGRU(Layer):
    # Constructor
    def __init__(self, in_channels, hidden_channels,
                 kernel_size, num_layers, batch_first=True):
        super(ConvBGRU, self).__init__()
        self.forward_net = ConvGRU(in_channels, hidden_channels // 2, kernel_size,
                                   num_layers, batch_first=batch_first)
        self.reverse_net = ConvGRU(in_channels, hidden_channels // 2, kernel_size,
                                   num_layers, batch_first=batch_first)

    def forward(self, xforward):
        """
        xforward, xreverse = B T C H W tensors.
        """
        xreverse = xforward[:, ::-1, :, :, :]
        y_out_fwd, _ = self.forward_net(xforward)
        y_out_rev, _ = self.reverse_net(xreverse)
        y_out_fwd = y_out_fwd[-1]  # outputs of last CLSTM layer = B, T, C, H, W
        y_out_rev = y_out_rev[-1]

        # print(reversed_idx)
        y_out_rev = y_out_rev[:, ::-1, :, :, :]
        # print(y_out_rev.shape)
        ycat = fluid.layers.concat([y_out_fwd, y_out_rev], axis=2)

        return ycat


if __name__ == '__main__':
    with fluid.dygraph.guard():
        input = np.random.randn(5, 20, 1280, 7, 7).astype('float32')
        x = to_variable(input)
        model = ConvBGRU(in_channels=1280, hidden_channels=64, kernel_size=(3, 3), num_layers=2)
        out = model(x)
        print(out.shape)
