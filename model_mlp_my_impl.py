from np_autograd.layer import Layer, Linear, ReLU, Softmax


class MLP(Layer):
    def __init__(self, input_dim, output_dim, num_layers=3, hidden_dim=None, dropout=0.2, *args, **kwargs):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.num_layers = num_layers
        self.hidden_dim = input_dim * 2 if hidden_dim is None else hidden_dim

        self.input_layer = Linear(self.input_dim, self.hidden_dim, name='input_layer')
        self.hid1 = Linear(self.hidden_dim, self.hidden_dim, name='hid1')
        self.hid2 = Linear(self.hidden_dim, self.hidden_dim, name='hid2')
        self.hid3 = Linear(self.hidden_dim, self.hidden_dim, name='hid3')
        self.out_layer = Linear(self.hidden_dim, self.output_dim, name='out_layer')

        self.softmax = Softmax()

    def forward(self, x):
        x = self.input_layer(x)
        x = ReLU()(x)

        x = self.hid1(x)
        x = ReLU()(x)

        x = self.hid2(x)
        x = ReLU()(x)

        x = self.hid3(x)
        x = ReLU()(x)

        x = self.out_layer(x)
        return x
