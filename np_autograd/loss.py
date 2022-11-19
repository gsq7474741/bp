from np_autograd import Layer
from np_autograd.ops import CrossEntropy
from np_autograd.tensor import Tensor


class CrossEntropyLoss(Layer):

    def __init__(self, *args):
        super(CrossEntropyLoss, self).__init__(*args)

    def forward(self, source, target) -> Tensor:
        return CrossEntropy().forward(source, target)


class MSELoss(Layer):

    def __init__(self, *args):
        super(MSELoss, self).__init__(*args)

    def forward(self, source: Tensor, target: Tensor) -> Tensor:
        return (source - target) ** 2
