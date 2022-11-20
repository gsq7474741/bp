from abc import abstractmethod
from functools import wraps
from inspect import getattr_static

import numpy as np
import torch

from np_autograd.tensor import Tensor


def method_register(cls: object):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            return func(self, *args, **kwargs)

        if getattr_static(cls, func.__name__, None):
            msg = 'Error method name REPEAT, {} has exist'.format(func.__name__)
            raise NameError(msg)
        else:
            setattr(cls, func.__name__, wrapper)
        return func

    return decorator


@method_register(Tensor)
def __add__(self: Tensor, other: Tensor) -> Tensor:
    return Add().forward(self, other)


@method_register(Tensor)
def __neg__(self: Tensor) -> Tensor:
    return Neg().forward(self)


@method_register(Tensor)
def __sub__(self: Tensor, other: Tensor) -> Tensor:
    return Sub().forward(self, other)


@method_register(Tensor)
def __mul__(self: Tensor, other: Tensor) -> Tensor:
    return Mul().forward(self, other)


class OpBase:
    def __init__(self):
        super().__init__()
        # OP_DICT[id(self)] = self

    @abstractmethod
    def forward(self, *args) -> Tensor:
        pass

    @abstractmethod
    def backward(self, dout: Tensor, out: Tensor) -> ...:
        pass

    def __repr__(self) -> str:
        return self.__class__.__name__

    def __call__(self, *args) -> Tensor:
        return self.forward(*args)


class UnaryOpBase(OpBase):
    def __init__(self):
        super().__init__()
        self.x = None

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        pass


class BinaryOpBase(OpBase):
    def __init__(self):
        super().__init__()
        self.x = None
        self.y = None

    @abstractmethod
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        pass


class Add(BinaryOpBase):

    def forward(self, x: Tensor, y: Tensor):
        self.x = x
        self.y = y

        if x.requires_grad is True and y.requires_grad is True:
            return Tensor(x.data + y.data,
                          requires_grad=True,
                          creators=[x, y],
                          creation_op=self)
        return Tensor(x.data + y.data)

    def backward(self, dout, out):
        dx = Tensor(dout.data, )
        dy = Tensor(dout.data, )

        if dx.shape != self.x.shape:
            dx.data = dx.data.mean(axis=tuple(range(dx.data.ndim - self.x.data.ndim)))
        if dy.shape != self.y.shape:
            dy.data = dy.data.mean(axis=tuple(range(dy.data.ndim - self.y.data.ndim)))

        self.x.backward(dx, out)
        self.y.backward(dy, out)


class Sub(BinaryOpBase):
    def __init__(self):
        super().__init__()
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y

        if x.requires_grad is True and y.requires_grad is True:
            return Tensor(x.data - y.data,
                          requires_grad=True,
                          creators=[x, y],
                          creation_op=self)
        return Tensor(x.data - y.data)

    def backward(self, dout, out):
        dx = Tensor(dout.data, )
        dy = Tensor(-dout.data, )
        self.x.backward(dx, out)
        self.y.backward(dy, out)


class Mul(BinaryOpBase):
    def __init__(self):
        super().__init__()
        self.y = None
        self.x = None

    def forward(self, x, y):
        self.x = x
        self.y = y

        if x.requires_grad is True and y.requires_grad is True:
            return Tensor(x.data * y.data,
                          requires_grad=True,
                          creators=[x, y],
                          creation_op=self)
        return Tensor(x.data * y.data)

    def backward(self, dout, out):
        dx = Tensor(dout.data * self.y.data, )
        dy = Tensor(dout.data * self.x.data, )
        self.x.backward(dx, out)
        self.y.backward(dy, out)


class Div(BinaryOpBase):
    def __init__(self):
        super().__init__()
        self.y = None
        self.x = None

    def forward(self, x, y):
        self.x = x
        self.y = y

        if x.requires_grad is True and y.requires_grad is True:
            return Tensor(x.data / y.data,
                          requires_grad=True,
                          creators=[x, y],
                          creation_op=self)
        return Tensor(x.data / y.data)

    def backward(self, dout, out):
        dx = Tensor(dout.data / self.y.data, )
        dy = Tensor(-dout.data * out.data / self.y.data, )
        self.x.backward(dx, out)
        self.y.backward(dy, out)


class Neg(UnaryOpBase):
    def __init__(self):
        super().__init__()
        self.x = None

    def forward(self, x):
        self.x = x
        if x.requires_grad is True:
            return Tensor(x.data * (-1),
                          requires_grad=True,
                          creators=[x],
                          creation_op=self)
        return Tensor(x.data * (-1))

    def backward(self, dout, out):
        dx = Tensor(dout.data * (-1))

        self.x.backward(dx, out)


class MatMul(BinaryOpBase):
    def __init__(self):
        super().__init__()
        self.y = None
        self.x = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        if x.requires_grad is True and y.requires_grad is True:
            return Tensor(x.data @ y.data,
                          requires_grad=True,
                          creators=[x, y],
                          creation_op=self)
        return Tensor(x.data @ y.data)

    def backward(self, dout, out):
        dx = Tensor(dout.data @ self.y.data.T, )
        dy = Tensor(self.x.data.T @ dout.data, )
        self.x.backward(dx, out)
        self.y.backward(dy, out)


class Sum(UnaryOpBase):
    def __init__(self):
        super().__init__()
        self.x = None

    def forward(self, x):
        self.x = x
        if x.requires_grad is True:
            return Tensor(x.data.sum(),
                          requires_grad=True,
                          creators=[x],
                          creation_op=self)
        return Tensor(x.data.sum())

    def backward(self, dout, out):
        dx = Tensor(np.ones_like(self.x.data) * dout.data, )
        self.x.backward(dx, out)


class Mean(UnaryOpBase):
    def __init__(self):
        super().__init__()
        self.x = None

    def forward(self, x):
        self.x = x
        if x.requires_grad is True:
            return Tensor(x.data.mean(),
                          requires_grad=True,
                          creators=[x],
                          creation_op=self)
        return Tensor(x.data.mean())

    def backward(self, dout, out):
        dx = Tensor(np.ones_like(self.x.data) / self.x.data.size * dout.data, )
        self.x.backward(dx, out)


class Expand(UnaryOpBase):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape
        self.x = None

    def forward(self, x):
        self.x = x
        if x.requires_grad is True:
            return Tensor(x.data.reshape(self.shape),
                          requires_grad=True,
                          creators=[x],
                          creation_op=self)
        return Tensor(x.data.reshape(self.shape))

    def backward(self, dout, out):
        dx = Tensor(dout.data.reshape(self.x.data.shape), )
        self.x.backward(dx, out)


class Transpose(UnaryOpBase):
    def __init__(self):
        super().__init__()
        self.x = None

    def forward(self, x):
        self.x = x
        if x.requires_grad is True:
            return Tensor(x.data.T,
                          requires_grad=True,
                          creators=[x],
                          creation_op=self)
        return Tensor(x.data.T)

    def backward(self, dout, out):
        dx = Tensor(dout.data.T, )
        self.x.backward(dx, out)


class Sigmoid(UnaryOpBase):
    def __init__(self):
        super().__init__()
        self.x = None

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, x):
        self.x = x
        if x.requires_grad is True:
            return Tensor(self.sigmoid(x.data),
                          requires_grad=True,
                          creators=[x],
                          creation_op=self)
        return Tensor(1 / (1 + np.exp(-x.data)))

    def backward(self, dout, out):
        dx = Tensor(out.data * (1 - out.data) * dout.data, )
        self.x.backward(dx, out)


class Tanh(UnaryOpBase):
    def __init__(self):
        super().__init__()
        self.x = None

    def forward(self, x):
        self.x = x
        if x.requires_grad is True:
            return Tensor(np.tanh(x.data),
                          requires_grad=True,
                          creators=[x],
                          creation_op=self)
        return Tensor(np.tanh(x.data))

    def backward(self, dout, out):
        dx = Tensor((1 - out.data ** 2) * dout.data, )
        self.x.backward(dx, out)


class ReLU(UnaryOpBase):
    def __init__(self):
        super().__init__()
        self.x = None

    def forward(self, x):
        self.x = x
        if x.requires_grad is True:
            return Tensor(np.maximum(0, x.data),
                          requires_grad=True,
                          creators=[x],
                          creation_op=self)
        return Tensor(np.maximum(0, x.data))

    def backward(self, dout, out):
        dx = Tensor((out.data > 0) * dout.data, )
        self.x.backward(dx, out)


class LeakyReLU(UnaryOpBase):
    def __init__(self, alpha=0.01):
        super().__init__()
        self.x = None
        self.alpha = alpha

    def forward(self, x):
        self.x = x
        f1 = np.maximum(0, x.data)
        f2 = self.alpha * (x.data - np.abs(x.data)) * 0.5

        if x.requires_grad is True:
            return Tensor(f1 + f2,
                          requires_grad=True,
                          creators=[x],
                          creation_op=self)
        return Tensor(f1 + f2)

    def backward(self, dout, out):
        dx = np.ones_like(out.data)
        dx[out.data < 0] = self.alpha
        dx = Tensor(dx, )
        self.x.backward(dx, out)


class Dropout(UnaryOpBase):
    def __init__(self, p=0.2):
        super().__init__()
        self.p = p
        self.x = None
        self.mask = None

    def forward(self, x):
        self.x = x
        self.mask = np.random.binomial(1, self.p, size=x.data.shape)
        if x.requires_grad is True:
            return Tensor(x.data * self.mask,
                          requires_grad=True,
                          creators=[x],
                          creation_op=self)
        return Tensor(x.data * self.mask)

    def backward(self, dout, out):
        dx = Tensor(dout.data * self.mask, )
        self.x.backward(dx, out)


class Softmax(UnaryOpBase):
    def __init__(self):
        super().__init__()
        self.x = None

    def forward(self, x, dim=-1):
        self.x = x
        self.dim = dim

        def softmax(x):
            # x.data-np.max(x.data)
            return np.exp(x.data - np.max(x.data)) / np.expand_dims(np.exp(x.data - np.max(x.data)).sum(axis=dim),
                                                                    axis=dim)

        if x.requires_grad is True:
            # np.exp(x.data) / np.exp(x.data).sum(axis=-1)
            return Tensor(softmax(x),
                          requires_grad=True,
                          creators=[x],
                          creation_op=self)
        return Tensor(softmax(x))

    def backward(self, dout, out):
        dx = Tensor((dout.data - np.expand_dims((out.data * dout.data).sum(axis=self.dim), axis=self.dim)) * out.data, )
        self.x.backward(dx, out)


class Log(UnaryOpBase):
    def __init__(self):
        super().__init__()
        self.x = None

    def forward(self, x):
        self.x = x
        if x.requires_grad is True:
            return Tensor(np.log(x.data),
                          requires_grad=True,
                          creators=[x],
                          creation_op=self)
        return Tensor(np.log(x.data))

    def backward(self, dout, out):
        dx = Tensor(1 / self.x.data * dout.data, )
        self.x.backward(dx, out)


class CrossEntropy(BinaryOpBase):
    def __init__(self):
        super().__init__()
        self.y_onehot = None
        self.loss = None
        self.x = None
        self.y = None

    @staticmethod
    def softmax(x, dim=1):
        x -= np.max(x)
        return np.exp(x) / np.sum(np.exp(x), axis=dim).reshape(-1, 1)

    def forward(self, x, y):
        # def softmax(x, dim=1):
        #     x = x - torch.max(x, dim=dim)[0].unsqueeze(dim=dim)  # 防止溢出
        #     res = torch.exp(x) / torch.sum(torch.exp(x), dim=dim).unsqueeze(dim=dim)
        #     return res

        self.x = x
        self.y = y

        num_data, num_class = x.shape

        # log_p = np.array([np.log(x[i]) for i in range(num_data)])
        self.y_onehot = np.eye(num_class)[y]

        loss = - (np.log(self.softmax(x.data, dim=-1)) * self.y_onehot).sum() / num_data
        self.loss = - np.sum(np.log(self.softmax(x.data, dim=-1)) * self.y_onehot, axis=0) / num_data

        x_torch = torch.tensor(x.data, requires_grad=True)
        y_torch = torch.tensor(y.data, dtype=torch.long)
        loss_torch = torch.nn.functional.cross_entropy(x_torch, y_torch)
        self.loss_torch = loss_torch
        self.x_torch = x_torch
        self.y_torch = y_torch

        if x.requires_grad is True:
            return Tensor(loss_torch.detach().numpy(),
                          requires_grad=True,
                          creators=[x, y],
                          creation_op=self)
        return Tensor(loss_torch.detach().numpy())

    def backward(self, dout, out):
        if dout is None:
            dout = Tensor(np.ones_like(out.data))
        # dx = Tensor(out.data * (1 - out.data) * dout.data)
        dx = Tensor(self.softmax(self.x.data, dim=-1) - self.y_onehot, )

        self.loss_torch.backward()
        dx_torch = self.x_torch.grad.numpy()

        self.x.backward(dx_torch, out)
