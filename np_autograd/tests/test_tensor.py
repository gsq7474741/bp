from unittest import TestCase
import gol
import pygraphviz as pgv
import torch

gol.init()
# 先必须在主模块初始化（只在Main模块需要一次即可）
TENSOR_GRAPH_ATTR = dict(fontname="Times-Roman", fontsize=14, shape="polygon",
                         style="rounded", color="black",
                         fixedsize=False)
PARA_GRAPH_ATTR = dict(fontname="Times-Roman", fontsize=14, shape="polygon",
                       style="filled", fillcolor="#B4E7B7",
                       fixedsize=False)

OP_GRAPH_ATTR = dict(fontname="Times-Roman", fontsize=14, shape="circle",
                     style="filled", fillcolor="#BBE4FF",
                     fixedsize=False)

G = pgv.AGraph(directed=True, rankdir="RL", overlap=False, normlized=True,
               encoding='UTF-8')
TENSOR_DICT = {}
PARA_DICT = {}
OP_DICT = {}

gol.set_value('TENSOR_GRAPH_ATTR', TENSOR_GRAPH_ATTR)
gol.set_value('PARA_GRAPH_ATTR', PARA_GRAPH_ATTR)
gol.set_value('OP_GRAPH_ATTR', OP_GRAPH_ATTR)
gol.set_value('G', G)
gol.set_value('TENSOR_DICT', TENSOR_DICT)
gol.set_value('PARA_DICT', PARA_DICT)
gol.set_value('OP_DICT', OP_DICT)
gol.set_value('GRAPH_FLAG', True)

from np_autograd.tensor import Tensor
import np_autograd as anp
import numpy as np


class TestTensor(TestCase):

    def test_init_tensor(self):
        x = Tensor(np.array([1, 2, 3, 4]))
        self.assertEqual(x.shape, (4,))
        self.assertEqual(x.requires_grad, False)
        print(x)

    def test_tensor_backward_2(self):
        a = Tensor([1, 2, 3, 4, 5], requires_grad=True)
        b = Tensor([2, 2, 2, 2, 2], requires_grad=True)
        c = Tensor([5, 4, 3, 2, 1], requires_grad=True)

        d = a + (-b)
        e = (-b) + c
        f = d + e

        f.backward(Tensor(np.array([1, 1, 1, 1, 1])))
        # f.backward()
        print(a.grad)
        print(b.grad.data == np.array([-2, -2, -2, -2, -2]))
        self.assertTrue(np.alltrue(b.grad.data == np.array([-2, -2, -2, -2, -2])))

    def test_tensor_backward_1(self):
        a = Tensor(2.0, requires_grad=True)
        b = Tensor(5.0, requires_grad=True)
        d = Tensor(4.0, requires_grad=True)
        c = a * b
        e = c * d
        e.backward(Tensor(1))
        # e.backward(Tensor(1))
        self.assertEqual(a.grad.data, 20)
        self.assertEqual(b.grad.data, 8)
        self.assertEqual(c.grad.data, 4)
        self.assertEqual(d.grad.data, 10)
        print("Tensor a's grad is: {}".format(a.grad))
        print("Tensor b's grad is: {}".format(b.grad))
        print("Tensor c's grad is: {}".format(c.grad))
        print("Tensor d's grad is: {}".format(d.grad))
        G.layout()
        G.draw('test_tensor_backward_1.png')

    def test_ce_backward_0(self):
        np.random.seed(42)
        torch.manual_seed(42)

        np_x = np.random.rand(3, 7)
        np_y = np.random.randint(0, 7, size=3)

        x = Tensor(np_x, requires_grad=True)
        y = Tensor(np_y, requires_grad=False)

        torch_x = torch.tensor(np_x, requires_grad=True)
        torch_y = torch.tensor(np_y, requires_grad=False, dtype=torch.long)

        ce = anp.CrossEntropyLoss()
        torch_ce = torch.nn.CrossEntropyLoss()

        loss = ce(x, y)
        torch_loss = torch_ce(torch_x, torch_y)

        torch_loss.backward()
        loss.backward()

        torch_x_grad = torch_x.grad.data.numpy()
        x_grad = x.grad.data

        print(x.grad)
        print(torch_x.grad)

    def test_matmul_backward_0(self):
        np.random.seed(42)
        torch.manual_seed(42)

        np_x = np.random.rand(3, 5)
        np_y = np.random.rand(5, 7)

        x = Tensor(np_x, requires_grad=True)
        y = Tensor(np_y, requires_grad=True)

        torch_x = torch.tensor(np_x, requires_grad=True)
        torch_y = torch.tensor(np_y, requires_grad=True)

        op = anp.ops.MatMul()
        # torch_op = torch.matmul

        res = op(x, y)
        torch_res = torch_x @ torch_y

        self.assertTrue(np.allclose(res.data, torch_res.data.numpy()))

        torch_res.backward(torch.ones_like(torch_res))
        res.backward(np.ones_like(res.data))

        torch_x_grad = torch_x.grad.data.numpy()
        torch_y_grad = torch_y.grad.data.numpy()
        x_grad = x.grad.data
        y_grad = y.grad.data

        self.assertTrue(np.allclose(torch_x_grad, x_grad))
        self.assertTrue(np.allclose(torch_y_grad, y_grad))

    def test_relu_backward_0(self):
        np.random.seed(42)
        torch.manual_seed(42)

        np_x = np.random.rand(3, 5)
        # np_y = np.random.rand(5, 7)

        x = Tensor(np_x, requires_grad=True)
        # y = Tensor(np_y, requires_grad=True)

        torch_x = torch.tensor(np_x, requires_grad=True)
        # torch_y = torch.tensor(np_y, requires_grad=True)

        op = anp.ops.ReLU()
        # torch_op = torch.matmul
        res = op(x)
        torch_res = torch.relu(torch_x)

        self.assertTrue(np.allclose(res.data, torch_res.data.numpy()))

        torch_res.backward(torch.ones_like(torch_res))
        res.backward(np.ones_like(res.data))

        torch_x_grad = torch_x.grad.data.numpy()
        # torch_y_grad = torch_y.grad.data.numpy()
        x_grad = x.grad.data
        # y_grad = y.grad.data

        self.assertTrue(np.allclose(torch_x_grad, x_grad))
        # self.assertTrue(np.allclose(torch_y_grad, y_grad))

    def test_leakyrelu_backward_0(self):
        np.random.seed(42)
        torch.manual_seed(42)

        np_x = np.random.rand(3, 5)
        # np_y = np.random.rand(5, 7)

        x = Tensor(np_x, requires_grad=True)
        # y = Tensor(np_y, requires_grad=True)

        torch_x = torch.tensor(np_x, requires_grad=True)
        # torch_y = torch.tensor(np_y, requires_grad=True)

        op = anp.ops.LeakyReLU()
        # torch_op = torch.matmul
        res = op(x)
        torch_res = torch.nn.functional.leaky_relu(torch_x)

        self.assertTrue(np.allclose(res.data, torch_res.data.numpy()))

        torch_res.backward(torch.ones_like(torch_res))
        res.backward(np.ones_like(res.data))

        torch_x_grad = torch_x.grad.data.numpy()
        # torch_y_grad = torch_y.grad.data.numpy()
        x_grad = x.grad.data
        # y_grad = y.grad.data

        self.assertTrue(np.allclose(torch_x_grad, x_grad))
        # self.assertTrue(np.allclose(torch_y_grad, y_grad))

    def test_sigmoid_backward_0(self):
        np.random.seed(42)
        torch.manual_seed(42)

        np_x = np.random.rand(3, 5)
        # np_y = np.random.rand(5, 7)

        x = Tensor(np_x, requires_grad=True)
        # y = Tensor(np_y, requires_grad=True)

        torch_x = torch.tensor(np_x, requires_grad=True)
        # torch_y = torch.tensor(np_y, requires_grad=True)

        op = anp.ops.Sigmoid()
        # torch_op = torch.matmul
        res = op(x)
        torch_res = torch.sigmoid(torch_x)

        self.assertTrue(np.allclose(res.data, torch_res.data.numpy()))

        torch_res.backward(torch.ones_like(torch_res))
        res.backward(np.ones_like(res.data))

        torch_x_grad = torch_x.grad.data.numpy()
        # torch_y_grad = torch_y.grad.data.numpy()
        x_grad = x.grad.data
        # y_grad = y.grad.data

        self.assertTrue(np.allclose(torch_x_grad, x_grad))
        # self.assertTrue(np.allclose(torch_y_grad, y_grad))

    def test_softmax_backward_0(self):
        np.random.seed(42)
        torch.manual_seed(42)

        np_x = np.random.rand(3, 5)
        # np_y = np.random.rand(5, 7)

        x = Tensor(np_x, requires_grad=True)
        # y = Tensor(np_y, requires_grad=True)

        torch_x = torch.tensor(np_x, requires_grad=True)
        # torch_y = torch.tensor(np_y, requires_grad=True)

        op = anp.ops.Softmax()
        dim = -1
        res = op.forward(x, dim=dim)
        torch_res = torch.softmax(torch_x, dim=dim)

        self.assertTrue(np.allclose(res.data, torch_res.data.numpy()))

        torch_res.backward(torch.ones_like(torch_res))
        res.backward(np.ones_like(res.data))

        torch_x_grad = torch_x.grad.data.numpy()
        # torch_y_grad = torch_y.grad.data.numpy()
        x_grad = x.grad.data
        # y_grad = y.grad.data

        self.assertTrue(np.allclose(torch_x_grad, x_grad))
        # self.assertTrue(np.allclose(torch_y_grad, y_grad))
