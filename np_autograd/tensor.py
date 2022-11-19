import numpy as np
import pygraphviz as pgv
import gol

TENSOR_GRAPH_ATTR: dict = gol.get_value('TENSOR_GRAPH_ATTR')
PARA_GRAPH_ATTR: dict = gol.get_value('PARA_GRAPH_ATTR')
OP_GRAPH_ATTR: dict = gol.get_value('OP_GRAPH_ATTR')
G: pgv.AGraph = gol.get_value('G')
# TENSOR_DICT: list = gol.get_value('TENSOR_DICT')
# PARA_DICT: list = gol.get_value('PARA_DICT')
# OP_DICT: list = gol.get_value('OP_DICT')
GRAPH_FLAG: bool = gol.get_value('GRAPH_FLAG')


class Tensor:
    def __init__(self, data, requires_grad=False, creators=None, creation_op=None, tensor_id=None, name: str = None):
        self.data = np.array(data)
        self.creation_op = creation_op
        self.creators = creators
        self.grad = None
        self.name = name
        self.requires_grad = requires_grad
        self.children = {}
        self.id = tensor_id if tensor_id is not None else id(self)
        self._is_param = False

        if GRAPH_FLAG:
            G.add_node(self.id, label=self.short_repr(), **TENSOR_GRAPH_ATTR)
            if creation_op is not None:
                G.add_node(id(creation_op), label=creation_op.__repr__(), **OP_GRAPH_ATTR)
                G.add_edge(id(creation_op), self.id)

        # corretion the number of children
        if creators is not None:
            for c in creators:
                if GRAPH_FLAG:
                    G.add_edge(c.id, id(creation_op))
                if self.id not in c.children:
                    c.children[self.id] = 1
                else:
                    c.children[self.id] += 1
        # TENSOR_DICT[self.id] = self

    def all_children_grads_accounted_for(self):
        # tensor gets grads from all children?
        for id, cnt in self.children.items():
            if cnt != 0:
                return False
        return True

    def backward(self, gradient=None, grad_origin=None):
        if self.requires_grad:
            if grad_origin is not None:

                # backward is possible? if yes counter -= 1
                if self.children[grad_origin.id] == 0:
                    raise Exception("cannot backprop more than once")
                else:
                    self.children[grad_origin.id] -= 1

            # accumulation of grads from several children

            if self.grad is None:
                # print(f'gradient:{gradient}')
                self.grad = gradient
            else:
                self.grad.data += gradient.data

            if self.creators is not None and (self.all_children_grads_accounted_for() or grad_origin is None):
                self.creation_op.backward(self.grad, self)

    @property
    def shape(self):
        return self.data.shape

    def __len__(self):
        return self.data.__len__()

    def __getitem__(self, idx):
        return self.data.__getitem__(idx)

    def __repr__(self):
        return str(f'Tensor({self.data})')

    def __str__(self):
        return str(f'Tensor({self.data})')

    def short_repr(self):
        if self.name is not None:
            return str(f'{self.name}: Tensor({self.data.shape})')
        return str(f'Tensor({self.data.shape})')

    def numpy(self):
        return self.data

    def numel(self):
        return self.data.size


class Parameter(Tensor):
    r"""
    A kind of Tensor that is to be considered a module parameter.
    """

    def __init__(self, data=None, requires_grad=True, name=None):
        super().__init__(data, requires_grad=requires_grad, name=name)
        self._is_param = True

        self.optm_state = {}

        if GRAPH_FLAG:
            G.add_node(self.id, label=self.short_repr(), **PARA_GRAPH_ATTR)
        # PARA_DICT[self.id] = self

    def __repr__(self):
        return 'Parameter containing:\n' + super(Parameter, self).__repr__()
