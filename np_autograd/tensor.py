import numpy as np
import pygraphviz as pgv
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, overload, Tuple, TypeVar, Union, List
import gol

# from np_autograd import OpBase

TENSOR_GRAPH_ATTR: dict = gol.get_value('TENSOR_GRAPH_ATTR')
PARA_GRAPH_ATTR: dict = gol.get_value('PARA_GRAPH_ATTR')
OP_GRAPH_ATTR: dict = gol.get_value('OP_GRAPH_ATTR')
G: pgv.AGraph = gol.get_value('G')
# TENSOR_DICT: list = gol.get_value('TENSOR_DICT')
# PARA_DICT: list = gol.get_value('PARA_DICT')
# OP_DICT: list = gol.get_value('OP_DICT')
GRAPH_FLAG: bool = gol.get_value('GRAPH_FLAG')


class Tensor:
    def __init__(self, data: Iterable, requires_grad: bool = False, creators: Union['Tensor', List['Tensor']] = None,
                 creation_op: Optional[Any] = None,
                 name: str = None):
        self.data: np.ndarray = np.array(data)
        self.creation_op: ... = creation_op
        self.creators: 'Tensor' = creators
        self.grad: Union['Tensor', np.ndarray, None] = None
        self.name: str = name
        self.requires_grad: bool = requires_grad
        self.children: dict = {}
        self.id = id(self)
        self._is_param: bool = False

        if GRAPH_FLAG:
            G.add_node(self.id, label=self.short_repr(), **TENSOR_GRAPH_ATTR)
            if creation_op is not None:
                G.add_node(id(creation_op), label=creation_op.__repr__(), **OP_GRAPH_ATTR)
                G.add_edge(id(creation_op), self.id)

        if creators is not None:
            for c in creators:
                if GRAPH_FLAG:
                    G.add_edge(c.id, id(creation_op))
                if self.id not in c.children:
                    c.children[self.id] = 1
                else:
                    c.children[self.id] += 1
        # TENSOR_DICT[self.id] = self

    def is_all_children_grads_get(self):
        # tensor gets grads from all children?
        for tensor_id, count in self.children.items():
            if count != 0:
                return False
        return True

    def backward(self, gradient: Optional['Tensor'] = None, grad_origin: Optional['Tensor'] = None) -> None:
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

            if self.creators is not None and (self.is_all_children_grads_get() or grad_origin is None):
                self.creation_op.backward(self.grad, self)

    @property
    def shape(self) -> Tuple[int]:
        return self.data.shape

    def __len__(self) -> int:
        return self.data.__len__()

    def __getitem__(self, idx: Union[int, slice, Iterable]):
        return self.data.__getitem__(idx)

    def __repr__(self) -> str:
        return str(f'Tensor({self.data})')

    def __str__(self) -> str:
        return str(f'Tensor({self.data})')

    def short_repr(self) -> str:
        if self.name is not None:
            return str(f'{self.name}: Tensor({self.data.shape})')
        return str(f'Tensor({self.data.shape})')

    def numpy(self) -> np.ndarray:
        return self.data

    def numel(self) -> int:
        return self.data.size


class Parameter(Tensor):
    r"""
    A kind of Tensor that is to be considered a layer parameter.
    """

    def __init__(self, data: Iterable, requires_grad: bool = True, name: str = None):
        super().__init__(data, requires_grad=requires_grad, name=name)
        self._is_param: bool = True

        self.optm_state: dict = {}

        if GRAPH_FLAG:
            G.add_node(self.id, label=self.short_repr(), **PARA_GRAPH_ATTR)
        # PARA_DICT[self.id] = self

    def __repr__(self) -> str:
        return 'Parameter containing:\n' + super(Parameter, self).__repr__()
