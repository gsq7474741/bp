import warnings
from collections import OrderedDict
from typing import Union, Iterator, Set, Optional, Dict

import numpy as np

import np_autograd.ops as t
from np_autograd.tensor import Tensor, Parameter


class Layer:
    def __init__(self, name: str = None):
        super().__init__()
        self.training = True
        self._parameters: Dict[str, Optional[Parameter]] = OrderedDict()
        self._buffers: Dict[str, Optional[Tensor]] = OrderedDict()
        self._layers: Dict[str, Optional['Layer']] = OrderedDict()
        self._name: str = name

    def forward(self, *inputs):
        raise NotImplementedError

    def train(self, mode: bool = True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            module.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def register_parameter(self, name: str, param: Optional[Parameter]) -> None:
        r"""
        Adds a parameter to the module.
        """
        if '_parameters' not in self.__dict__:
            raise AttributeError(
                "cannot assign parameter before Layer.__init__() call")

        elif not isinstance(name, str):
            raise TypeError("parameter name should be a string. "
                            "Got {}".format(type(name)))
        elif name == '':
            raise KeyError("parameter name can't be empty string \"\"")
        elif hasattr(self, name) and name not in self._parameters:
            raise KeyError(f"attribute '{name}' already exists")

        if param is None:
            self._parameters[name] = None
        elif not isinstance(param, Parameter):
            raise TypeError("cannot assign '{}' object to parameter '{}' "
                            "(Parameter or None required)"
                            .format(type(param), name))

        else:
            self._parameters[name] = param

    def register_buffer(self, name: str, tensor: Optional[Tensor], persistent: bool = True) -> None:
        r"""
        Adds a buffer to the module, a buffer that should not to be considered a model parameter.
        """
        if '_buffers' not in self.__dict__:
            raise AttributeError(
                "cannot assign buffer before Module.__init__() call")
        elif not isinstance(name, str):
            raise TypeError("buffer name should be a string. "
                            "Got {}".format(type(name)))
        elif '.' in name:
            raise KeyError("buffer name can't contain \".\"")
        elif name == '':
            raise KeyError("buffer name can't be empty string \"\"")
        elif hasattr(self, name) and name not in self._buffers:
            raise KeyError("attribute '{}' already exists".format(name))
        elif tensor is not None and not isinstance(tensor, Tensor):
            raise TypeError("cannot assign '{}' object to buffer '{}' "
                            "(Tensor or None required)"
                            .format(type(tensor), name))
        else:
            self._buffers[name] = tensor

    def register_module(self, name: str, module: Optional['Layer']) -> None:
        r"""Alias for :func:`add_module`."""
        self.add_module(name, module)

    def _named_members(self, get_members_fn, prefix='', recurse=True):
        r"""Helper method for yielding various names + members of modules."""
        memo = set()
        modules = self.named_modules(prefix=prefix) if recurse else [(prefix, self)]
        for module_prefix, module in modules:
            members = get_members_fn(module)
            for k, v in members:
                if v is None or v in memo:
                    continue
                memo.add(v)
                name = module_prefix + ('.' if module_prefix else '') + k
                yield name, v

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        gen = self._named_members(
            lambda module: module._parameters.items(),
            prefix='', recurse=recurse)

        for name, param in gen:
            yield param

    def buffers(self, recurse: bool = True) -> Iterator[Tensor]:
        r"""
        Returns an iterator over module buffers.
        """

        gen = self._named_members(
            lambda module: module._buffers.items(),
            prefix='', recurse=recurse)
        for _, buf in gen:
            yield buf

    def children(self) -> Iterator['Layer']:
        r"""
        Returns an iterator over immediate children modules.
        """
        memo = set()
        for name, module in self._layers.items():
            if module is not None and module not in memo:
                memo.add(module)
                yield module

    def named_modules(self, memo: Optional[Set['Layer']] = None, prefix: str = '', remove_duplicate: bool = True):
        r"""Returns an iterator over all modules in the network, yielding
        both the name of the module as well as the module itself.

        Args:
            memo: a memo to store the set of modules already added to the result
            prefix: a prefix that will be added to the name of the module
            remove_duplicate: whether to remove the duplicated module instances in the result
                or not

        Yields:
            (string, Module): Tuple of name and module

        Note:
            Duplicate modules are returned only once. In the following
            example, ``l`` will be returned only once.

        """

        if memo is None:
            memo = set()
        if self not in memo:
            if remove_duplicate:
                memo.add(self)
            yield prefix, self
            for name, module in self._layers.items():
                if module is None:
                    continue
                submodule_prefix = prefix + ('.' if prefix else '') + name
                for m in module.named_modules(memo, submodule_prefix, remove_duplicate):
                    yield m

    def zero_grad(self, set_to_none: bool = False) -> None:
        r"""
        Sets gradients of all model parameters to zero.
        """
        if getattr(self, '_is_replica', False):
            warnings.warn(
                "Calling .zero_grad() from a module created with nn.DataParallel() has no effect. "
                "The parameters are copied (in a differentiable manner) from the original module. "
                "This means they are not leaf nodes in autograd and so don't accumulate gradients. "
                "If you need gradients in your forward method, consider using autograd.grad instead.")

        for p in self.parameters():
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    if p.grad.grad_fn is not None:
                        p.grad.detach_()
                    else:
                        p.grad.requires_grad_(False)
                    p.grad.zero_()

    def __setattr__(self, name: str, value: Union[Tensor, 'Layer']) -> None:
        def remove_from(*dicts_or_sets):
            for d in dicts_or_sets:
                if name in d:
                    if isinstance(d, dict):
                        del d[name]
                    else:
                        d.discard(name)

        params = self.__dict__.get('_parameters')
        if isinstance(value, Parameter):
            if params is None:
                raise AttributeError(
                    "cannot assign parameters before Layer.__init__() call")
            remove_from(self.__dict__, )
            self.register_parameter(name, value)
        elif params is not None and name in params:
            if value is not None:
                raise TypeError("cannot assign '{}' as parameter '{}' "
                                "(Parameter or None expected)"
                                .format(type(value), name))
            self.register_parameter(name, value)
        else:
            layers = self.__dict__.get('_layers')
            if isinstance(value, Layer):
                if layers is None:
                    raise AttributeError(
                        "cannot assign module before Module.__init__() call")
                remove_from(self.__dict__, self._parameters, self._buffers)
                layers[name] = value
            elif layers is not None and name in layers:
                if value is not None:
                    raise TypeError("cannot assign '{}' as child module '{}' "
                                    "(Layer or None expected)"
                                    .format(type(value), name))
                layers[name] = value
            else:
                buffers = self.__dict__.get('_buffers')
                if buffers is not None and name in buffers:
                    if value is not None and not isinstance(value, Tensor):
                        raise TypeError("cannot assign '{}' as buffer '{}' "
                                        "(Tensor or None expected)"
                                        .format(type(value), name))
                    buffers[name] = value
                else:
                    object.__setattr__(self, name, value)

    def __getattr__(self, name: str) -> Union[Tensor, 'Layer']:
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return _parameters[name]
        if '_buffers' in self.__dict__:
            _buffers = self.__dict__['_buffers']
            if name in _buffers:
                return _buffers[name]
        if '_layers' in self.__dict__:
            modules = self.__dict__['_layers']
            if name in modules:
                return modules[name]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, name))

    def __delattr__(self, name):
        if name in self._parameters:
            del self._parameters[name]
        elif name in self._buffers:
            del self._buffers[name]
        elif name in self._layers:
            del self._layers[name]
        else:
            object.__delattr__(self, name)

    def __repr__(self):
        def _addindent(s_, numSpaces):
            s = s_.split('\n')
            # don't do anything for single-line stuff
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(numSpaces * ' ') + line for line in s]
            s = '\n'.join(s)
            s = first + '\n' + s
            return s

        extra_lines = []
        extra_repr = ''
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        for key, layer in self._layers.items():
            mod_str = repr(layer)
            print(mod_str)
            mod_str = _addindent(mod_str, 2)
            child_lines.append('(' + key + '): ' + mod_str)
        lines = extra_lines + child_lines

        main_str = self.__class__.__name__ + '('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        return main_str

    def __call__(self, *args):
        return self.forward(*args)  # forward pass


class Linear(Layer):
    def __init__(self, in_features, out_features, use_bias=True, name: str = None):
        super().__init__()
        self.use_bias = use_bias
        self.name = name if name is not None else self.__class__.__name__

        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(np.empty((in_features, out_features)), name=f'weight in {name}')
        if use_bias:
            self.bias = Parameter(np.empty(out_features), name=f'bias in {name}')
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)).
        self.weight.data = np.random.uniform(-1 / np.sqrt(self.in_features), 1 / np.sqrt(self.in_features),
                                             self.weight.shape)
        if self.bias is not None:
            self.bias.data = np.random.uniform(-1 / np.sqrt(self.in_features), 1 / np.sqrt(self.in_features),
                                               self.bias.shape)

    def forward(self, input: Tensor) -> Tensor:
        return t.Add().forward(t.MatMul().forward(input, self.weight), self.bias)

    def __repr__(self):
        if self.name is not None:
            return f"{self.name}: Linear({self.weight.shape[0]}, {self.weight.shape[1]})"
        return f"Linear({self.weight.shape[0]}, {self.weight.shape[1]})"


class ReLU(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        return t.ReLU().forward(inputs)

    def __repr__(self):
        return "ReLU()"


class Dropout(Layer):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, inputs):
        if self.training:
            # mask = np.random.binomial(1, self.p, size=inputs.shape)
            return t.Dropout(self.p).forward(inputs)
        return inputs

    def __repr__(self):
        return f"Dropout({self.p})"


class Softmax(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, dim=1):
        return t.Softmax().forward(inputs, dim)


class Sigmoid(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        return t.Sigmoid().forward(inputs)

    def __repr__(self):
        return "Sigmoid()"
