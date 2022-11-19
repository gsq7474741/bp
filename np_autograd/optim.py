from abc import abstractmethod
from typing import List

import numpy as np

from np_autograd import Parameter


class Optimizer:
    def __init__(self, params: List[Parameter], lr: float = 0.001):
        self.paras: List[Parameter] = params
        self.learning_rate = lr

    @abstractmethod
    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        for para in self.paras:
            para.grad.data *= 0.


class SGD(Optimizer):
    def __init__(self, paras, params: List[Parameter], lr=0.01, momentum=0):
        super().__init__(params, lr)
        self.momentum: float = momentum

    def step(self):
        for para in self.paras:
            original_shape = para.data.shape
            # If not initialized
            if 'w_updt' not in para.optm_state:
                para.optm_state['w_updt'] = np.zeros_like(para.data)

            # Use momentum if set
            para.optm_state['w_updt'] = self.momentum * para.optm_state['w_updt'] + (1 - self.momentum) * para.grad.data
            # Move against the gradient to minimize loss
            tmp = para.data - self.learning_rate * para.optm_state['w_updt']

            current_shape = tmp.shape

            if original_shape != current_shape:
                raise ValueError(f'para.data.shape: {current_shape} != original_shape: {original_shape}')
            else:
                para.data = tmp


class Adam(Optimizer):
    def __init__(self, params: List[Parameter], lr=0.001, b1=0.9, b2=0.999):
        super().__init__(params, lr)
        self.eps: float = 1e-8
        # Decay rates
        self.b1 = b1
        self.b2 = b2

    def step(self):
        for para in self.paras:
            original_shape = para.data.shape

            if 'w_updt' not in para.optm_state:
                para.optm_state['w_updt'] = np.zeros_like(para.data)

            if 'm' not in para.optm_state:
                para.optm_state['m'] = np.zeros(np.shape(para.grad.data))
                para.optm_state['v'] = np.zeros(np.shape(para.grad.data))

            para.optm_state['m'] = self.b1 * para.optm_state['m'] + (1 - self.b1) * para.grad.data
            para.optm_state['v'] = self.b2 * para.optm_state['v'] + (1 - self.b2) * np.power(para.grad.data, 2)

            m_hat = para.optm_state['m'] / (1 - self.b1)
            v_hat = para.optm_state['v'] / (1 - self.b2)

            para.optm_state['w_updt'] = self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)

            tmp = para.data - para.optm_state['w_updt']

            current_shape = tmp.shape

            if original_shape != current_shape:
                raise ValueError(f'para.data.shape: {current_shape} != original_shape: {original_shape}')
            else:
                para.data = tmp
