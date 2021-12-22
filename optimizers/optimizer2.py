# SGDM

import torch

class SGDM():
    def __init__(self, params, option):
        self.params = list(params)
        self.option = dict(option)
        self.states = self.init_states(self.params)

    def init_states(self, params):
        return [torch.zeros_like(param) for param in params]

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.data.zero_()

    def step(self):
        lr, momentum = self.option['lr'], self.option['momentum']

        for param, state in zip(self.params, self.states):
            state.data = momentum * state.data + lr * param.grad.data
            param.data = param.data - state.data

def get_optimizer(params, option):
    return SGDM(params, {
        'lr': option.learning_rate, 'momentum': option.momentum
    })
