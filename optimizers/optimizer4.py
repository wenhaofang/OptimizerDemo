# RMSProp

import torch

class RMSProp():
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
        lr, eps, gamma = self.option['lr'], self.option['eps'], self.option['gamma']

        for param, state in zip(self.params, self.states):
            state.data = gamma * state.data + (1 - gamma) * (param.data ** 2)
            param.data = param.data - lr * param.grad.data / torch.sqrt(state + eps)

def get_optimizer(params, option):
    return RMSProp(params, {
        'lr': option.learning_rate, 'eps': option.eps, 'gamma': option.gamma
    })
