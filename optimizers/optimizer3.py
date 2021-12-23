# AdaGrad

import torch

class AdaGrad():
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
        lr, eps = self.option['lr'], self.option['eps']

        for param, state in zip(self.params, self.states):
            state.data = state.data + param.grad.data ** 2
            param.data = param.data - lr * param.grad.data / torch.sqrt(state + eps)

def get_optimizer(params, option):
    if option.official:
        return torch.optim.Adagrad(params, lr = option.learning_rate, eps = option.eps)
    else:
        return AdaGrad(params, {
            'lr': option.learning_rate, 'eps': option.eps
        })
