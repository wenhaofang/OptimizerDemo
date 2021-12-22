# Adam

import torch

class Adam():
    def __init__(self, params, option):
        self.params = list(params)
        self.option = dict(option, ** {'t' : 1})
        self.states = self.init_states(self.params)

    def init_states(self, params):
        return [(
            torch.zeros_like(param), torch.zeros_like(param)
        ) for param in params]

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.data.zero_()

    def step(self):
        lr, beta1, beta2, eps = self.option['lr'], self.option['beta1'], self.option['beta2'], self.option['eps']

        for param, (v, s) in zip(self.params, self.states):
            v = beta1 * v + (1 - beta1) * param.grad.data
            s = beta2 * s + (1 - beta2) * param.grad.data ** 2
            v_bias_corr = v / (1 - beta1 ** self.option['t'])
            s_bias_corr = s / (1 - beta2 ** self.option['t'])
            param.data -= lr * v_bias_corr / (torch.sqrt(s_bias_corr) + eps)

        self.option['t'] += 1

def get_optimizer(params, option):
    return Adam(params, {
        'lr': option.learning_rate, 'beta1': option.beta1, 'beta2': option.beta2, 'eps': option.eps
    })
