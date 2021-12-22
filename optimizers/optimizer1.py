# SGD

class SGD():
    def __init__(self, params, option):
        self.params = list(params)
        self.option = dict(option)

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.data.zero_()

    def step(self):
        lr = self.option['lr']

        for param in self.params:
            param.data = param.data - lr * param.grad.data

def get_optimizer(params, option):
    return SGD(params, {
        'lr': option.learning_rate
    })
