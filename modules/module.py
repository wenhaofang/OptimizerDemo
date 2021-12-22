import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, layers_option):
        super(MLP, self).__init__()

        # todo: the last layer has no activation function
        self.layers = nn.Sequential(*[
            item for sublist in [self.get_layer(option) for option in layers_option] for item in sublist
        ])

    def get_layer(self, option):
        if (
            not isinstance(option, (list, tuple)) or len(option) != 4
        ):
            raise ValueError('Invalid Option')

        if option[3].lower() in ['relu', 'tanh', 'sigmoid']:
            return nn.Sequential(
                nn.Linear(option[0], option[1], option[2]),
                nn.ReLU() if option[3] == 'relu' else
                nn.Tanh() if option[3] == 'tanh' else
                nn.Sigmoid()
            )
        else:
            return nn.Sequential(
                nn.Linear(option[0], option[1], option[2])
            )

    def forward(self, x):
        return self.layers(x)

def get_module(option):
    layers_option = []

    for layer_idx in range(len(option.features) - 1):
        i_features = option.features[layer_idx]
        o_features = option.features[layer_idx + 1]

        if isinstance(option.bias, (list, tuple)):
            if layer_idx < len(option.bias):
                bias = option.bias[layer_idx]
            else:
                bias = option.bias[-1]
        else:
            bias = option.bias

        if isinstance(option.activate, (list, tuple)):
            if layer_idx < len(option.activate):
                activate = option.activate[layer_idx]
            else:
                activate = option.activate[-1]
        else:
            activate = option.activate

        layers_option.append((i_features, o_features, bias, activate))

    return MLP(layers_option)

if __name__ == '__main__':
    from utils.parser import get_parser

    parser = get_parser()
    option = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    module = get_module(option)
    print(module)
