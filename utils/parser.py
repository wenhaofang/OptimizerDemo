import argparse

def get_parser():
    parser = argparse.ArgumentParser()

    # For Basic
    parser.add_argument('--name', default = 'main', help = '')

    # For Loader
    parser.add_argument('--data_folder', default = '.data', help = '')

    # For Module
    parser.add_argument('--features', type = int, nargs = '+', default = [784, 250, 100, 10], help = '')
    parser.add_argument('--bias', type = bool, nargs = '+', default = True, help = '')
    parser.add_argument('--activate', choices = ['relu', 'tanh', 'sigmoid'], nargs = '+', default = 'relu', help = '')

    # For Train
    parser.add_argument('--batch_size', type = int, default = 64, help = '')
    parser.add_argument('--num_epochs', type = int, default = 10, help = '')

    return parser

if __name__ == '__main__':
    parser = get_parser()
    option = parser.parse_args()
