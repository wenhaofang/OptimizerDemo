import torch

from torchvision import datasets
from torchvision import transforms

from torch.utils.data import DataLoader

def get_loader(option):
    train_dataset = datasets.MNIST(
        root = option.data_folder,
        transform = transforms.Compose([
            transforms.ToTensor()
        ]),
        train = True,
        download = True
    )
    valid_dataset = datasets.MNIST(
        root = option.data_folder,
        transform = transforms.Compose([
            transforms.ToTensor()
        ]),
        train = False,
        download = True
    )
    train_loader = DataLoader(train_dataset, batch_size = option.batch_size, shuffle = True)
    valid_loader = DataLoader(valid_dataset, batch_size = option.batch_size, shuffle = True)
    return train_loader, valid_loader

if __name__ == '__main__':
    from utils.parser import get_parser

    parser = get_parser()
    option = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    train_loader, valid_loader = get_loader(option)

    print(len(train_loader.dataset)) # 60000
    print(len(valid_loader.dataset)) # 10000

    for mini_batch in train_loader:
        mini_batch = [data_item.to(device) for data_item in mini_batch]
        source, target = mini_batch
        print(source.shape) # (batch_size, 1, 28, 28)
        print(target.shape) # (batch_size)
        break
