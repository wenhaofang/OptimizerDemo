import tqdm
import torch
import numpy as np

from matplotlib import pyplot as plt

def evaluate(true , pred):
    true_min, true_max = true.min(), true.max()
    pred_min, pred_max = pred.min(), pred.max()
    min_val = min(true_min, pred_min)
    max_val = max(true_max, pred_max)
    matrix = np.zeros((max_val - min_val + 1, max_val - min_val + 1))
    for i, j in zip(true , pred):
        matrix[i - true_min][j - pred_min] += 1

    TP = np.diag(matrix)
    FP = np.sum(matrix, axis = 0) - TP
    FN = np.sum(matrix, axis = 1) - TP
    TN = np.sum(matrix) - (FP + FN + TP)

    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        p_i = TP / (TP + FP)
        r_i = TP / (TP + FN)
        f_i = 2 * p_i * r_i / (p_i + r_i)

    np.nan_to_num(p_i, copy = False, nan = 0)
    np.nan_to_num(r_i, copy = False, nan = 0)
    np.nan_to_num(f_i, copy = False, nan = 0)

    p_macro = np.sum(p_i) / len(p_i)
    r_macro = np.sum(r_i) / len(r_i)
    f_macro = 2 * p_macro * r_macro / (p_macro + r_macro)

    p_micro = np.sum(TP) / (np.sum(TP) + np.sum(FP))
    r_micro = np.sum(TP) / (np.sum(TP) + np.sum(FN))
    f_micro = 2 * p_micro * r_micro / (p_micro + r_micro)

    return f_i, f_macro, f_micro

# for regression (todo: fix numerical problem)
def mse_loss(pred, true, reduction = 'mean'):
    if (
        reduction != 'mean' and
        reduction != 'sum'
    ):
        raise ValueError('%s is not a valid value for reduction' % reduction)
    elif reduction == 'mean':
        return ((pred - true.view(pred.shape)) ** 2).mean()
    elif reduction == 'sum':
        return ((pred - true.view(pred.shape)) ** 2).sum()

# for classification (todo: fix numerical problem)
def cross_entropy(pred, true, reduction = 'mean'):
    if (
        reduction != 'mean' and
        reduction != 'sum'
    ):
        raise ValueError('%s is not a valid value for reduction' % reduction)
    elif reduction == 'mean':
        return (-torch.log((
            torch.exp(pred).gather(1, true.unsqueeze(-1)).squeeze(-1) / \
            torch.exp(pred).sum(1)
        ))).mean()
    elif reduction == 'sum':
        return (-torch.log((
            torch.exp(pred).gather(1, true.unsqueeze(-1)).squeeze(-1) / \
            torch.exp(pred).sum(1)
        ))).sum()

def plot_loss(train_loss, valid_loss, save_path):
    epoch = list(range(len(train_loss)))
    plt.plot(epoch, train_loss, color = '#1f77b4', label = 'train_loss')
    plt.plot(epoch, valid_loss, color = '#ff7f0e', label = 'valid_loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(save_path)

def train(loader, module, criterion, optimizer, device):
    module.train()
    total_loss = 0
    for mini_batch in tqdm.tqdm(loader):
        mini_batch = [data_item.to(device) for data_item in mini_batch]
        source , target = mini_batch
        source = source.view(source.shape[0], -1)
        output = module(source)
        output = output.squeeze(1)
        loss = criterion(output, target)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return {
        'loss': total_loss / len(loader)
    }

def valid(loader, module, criterion, optimizer, device):
    module.eval()
    total_loss = 0
    prob_fold = []
    true_fold = []
    with torch.no_grad():
        for mini_batch in tqdm.tqdm(loader):
            mini_batch = [data_item.to(device) for data_item in mini_batch]
            source , target = mini_batch
            source = source.view(source.shape[0], -1)
            output = module(source)
            output = output.squeeze(1)
            loss = criterion(output, target)
            total_loss += loss.item()
            true_fold.append(target)
            prob_fold.append(output)
    true_fold = torch.cat(true_fold).cpu().numpy()
    prob_fold = torch.cat(prob_fold).softmax(dim = -1).cpu().numpy()
    pred_fold = prob_fold.argmax(axis = 1)
    eval_info = evaluate(true_fold, pred_fold)
    return {
        'loss': total_loss / len(loader),
        'macro_f1': eval_info[1],
        'micro_f1': eval_info[2],
    }
