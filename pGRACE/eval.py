from typing import Optional

import torch
from torch.optim import Adam
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score

from pGRACE.model import LogReg, MLP


def get_idx_split(data, split, preload_split):
    if split[:4] == 'rand':
        train_ratio = float(split.split(':')[1])
        num_nodes = data.x.size(0)
        train_size = int(num_nodes * train_ratio)
        indices = torch.randperm(num_nodes)
        return {
            'train': indices[:train_size],
            'val': indices[train_size:2 * train_size],
            'test': indices[2 * train_size:]
        }
    elif split == 'ogb':
        return dataset.get_idx_split()
    elif split.startswith('wikics'):
        split_idx = int(split.split(':')[1])
        return {
            'train': dataset[0].train_mask[:, split_idx],
            'test': dataset[0].test_mask,
            'val': dataset[0].val_mask[:, split_idx]
        }
    elif split == 'preloaded':
        assert preload_split is not None, 'use preloaded split, but preloaded_split is None'
        train_mask, test_mask, val_mask = preload_split
        return {
            'train': train_mask,
            'test': test_mask,
            'val': val_mask
        }
    else:
        raise RuntimeError(f'Unknown split type {split}')


def train_mlp(epochs, device, train_data, train_y, test_data, test_y):
    test_device = device
    ft_in = train_data.shape[1]
    hidden_dim = 32
    nb_classes = int(train_y.max() + 1)
    train_data = torch.from_numpy(train_data).to(test_device)
    train_y = torch.from_numpy(train_y).type(torch.LongTensor).to(test_device)
    test_data = torch.from_numpy(test_data).to(test_device)
    test_y = torch.from_numpy(test_y).to(test_device)

    classifier = MLP(ft_in, hidden_dim, nb_classes).to(test_device)
    optimizer = Adam(classifier.parameters(), lr=0.001, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        classifier.train()
        optimizer.zero_grad()
        output = classifier(train_data)
        loss = criterion(output, train_y)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 20 == 0:
            classifier.eval()
            y_pred = classifier(test_data).argmax(-1).view(-1, 1)
            # print(y_pred)
            auc = roc_auc_score(test_y.detach().cpu().numpy(), y_pred.detach().cpu().numpy())
            # print(epoch+1, loss.item(), auc)

    return y_pred.detach().cpu().numpy()


def log_regression(z,
                   data,
                   evaluator,
                   num_epochs: int = 5000,
                   test_device: Optional[str] = None,
                   split: str = 'rand:0.1',
                   verbose: bool = False,
                   preload_split=None):
    test_device = z.device if test_device is None else test_device
    z = z.detach().to(test_device)
    num_hidden = z.size(1)
    y = data.y.view(-1).to(test_device)
    num_classes = data.y.max().item() + 1
    classifier = LogReg(num_hidden, num_classes).to(test_device)
    optimizer = Adam(classifier.parameters(), lr=0.01, weight_decay=0.0)

    split = get_idx_split(data, split, preload_split)
    split = {k: v.to(test_device) for k, v in split.items()}
    f = nn.LogSoftmax(dim=-1)
    nll_loss = nn.NLLLoss()

    best_test_acc = 0
    best_val_acc = 0
    best_epoch = 0

    for epoch in range(num_epochs):
        classifier.train()
        optimizer.zero_grad()

        output = classifier(z[split['train']])
        loss = nll_loss(f(output), y[split['train']])

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            if 'val' in split:
                # val split is available
                test_acc = evaluator.eval({
                    'y_true': y[split['test']].view(-1, 1),
                    'y_pred': classifier(z[split['test']]).argmax(-1).view(-1, 1)
                })['acc']
                val_acc = evaluator.eval({
                    'y_true': y[split['val']].view(-1, 1),
                    'y_pred': classifier(z[split['val']]).argmax(-1).view(-1, 1)
                })['acc']
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_test_acc = test_acc
                    best_epoch = epoch
            else:
                acc = evaluator.eval({
                    'y_true': y[split['test']].view(-1, 1),
                    'y_pred': classifier(z[split['test']]).argmax(-1).view(-1, 1)
                })['acc']
                if best_test_acc < acc:
                    best_test_acc = acc
                    best_epoch = epoch
            if verbose:
                print(f'logreg epoch {epoch}: best test acc {best_test_acc}')

    return {'acc': best_test_acc}


class MulticlassEvaluator:
    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def _eval(y_true, y_pred):
        y_true = y_true.view(-1)
        y_pred = y_pred.view(-1)
        total = y_true.size(0)
        correct = (y_true == y_pred).to(torch.float32).sum()
        return (correct / total).item()

    def eval(self, res):
        return {'acc': self._eval(**res)}