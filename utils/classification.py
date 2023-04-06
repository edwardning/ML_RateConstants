import os
import time
import torch
from torch import nn
from torch.utils import data


def load_data(data_path=r'../data/database.npy', batch_size=128, train_proportion=0.8):
    torch.manual_seed(0)
    npy_data = torch.load(data_path)
    features = torch.tensor(npy_data['rxnfp'], dtype=torch.float32)
    labels = torch.tensor(npy_data['rc']-1, dtype=torch.long)

    dataset = data.TensorDataset(features, labels)

    num_train = int(train_proportion * len(features))
    num_test = len(features) - num_train

    # print(num_train, num_test)
    train_dataset, test_dataset = data.random_split(dataset, [num_train, num_test])
    return (data.DataLoader(train_dataset, batch_size, shuffle=True),
            data.DataLoader(test_dataset, batch_size, shuffle=True))


def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


def evaluate_accuracy(net, data_iter):  # @save
    if isinstance(net, torch.nn.Module):
        net.eval()
    metric = Accumulator(3)
    loss_test = nn.CrossEntropyLoss(reduction='none')
    with torch.no_grad():
        for X, y in data_iter:
            y_hat = net(X)
            l = loss_test(y_hat, y)
            metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]


class Accumulator:  # @save
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def train(net, train_iter, loss, updater):
    if isinstance(net, torch.nn.Module):
        net.train()
    metric = Accumulator(3)

    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())

    return metric[0] / metric[2], metric[1] / metric[2]


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(),
                                 # nn.Linear(256, 64),
                                 # nn.ReLU(),
                                 nn.Linear(256, 9))

    def forward(self, X):
        return self.net(X)


if __name__ == '__main__':
    t0 = time.time()
    model = MLP()
    batch_size, lr, num_epochs = 8, 0.001, 1000
    loss = nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_iter, test_iter = load_data(data_path=r'../data/database.npy', batch_size=batch_size)

    # print(len(train_iter), len(test_iter))
    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_iter, loss, optimizer)
        test_loss, test_acc = evaluate_accuracy(model, test_iter)

        if epoch % 5 == 0:
            print('{}\t{:.4f}\t{:.2%}\t{:.4f}\t{:.2%}'.
                  format(epoch + 1, train_loss, train_acc, test_loss, test_acc))

    # torch.save(model.state_dict(), 'classify.npy')
    print('Time cost: {:.2f}s'.format(time.time() - t0))
