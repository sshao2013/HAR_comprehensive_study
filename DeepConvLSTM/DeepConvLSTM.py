import argparse
import parser
import pickle as cp

from sklearn import metrics
from sliding_window import sliding_window
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.dataset as Dataset
import torch.utils.data.dataloader as DataLoader

# From source
NB_SENSOR_CHANNELS = 113
NUM_CLASSES = 18
SLIDING_WINDOW_LENGTH = 24
FINAL_SEQUENCE_LENGTH = 8
SLIDING_WINDOW_STEP = 12
BATCH_SIZE = 100
NUM_FILTERS = 64
FILTER_SIZE = 5
NUM_UNITS_LSTM = 128

LEARNING_RAET = 0.01
EPOCHS = 20

# parser = argparse.ArgumentParser()
# args = parser.parse_args()
# parser.add_argument('--gpu', default = 1, type = int, help = 'gpu id')
# device = torch.device('cuda:' + str(args.gpu))
# torch.cuda.set_device(device)
# nvidia-smi

import warnings

warnings.filterwarnings('ignore')


def load_dataset(filename):
    f = open(filename, 'rb')
    data = cp.load(f)
    f.close()

    X_train, y_train = data[0]
    X_test, y_test = data[1]

    print(" ..from file {}".format(filename))
    print(" ..reading instances: train {0}, test {1}".format(X_train.shape, X_test.shape))

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    # The targets are casted to int8 for GPU compatibility.
    y_train = y_train.astype(np.uint8)
    y_test = y_test.astype(np.uint8)

    return X_train, y_train, X_test, y_test


def opp_sliding_window(data_x, data_y, ws, ss):
    data_x = sliding_window(data_x, (ws, data_x.shape[1]), (ss, 1))
    data_y = np.asarray([[i[-1]] for i in sliding_window(data_y, ws, ss)])
    return data_x.astype(np.float32), data_y.reshape(len(data_y)).astype(np.uint8)


class Net(nn.Module):
    def __init__(self, n_hidden=NUM_UNITS_LSTM, n_layers=1, n_filters=NUM_FILTERS,
                 n_classes=NUM_CLASSES, filter_size=5, drop_prob=0.5):
        super(Net, self).__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_filters = n_filters
        self.n_classes = n_classes
        self.filter_size = filter_size

        self.conv1 = nn.Conv1d(NB_SENSOR_CHANNELS, n_filters, self.filter_size)
        self.conv2 = nn.Conv1d(n_filters, n_filters, self.filter_size)
        self.conv3 = nn.Conv1d(n_filters, n_filters, self.filter_size)
        self.conv4 = nn.Conv1d(n_filters, n_filters, self.filter_size)

        self.lstm1 = nn.LSTM(input_size=n_filters, hidden_size=n_hidden, num_layers=1)
        self.lstm2 = nn.LSTM(input_size=n_hidden, hidden_size=n_hidden, num_layers=1)
        self.out = nn.Linear(n_hidden, n_classes)

        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x, hidden):

        x = x.view(-1, NB_SENSOR_CHANNELS, SLIDING_WINDOW_LENGTH)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        # x = self.dropout(x)

        x = x.view(8, -1, self.n_filters)
        x, hidden = self.lstm1(x, hidden)

        x, hidden = self.lstm2(x, hidden)

        x = x.contiguous().view(-1, self.n_hidden)
        x = self.dropout(x)
        x = self.out(x)

        out = x.view(BATCH_SIZE, -1, self.n_classes)[:, -1, :]

        return out, hidden

    def init_hidden(self):
        weight = next(self.parameters()).data

        if (train_on_gpu):
            hidden = (weight.new(self.n_layers, BATCH_SIZE, self.n_hidden).zero_().cuda(),
                      weight.new(self.n_layers, BATCH_SIZE, self.n_hidden).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, BATCH_SIZE, self.n_hidden).zero_(),
                      weight.new(self.n_layers, BATCH_SIZE, self.n_hidden).zero_())

        return hidden


def weights_init(m):
    if type(m) == nn.LSTM:
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
    elif type(m) == nn.Conv1d or type(m) == nn.Linear:
        torch.nn.init.orthogonal_(m.weight)
        m.bias.data.fill_(0)


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


if __name__ == '__main__':
    train_on_gpu = torch.cuda.is_available()

    if train_on_gpu:
        torch.cuda.set_device(1)
        print("Training on GPU", torch.cuda.current_device())

    print("Loading data...")
    X_train, y_train, X_test, y_test = load_dataset('oppChallenge_gestures.data')
    assert NB_SENSOR_CHANNELS == X_train.shape[1]

    X_train, y_train = opp_sliding_window(X_train, y_train, SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP)
    X_test, y_test = opp_sliding_window(X_test, y_test, SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP)

    print(" ..after sliding window (testing): inputs {0}, targets {1}".format(X_test.shape, y_test.shape))

    X_train = X_train.reshape((-1, 1, NB_SENSOR_CHANNELS, SLIDING_WINDOW_LENGTH,))
    X_test = X_test.reshape((-1, 1, NB_SENSOR_CHANNELS, SLIDING_WINDOW_LENGTH,))

    print(" ..after sliding and reshaping, train data: inputs {0}, targets {1}".format(X_train.shape, y_train.shape))
    print(" ..after sliding and reshaping, test data : inputs {0}, targets {1}".format(X_test.shape, y_test.shape))

    # y_train = y_train.reshape(len(y_train), 1)
    # y_test = y_test.reshape(len(y_test), 1)

    net = Net()
    net.apply(weights_init)
    print(net)

    if train_on_gpu:
        net.cuda()

    optimizer = torch.optim.SGD(net.parameters(), momentum=0.9, lr=LEARNING_RAET)
    loss_F = torch.nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        h = net.init_hidden()
        train_losses = []
        net.train()
        for batch in iterate_minibatches(X_train, y_train, BATCH_SIZE, True):
            x, y = batch

            inputs, targets = torch.from_numpy(x), torch.from_numpy(y)

            if train_on_gpu:
                inputs, targets = inputs.cuda(), targets.cuda()

            h = tuple([each.data for each in h])

            optimizer.zero_grad()

            output, h = net(inputs, h)

            loss = loss_F(output, targets.long())
            train_losses.append(loss.item())
            loss.backward()
            optimizer.step()
        net.train()

    val_losses = []
    val_h = net.init_hidden()
    accuracy = 0
    f1score = 0

    with torch.no_grad():
        count_test = 0
        for batch in iterate_minibatches(X_test, y_test, BATCH_SIZE):
            x, y = batch

            inputs, targets = torch.from_numpy(x), torch.from_numpy(y)

            if train_on_gpu:
                inputs, targets = inputs.cuda(), targets.cuda()

            output, val_h = net(inputs, val_h)

            val_loss = loss_F(output, targets.long())
            val_losses.append(val_loss.item())

            top_p, top_class = output.topk(1, dim=1)
            equals = top_class == targets.view(*top_class.shape).long()
            accuracy += torch.mean(equals.type(torch.FloatTensor))
            f1score += metrics.f1_score(top_class.cpu(), targets.view(*top_class.shape).long().cpu(),
                                        average='weighted')

    print("Epoch: {}/{}...".format(epoch + 1, EPOCHS),
          "Train Loss: {:.4f}...".format(np.mean(train_losses)),
          "Val Loss: {:.4f}...".format(np.mean(val_losses)),
          "Val Acc: {:.4f}...".format(accuracy / (len(X_test) // BATCH_SIZE)),
          "F1-Score: {:.4f}...".format(f1score / (len(X_test) // BATCH_SIZE)))
