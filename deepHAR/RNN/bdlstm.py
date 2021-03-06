import torch
from sklearn.metrics import f1_score
from torch import nn
import numpy as np
from deepHAR.data.data_reader import loadingDB

model_name = 'DB'
DB = 79
BATCH_SIZE = 100
NUM_FILTERS = 64
NUM_HIDDEN = 64
NUM_CLASS = 18
FILTER_SIZE = 5
LEARNING_RATE = 0.001
LAYERSIZE = 64
DROPOUT = 0.5

X_train, X_valid, X_test, y_train, y_valid, y_test = loadingDB('/data/', DB)

data_train = np.concatenate((X_train, np.reshape(y_train, (-1, 1))), axis=1)
data_valid = np.concatenate((X_valid, np.reshape(y_valid, (-1, 1))), axis=1)
data_test = np.concatenate((X_test, np.reshape(y_test, (-1, 1))), axis=1)

NB_SENSOR_CHANNELS = X_test.shape[1]
NUM_CLASSES = np.max(y_train) + 1
win_len = 100

train_on_gpu = torch.cuda.is_available()


def windowing(data_all, nolap=0.5, win_len=100):
    temp = np.int32(1 / nolap)
    dimension_Xy = data_all.shape[1]
    trial_len = win_len * (data_all.shape[0] / win_len) - win_len
    print(trial_len)
    shift = np.int32(win_len * nolap)
    data = np.empty((0, dimension_Xy), dtype='float32')

    for i in range(temp):
        start_idx = 0 + shift * i
        end_idx = trial_len + shift * i
        data_temp = data_all[start_idx:end_idx, :]
        data = np.concatenate((data, data_temp), axis=0)

    data_segs = np.reshape(data, (-1, win_len, dimension_Xy))

    return data_segs[:, :, 0:-1], label_extraction(data_segs[:, :, -1])


def label_extraction(label_mat, threshold=win_len * 0.5):
    a = np.int32(label_mat)
    out = np.zeros(a.shape[0], dtype='int32')
    for i in range(a.shape[0]):
        b = np.bincount(a[i, :])
        idx = b <= threshold
        b[idx] = 0
        b = b[1:]
        if len(b) != 0 and sum(b) != 0:
            out[i] = np.argmax(b) + 1
    return out


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


class RNN(nn.Module):

    def __init__(self, n_hidden=NUM_HIDDEN, n_layers=1, n_filters=NUM_FILTERS, n_classes=NUM_CLASSES, drop_prob=0.5):
        super(RNN, self).__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_filters = n_filters
        self.n_classes = n_classes

        self.rnn = nn.LSTM(
            input_size=NB_SENSOR_CHANNELS,
            hidden_size=NUM_HIDDEN,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.out = nn.Linear(NUM_HIDDEN, NUM_CLASS)

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)
        out = self.out(r_out[:, -1, :])
        return out

    def init_hidden(self):
        weight = next(self.parameters()).data

        if (train_on_gpu):
            hidden = (weight.new(self.n_layers, BATCH_SIZE, self.n_hidden).cuda(),
                      weight.new(self.n_layers, BATCH_SIZE, self.n_hidden).cuda())
        else:
            hidden = (weight.new(self.n_layers, BATCH_SIZE, self.n_hidden),
                      weight.new(self.n_layers, BATCH_SIZE, self.n_hidden))

        return hidden


rnn = RNN()
optimizer = torch.optim.Adam(rnn.parameters(), lr=LEARNING_RATE)
loss_func = nn.CrossEntropyLoss()

num_epochs = 100

results = np.zeros(num_epochs)

for epoch in range(num_epochs):
    h = rnn.init_hidden()
    train_losses = []
    rnn.train()
    for batch in iterate_minibatches(X_train, y_train, BATCH_SIZE, True):
        x, y = batch

        inputs, targets = torch.from_numpy(x), torch.from_numpy(y)

        if train_on_gpu:
            inputs, targets = inputs.cuda(), targets.cuda()

        h = tuple([each.data for each in h])

        optimizer.zero_grad()

        output, h = rnn(inputs, h)

        loss = loss_func(output, targets.long())
        train_losses.append(loss.item())
        loss.backward()
        optimizer.step()
    rnn.train()

    val_losses = []
    val_h = rnn.init_hidden()
    accuracy = 0
    f1score = 0

    with torch.no_grad():
        count_test = 0
        for batch in iterate_minibatches(X_test, y_test, BATCH_SIZE):
            x, y = batch

            inputs, targets = torch.from_numpy(x), torch.from_numpy(y)

            if train_on_gpu:
                inputs, targets = inputs.cuda(), targets.cuda()

            output, val_h = rnn(inputs, val_h)

            val_loss = loss_func(output, targets.long())
            val_losses.append(val_loss.item())

            top_p, top_class = output.topk(1, dim=1)
            equals = top_class == targets.view(*top_class.shape).long()
            accuracy += torch.mean(equals.type(torch.FloatTensor))
            f1score += f1_score(top_class.cpu(), targets.view(*top_class.shape).long().cpu(),
                                average='weighted')

    print("Epoch: {}/{}...".format(epoch + 1, num_epochs),
          "Train Loss: {:.4f}...".format(np.mean(train_losses)),
          "Val Loss: {:.4f}...".format(np.mean(val_losses)),
          "Val Acc: {:.4f}...".format(accuracy / (len(X_test) // BATCH_SIZE)),
          "F1-Score: {:.4f}...".format(f1score / (len(X_test) // BATCH_SIZE)))
