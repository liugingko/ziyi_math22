# -*- Coding: UTF-8 -*-
# Author:    Time:2022/10/08
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import time
import math
from matplotlib import pyplot
from sklearn.model_selection import train_test_split

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
torch.manual_seed(0)
np.random.seed(0)

# This concept is also called teacher forceing.
# The flag decides if the loss will be calculted over all
# or just the predicted values.
calculate_loss_over_all_values = False

input_window = 118
output_window = 21
batch_size = 4  # batch size
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_data1():

    data = pd.read_excel('src/附件3、土壤湿度2022—2012年.xls')
    # data = pd.concat(data, ignore_index=True)

    data = data.dropna()
    # print(data.columns)
    data = data.loc[:, '10cm湿度(kg/m2)':'200cm湿度(kg/m2)']
    max1 = max(data.max())
    print('max1=', max1)
    data = data/max1

    data_arr10 = list(data['10cm湿度(kg/m2)'])[::-1]
    data_arr40 = list(data['40cm湿度(kg/m2)'])[::-1]
    data_arr100 = list(data['100cm湿度(kg/m2)'])[::-1]
    data_arr200 = list(data['200cm湿度(kg/m2)'])[::-1]
    data_arr = np.array([data_arr10,
                         data_arr40,
                         data_arr100])

    data_arr2  = np.array([data_arr10,
                         data_arr40,
                         data_arr100,
                         data_arr200])

    print(data_arr.shape)


    train_seq = torch.from_numpy(np.array(data_arr[:, :-output_window]))

    train_label = torch.from_numpy(np.array(data_arr[:, output_window:]))

    train_label = torch.tensor(train_label, dtype=torch.float)

    padding = torch.from_numpy(np.zeros([3, 102]))
    padding = torch.tensor(padding, dtype=torch.float)

    data, paddata = train_label.to(device), padding.to(device)

    target = torch.stack(torch.stack([item for item in data]).chunk(input_window, 1))


    print('target, padding, max1', target.shape, padding.shape)

    return target, padding, max1

def get_data():

    data = pd.read_excel('src/附件3、土壤湿度2022—2012年.xls')
    # data = pd.concat(data, ignore_index=True)

    data = data.dropna()
    # print(data.columns)
    data = data.loc[:, '10cm湿度(kg/m2)':'200cm湿度(kg/m2)']
    max1 = max(data.max())
    print('max1=', max1)
    data = data/max1

    data_arr10 = list(data['10cm湿度(kg/m2)'])[::-1]
    data_arr40 = list(data['40cm湿度(kg/m2)'])[::-1]
    data_arr100 = list(data['100cm湿度(kg/m2)'])[::-1]
    data_arr200 = list(data['200cm湿度(kg/m2)'])[::-1]
    data_arr = np.array([data_arr10,
                         data_arr40,
                         data_arr100,
                         data_arr200])

    print(data_arr.shape)


    train_seq = torch.from_numpy(np.array(data_arr[:, :-output_window]))
    train_label = torch.from_numpy(np.array(data_arr[:, output_window:]))
    train_padding = torch.from_numpy(np.zeros(train_label.shape))

    # print(train_seq.shape)
    # print(train_label.shape)
    # print(train_padding.shape)

    train_sequence = torch.stack((train_seq, train_label, train_padding), dim=1).type(torch.FloatTensor)
    # print(train_sequence.shape)

    return train_sequence.to(device), train_sequence.to(device), max1


def get_batch(source, i, batch_size):
    seq_len = min(batch_size, len(source) - 1 - i)
    print('====000', seq_len, len(source))
    # print(i, i + seq_len)
    data = source[i:i + seq_len]
    print('data.shape', data.shape)
    # seq_len = min(batch_size, len(source) - 1 - i)
    # data = source[i:i+seq_len]

    input = torch.stack(torch.stack([item[0] for item in data]).chunk(input_window, 1))  # 1 is feature size
    target = torch.stack(torch.stack([item[1] for item in data]).chunk(input_window, 1))
    padding = torch.stack(torch.stack([item[2] for item in data]).chunk(input_window, 1))
    padding = padding.squeeze(2).transpose(0, 1)

    return input, target, padding

#### positional encoding ####
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=118):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


#### model stracture ####
class TransAm(nn.Module):
    def __init__(self, feature_size=512, num_layers=1, dropout=0):
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=8, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(feature_size, 1)
        self.init_weights()
        self.src_key_padding_mask = None

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_padding):
        # if self.src_mask is None or self.src_mask.size(0) != len(src):
        #     device = src.device
        #     mask = self._generate_square_subsequent_mask(len(src)).to(device)
        #     self.src_mask = mask
        if self.src_key_padding_mask is None:
            mask_key = src_padding.bool()
            self.src_key_padding_mask = mask_key

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask, self.src_key_padding_mask)  # , self.src_mask)
        output = self.decoder(output)
        return output


def train(train_data):
    model.train()  # Turn on the train mode
    total_loss = 0.
    start_time = time.time()

    for batch, i in enumerate(range(0, len(train_data) - 1, batch_size)):
        data, targets, key_padding_mask = get_batch(train_data, i, batch_size)
        optimizer.zero_grad()
        output = model(data, key_padding_mask)

        if calculate_loss_over_all_values:
            loss = criterion(output, targets)
        else:
            loss = criterion(output[-output_window:], targets[-output_window:])

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        log_interval = int(len(train_data) / batch_size / 5) +1
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.6f} | {:5.2f} ms | '
                  'loss {:5.5f}'.format(
                epoch, batch, len(train_data) // batch_size, scheduler.get_last_lr()[0],
                              elapsed * 1000 / log_interval,
                cur_loss))  # , math.exp(cur_loss)
            total_loss = 0
            start_time = time.time()


def plot_and_loss(eval_model, data_source, epoch):
    eval_model.eval()
    total_loss = 0.
    eval_batch_size = 50
    test_result = torch.Tensor(0)
    truth = torch.Tensor(0)
    test_result1 = torch.Tensor(0)
    truth1 = torch.Tensor(0)
    with torch.no_grad():
        for i in range(0, len(data_source) - 1, eval_batch_size):
            data, target, key_padding_mask = get_batch(data_source, i, eval_batch_size)
            # look like the model returns static values for the output window
            output = eval_model(data, key_padding_mask)
            if calculate_loss_over_all_values:
                total_loss += criterion(output, target).item()
            else:
                total_loss += criterion(output[-output_window:], target[-output_window:]).item()

            test_result = torch.cat((test_result, output[-1].squeeze(1).view(-1).cpu() * max1),
                                    0)  # todo: check this. -> looks good to me
            truth = torch.cat((truth, target[-1].squeeze(1).view(-1).cpu() * max1), 0)
            test_result1 = torch.cat((test_result1, output[-5:].squeeze(2).transpose(0, 1).cpu() * max1),
                                     0)  # todo: check this. -> looks good to me
            # test_result1 = pd.concat([test_result1,pd.DataFrame(output[-5:].squeeze(1).view(-1).cpu())], axis=0)
            # truth1 = pd.concat([truth1, pd.DataFrame(target[-5:].squeeze(1).view(-1).cpu())], axis=0)
            truth1 = torch.cat((truth1, target[-5:].squeeze(2).transpose(0, 1).cpu() * max1), 0)
    # test_result = test_result.cpu().numpy()
    # pyplot.plot(test_result[-output_window:], color="red")
    # pyplot.plot(truth[-output_window:], color="blue")
    # pyplot.plot(test_result[-output_window:] - truth[-output_window:], color="green")
    #     print(test_result)
    #     print(truth)
    pyplot.plot(torch.round(truth), color="blue", alpha=0.5)
    pyplot.plot(torch.round(test_result), color="red", alpha=0.5)
    pyplot.plot(torch.round(test_result - truth), color="green", alpha=0.8)
    pyplot.grid(True, which='both')
    pyplot.axhline(y=0, color='k')
    # pyplot.ylim((-2, 2))
    pyplot.savefig('epo%d.png' % epoch)
    pyplot.close()

    return total_loss / i, test_result, truth, torch.round(test_result1), torch.round(truth1)



def evaluate(eval_model, data_source):
    eval_model.eval()  # Turn on the evaluation mode
    total_loss = 0.
    eval_batch_size = 4
    with torch.no_grad():
        for i in range(0, len(data_source) - 1, eval_batch_size):
            data, targets, key_padding_mask = get_batch(data_source, i, eval_batch_size)
            print('============', data.shape, key_padding_mask.shape)
            output = eval_model(data, key_padding_mask)
            print('000', output.shape)
            if calculate_loss_over_all_values:
                total_loss += len(data[0]) * criterion(output, targets).cpu().item()
            else:
                total_loss += len(data[0]) * criterion(output[-output_window:], targets[-output_window:]).cpu().item()
    return total_loss / len(data_source)


train_data, val_data, max1 = get_data()
model = TransAm().to(device)

criterion = nn.MSELoss()
lr = 0.00001
# optimizer = torch.optim.SGD(model.parameters(), lr=lr)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 3, gamma=0.96)

best_val_loss = float("inf")
epochs = 2  # The number of epochs
best_model = None

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train(train_data)
    train_loss = evaluate(model, train_data)

    if (epoch % 5 == 0):
        val_loss, tran_output, tran_true, tran_output5, tran_true5 = plot_and_loss(model, val_data, epoch)
        # predict_future(model, val_data, 200)
    else:
        val_loss = evaluate(model, val_data)

    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.5f} | train loss {:5.5f} '.format(
        epoch, (time.time() - epoch_start_time),
        val_loss, train_loss))  # , math.exp(val_loss) | valid ppl {:8.2f}
    print('-' * 89)
    scheduler.step()

# src = torch.rand(input_window, batch_size, 1) # (source sequence length,batch size,feature number)
# out = model(src)
#
# print(out)
# print(out.shape)

# t1, t2, t3 = get_batch(train_data, 0, 1)
# out = model(t1, t3)
data, key_padding_mask, max1 = get_data1()
model.eval()
print(data.shape, key_padding_mask.shape)

output = model(data, key_padding_mask)

tmp = []
for i in range(3):
    for item in output[:, i, :].detach().numpy().tolist()[:-21]:
        tmp.append(round(item[0] * max1, 2))
    print(tmp)
    tmp = []