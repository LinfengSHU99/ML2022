# %%

import json
import torch
import os
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# %%


# %%
metadata = json.load(open(os.path.join('Dataset', 'metadata.json'), 'r'))
testdata = json.load(open(os.path.join('Dataset', 'testdata.json', ), 'r'))
mapping = json.load(open(os.path.join('Dataset', 'mapping.json', ), 'r'))
n_speakers = len(mapping['speaker2id'])
min_d = min([min([x['mel_len'] for x in metadata['speakers'][id]]) for id in metadata['speakers']])
print(min_d)
# print(len(metadata['speakers']))
# for id in metadata['speakers']:
#     print(id)
n = sum([len(metadata['speakers'][id]) for id in metadata['speakers']])
d_model = metadata['n_mels']
train_index = torch.randperm(n)[:int(n * 0.9)].tolist()


class MyDataset(Dataset):

    def __init__(self, mode='train', segment_len=128):
        super().__init__()
        self.mode = mode
        self.segment_len = 128
        self.y = None
        self.data = []
        if self.mode == 'train' or self.mode == 'validate':
            speakers = metadata['speakers']
            for speaker in speakers.keys():
                for path in speakers[speaker]:

                    self.data.append([path, speaker])
                print(len(self.data), n)
            if self.mode == 'train':
                self.data = [self.data[x] for x in train_index]
            else:
                self.data = [self.data[x] for x in range(n) if x not in train_index]
        else:
            self.data = testdata['utterances']
        # print(self.data[0][1])
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # print(os.path.join('Dataset', self.data[index][0]['feature_path']))
        x = torch.load(os.path.join('Dataset', self.data[index][0]['feature_path']))
        y = mapping['speaker2id'][self.data[index][1]]
        # if x.shape[0] >
        if self.mode == 'test':
            return x
        else:
            return x, y

train_data = DataLoader(MyDataset('train'),  shuffle=True, batch_size=256)
validation_data = DataLoader(MyDataset('validate'),  shuffle=False, batch_size=256)
test_data = DataLoader(MyDataset('test'),  shuffle=False, batch_size=256)


class MyNet(nn.Module):
    def __init__(self, d_model=d_model):
        super().__init__()
        self.encoder = nn.TransformerEncoderLayer(d_model=d_model, dropout=0.1, batch_first=True, nhead=8)
        self.fc = nn.Linear(d_model, n_speakers)

    def forward(self, x):
        x = self.encoder(x)
        x = x.mean(dim=1)
        x = nn.ReLU(x)
        return self.fc(x)


config = {'d_moedl': d_model, 'epoch': 10, 'lr': 0.01}

net = MyNet(d_model)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=config['lr'])

def train():
    net.train()
    for i in range(config['epoch']):
        for x, y in train_data:
            optimizer.zero_grad()
            pred = net(x)
            loss_v = loss(pred, y)
            loss_v.backward()
            optimizer.step()
            print('loss = ', loss_v)


train()