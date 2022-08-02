# %%

import json
import torch
import os
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# %%

# f = open('./Dataset/metadata.json', 'r')
# t = json.load(f)
# print(t['n_mels'])
#
# print(len(t['speakers']['id03074'][0]['feature_path']))
# path = t['speakers']['id03074'][0]['feature_path']
# print(path)
# tensor = torch.load(os.path.join('Dataset', path))
# print(tensor.shape)
transformer = nn.TransformerEncoderLayer(40, 8)
# x = torch.rand([2,123,40])
# y = transformer(x)
# print(y.shape)
# x2 = torch.rand([3,123,40])
# y = transformer(x, x2)
# print(y.shape)
# %%
metadata = json.load(open(os.path.join('Dataset', 'metadata.json', 'r')))
testdata = json.load(open(os.path.join('Dataset', 'testdata.json', 'r')))
mapping = json.load(open(os.path.join('Dataset', 'mapping.json', 'r')))
n = sum([len(id) for id in metadata['speakers']])
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
                self.data.append([speakers[speaker], speaker])
            if self.mode == 'train':
                self.data = [self.data[x] for x in train_index]
            else:
                self.data = [self.data[x] for x in range(n) if x not in train_index]
        else:
            self.data = testdata['utterances']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = torch.load(open(os.path.join('Dataset', self.data[index][0]['feature_path'])))
        y = mapping['speaker2id'][self.data[index][1]]
        # if x.shape[0] >
        if self.mode == 'test':
            return x
        else:
            return x, y
