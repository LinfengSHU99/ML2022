import os
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.transforms import transforms

import matplotlib.pyplot as plt
from tqdm import tqdm

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'


def same_seed(seed):
    random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


same_seed(6666)


class ImageData(Dataset):
    def __init__(self):
        # self.compose = transforms.Compose([transforms.ToPILImage(), transforms.Resize((64,64)),
        #                                    transforms.ToTensor(), transforms.Normalize(mean=(0,5,0.5,0.5), std=(0.5,0.5,0.5))])
        self.transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((64, 64)), transforms.ToTensor()])
        self.names = os.listdir('./faces')

    def __getitem__(self, index):
        name = self.names[index]
        img = torchvision.io.read_image(os.path.join('faces', name))
        return self.transform(img)

    def __len__(self):
        return len(self.names)


image_data = ImageData()
# image_list = [image_data[i] for i in range(4)]
# grid_img = torchvision.utils.make_grid(image_list)
# plt.imshow(grid_img.permute(1, 2, 0))
# plt.show()


class Generator(nn.Module):
    def dconv_bn_relu(self, input_dim, output_dim):
        return nn.Sequential(nn.ConvTranspose2d(input_dim, output_dim, kernel_size=5, stride=2, padding=2),
                             nn.BatchNorm2d(output_dim),
                             nn.ReLU())

    def __init__(self, input_dim, output_dim=64):
        super.__init__()
        self.l1 = nn.Sequential(nn.Linear(input_dim, output_dim * 8 * 4 * 4),
                                nn.BatchNorm1d(),
                                nn.ReLU()) # need to be reshaped to outd * 8, 4, 4
        self.l2 = nn.Sequential(self.dconv_bn_relu(output_dim * 8, output_dim * 4), # outd * 4, 7, 7
                                self.dconv_bn_relu(output_dim * 4, output_dim * 2), # outd * 2, 10, 10
                                self.dconv_bn_relu(output_dim * 2, output_dim))# outd, 13, 13
        self.l3 = self.dconv_bn_relu(output_dim, 3) # 3, 16, 16

    def forward(self, x):
        x = self.l1(x)
        x = torch.reshape(x, (x.shape(0), -1, 4, 4))
        return self.l3(self.l2(x))


class Discriminator(nn.Module):
    def __init__(self, in_dim, out_dim=64):
        super.__init__()
        # 3, 64, 64 -> 1, 1, 1
        self.l1 = nn.Sequential(self.conv_bn_relu(in_dim, out_dim),
                                self.conv_bn_relu(out_dim, out_dim * 2),
                                self.conv_bn_relu(out_dim, out_dim * 4),
                                nn.Conv2d(out_dim * 4, 1, kernel_size=4, stride=2, padding=1))

    def conv_bn_relu(self, in_dim, out_dim):
        return nn.Sequential(nn.Conv2d(in_dim, out_dim, stride=2, padding=1, kernel_size=4),
                             nn.BatchNorm2d(out_dim),
                             nn.ReLU())

    def forward(self, x):
        x = self.l1(x)

        return x


config = {'lr': 0.001, 'epoch': 10, 'in_dim': 64}
class Trainer():
    def __init__(self):
        self.G = Generator(config['in_dim'])
        self.D = Discriminator(3)
        self.G_optimizer = torch.optim.Adam(lr=config['lr'])
        self.D_optimizer = torch.optim.RMSprop(lr=config['lr'])
        self.G_loss = nn.BCELoss()
        self.real_image_loader = DataLoader(image_data, 128, num_workers=2)
        self.G_input = torch.randn(100)

    def train(self):
        self.G.to(device)
        self.D.to(device)
        prograss_bar = tqdm(self.real_image_loader)
        for e in range(config['epoch']):
            prograss_bar.set_description(f'Epoch {e+1}')
            for i, data in enumerate(prograss_bar):
                data = data.to(device)

                '''             Train D              '''
                

