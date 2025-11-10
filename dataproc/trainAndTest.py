import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchDatasets as ds
import networks as custNN
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import shutil
import utilsTrainTest as utt
from datetime import datetime

sns.set_theme(style='whitegrid')

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern Roman']
plt.rcParams['figure.dpi']= 300

base_dir = '/home/shashank/Code/gonihedric/'; data_dir = base_dir + "data/"


class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=8, dropout=0.2):
        super(ConvAutoencoder, self).__init__()

        self.drpt = nn.Dropout(dropout)

        # ----- Encoder -----
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, padding_mode='circular'),  # -> (32, H/2, W/2)
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.Dropout(dropout),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # -> (64, H/4, W/4)
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Dropout(dropout),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),# -> (128, H/8, W/8)
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )

        # Bottleneck (latent space)
        self.fc_enc = nn.Linear(128 * 4 * 4, latent_dim)   # assumes input = 32x32
        self.fc_dec = nn.Linear(latent_dim, 128 * 4 * 4)

        # ----- Decoder -----
        self.decoder = nn.Sequential(
            nn.Dropout(dropout),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1), # -> (64, H/4, W/4)
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Dropout(dropout),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), # -> (32, H/2, W/2)
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.Dropout(dropout),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # -> (1, H, W)
            nn.Sigmoid()  # keeps output in [0,1] for binary images
        )

    def forward(self, x):
        # Encode
        x = self.encoder(x)
        x = x.view(x.size(0), -1)      # flatten for FC
        x = self.drpt(x)
        z = self.fc_enc(x)

        # Decode
        z = self.drpt(z)
        x = self.fc_dec(z)
        x = x.view(x.size(0), 128, 4, 4)  # reshape back
        x = self.decoder(x)
        return x

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# seed = 48348795161802367
seed=483482923932993232
utt.manualSeed(seed)

side = 30
batchSize = 64# 128# 256#
criterion = nn.MSELoss()# nn.BCELoss()#
# model = custNN.Autoencoder([900, 750, 600, 450, 300, 150, 75, 30, 10, 2], nn.Sigmoid(), nn.Sigmoid(), F.sigmoid)
model = custNN.Autoencoder([900, 750, 600, 450, 300, 150, 75, 30, 10, 2], nn.Tanh(), nn.Tanh(), F.tanh)
# model = custNN.Autoencoder([900, 300, 30, 2], nn.Tanh(), nn.Tanh(), F.tanh)
# model = ConvAutoencoder(8, dropout=0.2)
custNN.initialize_weights(model); model.to(device)

transform = v2.Compose([utt.ReshapeTransform(([side*side])), v2.Lambda(lambda x: 2*x - 1)])# None#

# transform = v2.Compose([v2.RandomVerticalFlip(),utt.ReshapeTransform(([side*side])), v2.Lambda(lambda x: 2*x - 1)])
# smallTrainset = ds.CustomAutoencoderDataset(data_dir+"tiny", side, transform) #2DGH32
trainset = ds.CustomAutoencoderDataset(data_dir+"small", side, transform) #2DGH32
_, smallTrainset = torch.utils.data.random_split(trainset, [1000, 400])# [7000, 3000]
testset = ds.CustomAutoencoderDataset(data_dir+"test", side, transform)
trainLoader = torch.utils.data.DataLoader(trainset, batch_size=batchSize, shuffle=True, num_workers=4)
smallTrainLoader = torch.utils.data.DataLoader(smallTrainset, batch_size=batchSize, shuffle=True, num_workers=4)
testLoader = torch.utils.data.DataLoader(testset, batch_size=batchSize, shuffle=False)

# Loss function and optimizer
# optimizer = torch.optim.SGD(model.parameters(), lr=5e-4, momentum=0.93079, weight_decay=0.01)

# Get current date and time
now = datetime.now()

folder_name = data_dir+"checkpoints/Autoencoder/"+now.strftime("%Y-%m-%d_%H-%M-%S")
os.makedirs(folder_name+'/', exist_ok=True)

source_path = Path(__file__).resolve()

# Copy File
shutil.copy(source_path, folder_name+'/'+source_path.name)
# wd=0.0011; lam=0.0046
wd = 0.0; lam = 0.0
# optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=25e-5); epochs = 50
# utt.trainAndTest(model, device, trainLoader, testLoader, criterion, optimizer, side,
#                  lam=lam, folder=folder_name, number=1 ,num_epochs=epochs, saving=False)
# torch.save(model.state_dict(), folder_name+"/modelFirst.pth")
# optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=25e-5); epochs = 100
# utt.trainAndTest(model, device, trainLoader, testLoader, criterion, optimizer, side,
#                  lam=lam, folder=folder_name, number=2 , num_epochs=epochs, saving=False)
# torch.save(model.state_dict(), folder_name+"/modelSecond.pth")
# optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-3); epochs = 100
# utt.trainAndTest(model, device, trainLoader, testLoader, criterion, optimizer, side,
#                  lam=lam, folder=folder_name, number=3, num_epochs=epochs, saving=False)
# torch.save(model.state_dict(), folder_name+"/modelFinal.pth")



optimizer = torch.optim.Adam(model.parameters(), lr=5e-4); epochs = 1500
utt.trainAndTest(model, device, smallTrainLoader, testLoader, criterion, optimizer, side,
                 lam=lam, folder=folder_name, number=1 ,num_epochs=epochs, saving=True)
torch.save(model.state_dict(), folder_name+"/modelFirst.pth")
epochs = 4000
utt.trainAndTest(model, device, trainLoader, testLoader, criterion, optimizer, side,
                 lam=lam, folder=folder_name, number=2 , num_epochs=epochs, saving=True)
torch.save(model.state_dict(), folder_name+"/modelSecond.pth")
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5); epochs = 2000
utt.trainAndTest(model, device, trainLoader, testLoader, criterion, optimizer, side,
                 lam=lam, folder=folder_name, number=3, num_epochs=epochs, saving=True)
torch.save(model.state_dict(), folder_name+"/modelFinal.pth")