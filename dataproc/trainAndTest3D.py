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

base_dir = '/home/shashank/Code/gonihedric/'; data_dir = base_dir + "data/ghData/"


class ConvAutoencoder(nn.Module):
    def __init__(self, dropout=0.2):
        super(ConvAutoencoder, self).__init__()

        # self.drpt = nn.Dropout(dropout)
        # self.First = nn.Conv3d(1, 16, kernel_size=3, stride=2, padding=1),  # -> (16, H/2, W/2)
        # self.Second = nn.Conv3d(16, 24, kernel_size=3, stride=2, padding=1) # -> (8, H/4, W/4)
        # self.Third = nn.ConvTranspose3d(24, 40, kernel_size=3, stride=2, padding=1, output_padding=1), # -> (8, H/4, W/4)
        # self.Fourth = nn.ConvTranspose3d(40, 48, kernel_size=3, stride=2, padding=1, output_padding=1), # -> (16, H/2, W/2)
        # self.Fifth = nn.Conv3d(48, 1, kernel_size=3, stride=1, padding=1)
        # ----- Encoder -----
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, stride=2, padding=1),  # -> (16, H/2, W/2)
            nn.BatchNorm3d(16),
            nn.ReLU(True),

            nn.Dropout(dropout),
            nn.Conv3d(16, 24, kernel_size=3, stride=2, padding=1), # -> (8, H/4, W/4)
            nn.BatchNorm3d(24),
            nn.ReLU(True)
        )

        # ----- Decoder -----
        self.decoder = nn.Sequential(
            nn.Dropout(dropout),
            nn.ConvTranspose3d(24, 40, kernel_size=3, stride=2, padding=1, output_padding=1), # -> (8, H/4, W/4)
            nn.BatchNorm3d(40),
            nn.ReLU(True),

            nn.Dropout(dropout),
            nn.ConvTranspose3d(40, 48, kernel_size=3, stride=2, padding=1, output_padding=1), # -> (16, H/2, W/2)
            nn.BatchNorm3d(48),
            nn.ReLU(True),

            nn.Dropout(dropout),
            nn.Conv3d(48, 1, kernel_size=3, stride=1, padding=0),
            nn.Sigmoid()  # keeps output in [0,1] for binary images
        )

    def forward(self, x):
        # Encode
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# seed = 48348795161802367
seed=483482923932993232
utt.manualSeed(seed)

side = 10
batchSize = 256#
criterion = nn.MSELoss()# nn.BCELoss()#
model = ConvAutoencoder(dropout=0.2)
custNN.initialize_weights(model); model.to(device)

# transform = v2.Compose([utt.ReshapeTransform(([side*side])), v2.Lambda(lambda x: 2*x - 1)])# None#
# transform = v2.Compose([v2.Lambda(lambda x: 2*x - 1)])# None#


# transform = v2.Compose([v2.RandomVerticalFlip(),utt.ReshapeTransform(([side*side])), v2.Lambda(lambda x: 2*x - 1)])
# smallTrainset = ds.CustomAutoencoderDataset(data_dir+"tiny", side, transform) #2DGH32
dataset = ds.CustomAutoencoderDataset3D(data_dir+"L10k0", side) #2DGH32
trainset, testset = torch.utils.data.random_split(dataset, [90000, 10000])# [7000, 3000]
# testset = ds.CustomAutoencoderDataset3D(data_dir+"testG3D", side, transform)
trainLoader = torch.utils.data.DataLoader(trainset, batch_size=batchSize, shuffle=True, num_workers=4)
# smallTrainLoader = torch.utils.data.DataLoader(smallTrainset, batch_size=batchSize, shuffle=True, num_workers=4)
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
wd = 1e-4; lam = 0.0
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



optimizer = torch.optim.Adam(model.parameters(), lr=5e-4); epochs = 200
utt.trainAndTest(model, device, trainLoader, testLoader, criterion, optimizer, 40, 25,
                 lam=lam, folder=folder_name, number=1 ,num_epochs=epochs, saving=False)
torch.save(model.state_dict(), folder_name+"/modelFirst.pth")
# epochs = 400
# utt.trainAndTest(model, device, trainLoader, testLoader, criterion, optimizer, 40, 25,
#                  lam=lam, folder=folder_name, number=2 , num_epochs=epochs, saving=False)
# torch.save(model.state_dict(), folder_name+"/modelSecond.pth")
# optimizer = torch.optim.Adam(model.parameters(), lr=5e-5); epochs = 200
# utt.trainAndTest(model, device, trainLoader, testLoader, criterion, optimizer, side,
#                  lam=lam, folder=folder_name, number=3, num_epochs=epochs, saving=True)
# torch.save(model.state_dict(), folder_name+"/modelFinal.pth")