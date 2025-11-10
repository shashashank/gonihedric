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
from torchvision import datasets
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
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1, padding_mode='zeros'),  # -> (32, H/2, W/2)
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.Dropout(dropout),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # -> (64, H/4, W/4)
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Dropout(dropout),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),# -> (128, H/7, W/7)
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )

        # Bottleneck (latent space)
        self.fc_enc = nn.Linear(128 * 4 * 4, latent_dim)   # assumes input = 32x32
        self.fc_dec = nn.Linear(latent_dim, 128 * 4 * 4)

        # ----- Decoder -----
        self.decoder = nn.Sequential(
            nn.Dropout(dropout),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=0), # -> (64, H/4, W/4)
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Dropout(dropout),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, output_padding=0), # -> (32, H/2, W/2)
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.Dropout(dropout),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1, output_padding=0),  # -> (1, H, W)
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

seed = 483482923932993232
utt.manualSeed(seed)

side = 28
batchSize =128# 256#  64#
criterion = nn.BCELoss()# nn.MSELoss()#
# model = custNN.Autoencoder([900, 750, 600, 450, 300, 150, 75, 30, 10, 2], nn.Tanh(), nn.Tanh())
model = custNN.Autoencoder([784, 200, 50, 10], nn.ReLU(), nn.Sigmoid(), F.sigmoid)
# model = ConvAutoencoder(10, dropout=0.2)
custNN.initialize_weights(model); model.to(device)

# Define transforms
transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    utt.ReshapeTransform(([side*side])),
    # v2.Lambda(lambda x: 2*x - 1),
    # v2.Lambda(lambda x: torch.flatten(x, start_dim=1)),  # Flatten the image
])

# Load MNIST dataset
train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    transform=transform,
    download=True
)
test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    transform=transform
)

# _, smallTrainset = torch.utils.data.random_split(train_dataset, [43000, 17000])
# _, trainset, smallTrainset = torch.utils.data.random_split(train_dataset, [58600, 1000, 400])
trainLoader = torch.utils.data.DataLoader(train_dataset, batch_size=batchSize, shuffle=True, num_workers=4)
# smallTrainLoader = torch.utils.data.DataLoader(smallTrainset, batch_size=batchSize, shuffle=True, num_workers=4)
# tinyLoader = torch.utils.data.DataLoader(tinyset, batch_size=batchSize, shuffle=True, num_workers=4)
testLoader = torch.utils.data.DataLoader(test_dataset, batch_size=batchSize, shuffle=False)

# Get current date and time
now = datetime.now()

folder_name = data_dir+"checkpoints/Autoencoder/"+now.strftime("%Y-%m-%d_%H-%M-%S")
os.makedirs(folder_name+'/', exist_ok=True)

source_path = Path(__file__).resolve()

# Copy File
shutil.copy(source_path, folder_name+'/'+source_path.name)
wd=1e-4; lam=1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=wd); epochs = 50
utt.trainAndTest(model, device, trainLoader, testLoader, criterion, optimizer, side,
                 lam=lam, folder=folder_name, number=1 ,num_epochs=epochs, saving=False)
torch.save(model.state_dict(), folder_name+"/modelFirst.pth")
# optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=wd); epochs = 40
# utt.trainAndTest(model, device, trainLoader, testLoader, criterion, optimizer, side,
#                  lam=lam, folder=folder_name, number=2 , num_epochs=epochs, saving=False)
# torch.save(model.state_dict(), folder_name+"/modelSecond.pth")
# optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=wd); epochs = 20
# utt.trainAndTest(model, device, trainLoader, testLoader, criterion, optimizer, side,
#                  lam=lam, folder=folder_name, number=3, num_epochs=epochs, saving=False)
# torch.save(model.state_dict(), folder_name+"/modelFinal.pth")
