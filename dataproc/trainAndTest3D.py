import torch
# import numpy as np
import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
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
import argparse

sns.set_theme(style='whitegrid')

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern Roman']
plt.rcParams['figure.dpi']= 300

base_dir = '/home/shashank/Code/gonihedric/';
parser = argparse.ArgumentParser(description='Train a 3D autoencoder on lattice data')
# parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
# parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for optimizer')
# parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
# parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for optimizer')
# parser.add_argument('--total_epochs', type=int, default=100, help='Total number of training epochs')
# parser.add_argument('--save_every', type=int, default=10, help='Save model checkpoint every N epochs')
parser.add_argument('--side', type=int, default=10, help='Side length of the lattice')
# parser.add_argument('--iter', type=int, default=1, help='Iteration number for checkpoint naming')
# parser.add_argument('--seed', type=int, default=48348795161802367, help='Random seed for reproducibility')
parser.add_argument('--folder', type=str, default='data/ghData', help='Folder name for saving checkpoints')
args = parser.parse_args()
 
# data_dir = base_dir + "data/ghData/"
# data_dir = base_dir + "data/"
data_dir = base_dir + args.folder + "/"

class ConvAutoencoder(nn.Module):
    def __init__(self, channelMult=8, dropout=0.2):
        super(ConvAutoencoder, self).__init__()

        # ----- Encoder -----
        self.encoder = nn.Sequential(
            nn.Conv3d(1, channelMult, kernel_size=3, stride=2, padding=1),  # -> (16, H/2, W/2)
            nn.BatchNorm3d(channelMult),
            nn.ReLU(True),

            nn.Dropout(dropout),
            nn.Conv3d(channelMult, 2*channelMult, kernel_size=3, stride=2, padding=1), # -> (8, H/4, W/4)
            nn.BatchNorm3d(2*channelMult),
            nn.ReLU(True)
        )

        # ----- Decoder -----
        self.decoder = nn.Sequential(
            nn.Dropout(dropout),    # output padding changed 0 -> 1
            nn.ConvTranspose3d(2*channelMult, channelMult, kernel_size=3, stride=2, padding=1, output_padding=1), # -> (8, H/4, W/4)
            nn.BatchNorm3d(channelMult),
            nn.ReLU(True),

            nn.Dropout(dropout),
            nn.ConvTranspose3d(channelMult, 1, kernel_size=3, stride=2, padding=1, output_padding=1), # -> (16, H/2, W/2)
            # nn.BatchNorm3d(1),
            # nn.ReLU(True),

            # nn.Dropout(dropout),
            # nn.Conv3d(48, 1, kernel_size=3, stride=1, padding=0),
            nn.Sigmoid()  # keeps output in [0,1] for binary images
        )

    def forward(self, x):
        # Encode
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

seed = 48348795161802367
# seed=483482923932993233
utt.manualSeed(seed)

side = args.side
batchSize = 128#
criterion = nn.MSELoss()# nn.BCELoss()#
model = ConvAutoencoder(channelMult=16, dropout=0.14404089356326266)
custNN.initialize_weights(model); model.to(device)

# transform = v2.Compose([utt.ReshapeTransform(([side*side])), v2.Lambda(lambda x: 2*x - 1)])# None#
# transform = v2.Compose([v2.Lambda(lambda x: 2*x - 1)])# None#


# transform = v2.Compose([v2.RandomVerticalFlip(),utt.ReshapeTransform(([side*side])), v2.Lambda(lambda x: 2*x - 1)])
# smallTrainset = ds.CustomAutoencoderDataset(data_dir+"tiny", side, transform) #2DGH32
dataset = ds.CustomAutoencoderDataset3D(data_dir+"trainL"+str(args.side), side) #2DGH32 # testL10k.9
trainset, testset = torch.utils.data.random_split(dataset, [0.8, 0.2])
# testset = ds.CustomAutoencoderDataset3D(data_dir+"testIsing", side) # testG3D
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

# 0.0017158985105088596, 'batch_size': 64, 'wd': 1.5490108431459896e-05
#0.002065691663432126, 'batch_size': 128, 'wd': 0.00016894053078186604, 'dropout': 0.14404089356326266

optimizer = torch.optim.Adam(model.parameters(), lr=0.002065691663432126, weight_decay=0.00016894053078186604); epochs = 50
utt.trainAndTestLabel(model, device, trainLoader, testLoader, criterion, optimizer, side,
                 lam=lam, folder=folder_name, number=1 ,num_epochs=epochs, saving=False)
torch.save(model.state_dict(), folder_name+"/modelFirst.pth")
# epochs = 100
# utt.trainAndTestLabel(model, device, trainLoader, testLoader, criterion, optimizer, side,
#                  lam=lam, folder=folder_name, number=2 , num_epochs=epochs, saving=False)
# torch.save(model.state_dict(), folder_name+"/modelSecond.pth")