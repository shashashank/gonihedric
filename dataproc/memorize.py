import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchDatasets as ds
import networks as custNN
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

sns.set_theme(style='whitegrid')

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern Roman']
plt.rcParams['figure.dpi']= 300

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

def manualSeed(seed:int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ReshapeTransform:
    def __init__(self, shape):
        self.shape = shape
    def __call__(self, x):
        return x.view(*self.shape)

base_dir = '/home/shashank/Code/gonihedric/'; data_dir = base_dir + "data/"

def visualize_reconstruction(model, data_loader):
    model.eval()
    with torch.no_grad():
        images, _ = next(iter(data_loader))
        images = images.to(device)
        reconstructed = model(images)

        # Plot original vs reconstructed images
        fig, axes = plt.subplots(2, 8, figsize=(15, 4))
        for i in range(8):
            # Original images
            axes[0,i].imshow(images[i].cpu().numpy().squeeze().reshape(30, 30), cmap='gray')
            axes[0,i].axis('off')

            # Reconstructed images
            axes[1,i].imshow(reconstructed[i].cpu().numpy().squeeze().reshape(30, 30), cmap='gray')
            axes[1,i].axis('off')

        plt.tight_layout()
        plt.show()

def plotLosses(trainArr, testArr, epochs, minX, maxX, folder:str=None):
    # Plotting the training and test losses
    plt.figure(figsize=(10, 5))
    plt.plot(trainArr, label='Train Loss', color='blue')
    plt.plot(testArr, label='Test Loss', color='orange')
    plt.hlines(y=testArr[20:].mean(), xmin=0, xmax=epochs-1, color='red', linestyle='--', label='Threshold')
    plt.title(f"Minimum test loss: {np.min(testArr[minX:maxX])} at epoch: {np.argmin(testArr[minX:maxX])+minX}")
    plt.plot(np.argmin(testArr[minX:maxX])+minX, testArr[np.argmin(testArr[minX:maxX])+minX], 'g.',label='Min Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xlim(minX, maxX)
    # plt.ylim(np.min(trainArr), np.max(testArr[20:])+0.001)
    plt.legend()
    # plt.grid()
    plt.savefig(folder + "loss_plot.png")
    # plt.show()


def trainAndTest(model, trainLoader, testLoader, criterion, optimizer, folder:str, number:int, num_epochs=10, saving=False):
    # Create the folder
    folder_name = folder + '/' + str(number) +'/'
    os.makedirs(folder_name, exist_ok=True)

    trainLosses = np.empty(num_epochs)
    testLosses = np.empty(num_epochs)
    for epoch in range(num_epochs):
        model.train()
        # Iterate over data.
        total_loss = 0
        for batch_idx, (images, _) in enumerate(trainLoader):
            # Move images to device
            images = images.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, images)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        trainLosses[epoch] = total_loss / len(trainLoader)
        model.eval()
        with torch.no_grad():
            total_loss = 0
            for batch_idx, (images, _) in enumerate(testLoader):
                images = images.to(device)
                outputs = model(images)
                loss = criterion(outputs, images).cpu().numpy()
                total_loss += loss
            testLosses[epoch] = total_loss / len(testLoader)
        if epoch % 10 == 0 and saving:
            torch.save(model.state_dict(), folder_name+f"/model_epoch_{epoch}.pth")
    np.save(folder_name + "trainLosses.npy", trainLosses)
    np.save(folder_name + "testLosses.npy", testLosses)
    plotLosses(trainLosses, testLosses, num_epochs, 0, num_epochs-1, folder=folder_name)


seed = 483482923932993232
manualSeed(seed)

side = 30
batchSize =128# 256#  64# 
criterion = nn.MSELoss()# nn.BCELoss()#
model = custNN.Autoencoder([900, 750, 600, 450, 300, 150, 75, 30, 10, 2], nn.Tanh(), nn.Tanh())
# model = custNN.Autoencoder([1024, 750, 600, 450, 300, 150, 75, 30, 10, 2])
# model = ConvAutoencoder(2, dropout=0.2)
custNN.initialize_weights(model); model.to(device)

# transform = transform = v2.Compose([v2.RandomHorizontalFlip(),
#                                     v2.RandomVerticalFlip(),
#                                     ReshapeTransform(([side*side]))])# None 

transform = v2.Compose([ReshapeTransform(([side*side])), v2.Lambda(lambda x: 2*x - 1)])
trainset = ds.CustomAutoencoderDataset(data_dir+"small", side, transform)
_, smallTrainset = torch.utils.data.random_split(trainset, [1000, 400])
# tinyset = ds.CustomAutoencoderDataset(data_dir+"tiny", side, transform)
testset = ds.CustomAutoencoderDataset(data_dir+"test", side, transform)
trainLoader = torch.utils.data.DataLoader(trainset, batch_size=batchSize, shuffle=True, num_workers=4)
smallTrainLoader = torch.utils.data.DataLoader(smallTrainset, batch_size=batchSize, shuffle=True, num_workers=4)
# tinyLoader = torch.utils.data.DataLoader(tinyset, batch_size=batchSize, shuffle=True, num_workers=4)
testLoader = torch.utils.data.DataLoader(testset, batch_size=batchSize, shuffle=False)

# Loss function and optimizer
# optimizer = torch.optim.SGD(model.parameters(), lr=5e-4, momentum=0.93079, weight_decay=0.01)

# Get current date and time
now = datetime.now()

folder_name = data_dir+"checkpoints/Autoencoder/"+now.strftime("%Y-%m-%d_%H-%M-%S")

optimizer = torch.optim.Adam(model.parameters(), lr=5e-4); epochs = 1500
trainAndTest(model, smallTrainLoader, testLoader, criterion, optimizer, folder=folder_name, number=1 ,num_epochs=epochs, saving=True)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4); epochs = 4000
trainAndTest(model, trainLoader, testLoader, criterion, optimizer, folder=folder_name, number=2 , num_epochs=epochs, saving=True)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5); epochs = 2000
trainAndTest(model, trainLoader, testLoader, criterion, optimizer, folder=folder_name, number=3, num_epochs=epochs, saving=True)
torch.save(model.state_dict(), folder_name+"modelFinal.pth")