import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torchDatasets as ds
from torchvision.transforms import v2

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

def visualize_reconstruction(model, device, data_loader, side1, side2, location="", name:str="reconstruction"):
    model.eval()
    with torch.no_grad():
        images = next(iter(data_loader))
        images = images.to(device)
        reconstructed = model(images)

        # Plot original vs reconstructed images
        fig, axes = plt.subplots(2, 8, figsize=(15, 4))
        for i in range(8):
            # Original images
            axes[0,i].imshow(images[i].cpu().numpy().squeeze().reshape(side1, side2), cmap='gray')
            axes[0,i].axis('off')

            # Reconstructed images
            axes[1,i].imshow(reconstructed[i].cpu().numpy().squeeze().reshape(side1, side2), cmap='gray')
            axes[1,i].axis('off')

        plt.tight_layout()
        if location != "":
            plt.savefig(location+"/"+name+".png")
        else:
            plt.show()
        plt.close()

def visualize_dataset(dataDir:str, side:int):
    transform = ReshapeTransform(([1, side,side]))
    dataset = ds.CustomAutoencoderDataset(dataDir, side, transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)
    images, _ = next(iter(dataloader))
    fig, axes = plt.subplots(1, 10, figsize=(15, 4))
    for i in range(10):
        axes[i].imshow(images[i].numpy().squeeze().reshape(side, side), cmap='gray')
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()
    plt.close()

def visualize_dataset3D(dataDir:str, side1:int, side2:int):
    transform = ReshapeTransform(([1, side1,side2]))
    dataset = ds.CustomAutoencoderDataset3D(dataDir, int(np.cbrt(side1*side2)), transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)
    images, _ = next(iter(dataloader))
    fig, axes = plt.subplots(1, 10, figsize=(15, 4))
    for i in range(10):
        axes[i].imshow(images[i].numpy().squeeze().reshape(side1, side2), cmap='gray')
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()
    plt.close()

def plotLosses(trainArr, testArr, epochs, minX, maxX, folder:str=None):
    # Plotting the training and test losses
    plt.figure(figsize=(10, 5))
    plt.plot(trainArr, label='Train Loss', color='blue')
    plt.plot(testArr, label='Test Loss', color='orange')
    plt.hlines(y=testArr[epochs//10:].mean(), xmin=0, xmax=epochs-1, color='red', linestyle='--', label='Threshold')
    plt.title(f"Min test:{np.min(testArr[minX:maxX])} at epoch:{np.argmin(testArr[minX:maxX])+minX}"+
              f" and min train:{np.min(trainArr[minX:maxX])} at epoch:{np.argmin(trainArr[minX:maxX])+minX}")
    plt.plot(np.argmin(testArr[minX:maxX])+minX, testArr[np.argmin(testArr[minX:maxX])+minX], 'g.',label='Min Test Loss')
    plt.plot(np.argmin(trainArr[minX:maxX])+minX, trainArr[np.argmin(trainArr[minX:maxX])+minX], 'm.',label='Min Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xlim(minX, maxX)
    # plt.ylim(np.min(trainArr), np.max(testArr[20:])+0.001)
    plt.legend()
    # plt.grid()
    plt.savefig(folder + "loss_plot.png")
    # plt.show()
    plt.close()


def trainAndTest(model, device, trainLoader, testLoader, criterion, optimizer, side1, side2, lam:float, folder:str, number:int, num_epochs=10, saving=False):
    # Create the folder
    side = int(np.cbrt(side1*side2))
    folder_name = folder + '/' + str(number) +'/'
    os.makedirs(folder_name, exist_ok=True)

    if lam > 0:
        contractive_loss = lambda enc, imag : torch.norm(torch.autograd.functional.jacobian(enc, imag, create_graph=True))
    else:
        contractive_loss = lambda enc, imag : 0
    
    trainLosses = np.empty(num_epochs); minTrainLoss = np.inf
    testLosses = np.empty(num_epochs); minTestLoss = np.inf
    for epoch in range(num_epochs):
        model.train()
        # Iterate over data.
        total_loss = 0
        for batch_idx, images in enumerate(trainLoader):
            # Move images to device
            images = images.to(device)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, images) + lam * contractive_loss(model.encoder, images)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        trainLosses[epoch] = total_loss / len(trainLoader)
        model.eval()
        with torch.no_grad():
            total_loss = 0
            for batch_idx, images in enumerate(testLoader):
                images = images.to(device)
                outputs = model(images)
                loss = criterion(outputs, images).cpu().numpy()
                total_loss += loss
            testLosses[epoch] = total_loss / len(testLoader)
            if testLosses[epoch] < minTestLoss:
                minTestLoss = testLosses[epoch]
                torch.save(model.state_dict(), folder_name+f"/best_model.pth")
                visualize_reconstruction(model, device, testLoader, side1, side2, folder_name, name="best_test")
            if trainLosses[epoch] < minTrainLoss:
                minTrainLoss = trainLosses[epoch]
                torch.save(model.state_dict(), folder_name+f"/best_train_model.pth")
                visualize_reconstruction(model, device, trainLoader, side1, side2, folder_name, name="best_train")
        if epoch % 10 == 0 and saving:
            torch.save(model.state_dict(), folder_name+f"/model_epoch_{epoch}.pth")
    np.save(folder_name + "trainLosses.npy", trainLosses)
    np.save(folder_name + "testLosses.npy", testLosses)
    plotLosses(trainLosses, testLosses, num_epochs, 0, num_epochs-1, folder=folder_name)
