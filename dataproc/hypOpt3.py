from functools import partial
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from ray import tune, train
import tempfile
import torchDatasets as ds
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler
# from torchvision.models import resnet50, resnet18
import networks as custNN

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

def load_data(data_dir, nTargets):
    # model, transform = custNN.modelPicker(modelName, side, nTargets, data_dir)
    model = custNN.Autoencoder([900, 750, 600, 450, 300, 150, 75, 30, 10, nTargets])
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    else:
        model = model.to(device)
    trainset = ds.CustomAutoencoderDataset(data_dir+"LatticeTest", side)
    testset = ds.CustomAutoencoderDataset(data_dir+"LatticeData1", side)
    return model, trainset, testset

def initialize_weights(model):
    nn.init.normal_(model.conv1.weight, 0, 0.1)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.1)
            nn.init.constant_(m.bias, 0)


def train_pClas(config):

    net, trainset, _ = load_data(dataDir, nTargets)
    optimizer = optim.SGD(net.parameters(), lr=config["batch_size"],
                          momentum=0.875,
                          weight_decay = 1.0/32768)
    # optimizer = optim.Adam(net.parameters(), 
    #                        lr=config["lr"], 
    #                        weight_decay = config["wd"], 
    #                        amsgrad=config["amsgrad"])
    # optimizer = optim.Adadelta(net.parameters(), lr=config["lr"], weight_decay=config["wd"])

    exp_lr_scheduler = None# optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=config["gamma"])

    # Load existing checkpoint through `get_checkpoint()` API.
    if train.get_checkpoint():
        loaded_checkpoint = train.get_checkpoint()
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
            model_state, optimizer_state = torch.load(
                os.path.join(loaded_checkpoint_dir, "checkpoint.pt")
            )
            net.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)

    test_abs = int(len(trainset) * 0.8)
    train_subset, val_subset = random_split(
        trainset, [test_abs, len(trainset) - test_abs])

    trainloader = DataLoader( train_subset,
                             batch_size=int(config["batch_size"]),
                             shuffle=True,
                             num_workers=4)
    valloader = DataLoader( val_subset,
                           batch_size=int(config["batch_size"]),
                           shuffle=True,
                           num_workers=4)

    for epoch in range(50):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        net.train(); i = 0
        for inputs, labels in trainloader:
            # get the inputs; data is a list of [inputs, labels]
            inputs = inputs.to(device); labels = labels.to(device)

            optimizer.zero_grad() # zero the parameter gradients
            # forward + backward + optimize
            outputs = net(inputs).squeeze()
            # outputs = outputs.float(); labels = labels.float()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(
                    "[%d, %5d] loss: %.3f"
                    % (epoch + 1, i + 1, running_loss / epoch_steps)
                )
                epoch_steps = 0
                running_loss = 0.0
                if exp_lr_scheduler is not None:
                    exp_lr_scheduler.step()
            i += 1

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        # correct = 0
        net.eval()
        with torch.no_grad():
            for inputs, labels in valloader:
                inputs = inputs.to(device); labels = labels.to(device)
                outputs = net(inputs)
                outputs = outputs.detach().squeeze().float(); labels = labels.float()
                val_loss += criterion(outputs, labels).cpu().numpy()
                # predicted = torch.max(F.softmax(outputs, dim=1), 1).indices
                # correct += (predicted == labels.max(1).indices).sum().div(torch.numel(predicted)).item()
                val_steps += 1

# Here we save a checkpoint. It is automatically registered with
        # Ray Tune and will potentially be accessed through in ``get_checkpoint()``
        # in future iterations.
        # Note to save a file like checkpoint, you still need to put it under a directory
        # to construct a checkpoint.
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
            torch.save(
                (net.state_dict(), optimizer.state_dict()), path
            )
            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
            train.report(
                {"loss": (val_loss / val_steps), 
                #  "accuracy": (correct / val_steps)
                 },
                checkpoint=checkpoint,
            )
    print("Finished Training")

def test_best_model(best_result):
    best_trained_model, _, testset = load_data(dataDir, nTargets=nTargets)

    checkpoint_path = os.path.join(best_result.checkpoint.to_directory(), "checkpoint.pt")

    model_state, _ = torch.load(checkpoint_path)
    best_trained_model.load_state_dict(model_state)

    testloader = DataLoader( testset, batch_size=best_result.config["batch_size"], 
                            shuffle=False, num_workers=4)

    correct = 0; total = 0; loss = 0
    best_trained_model.eval()
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = best_trained_model(inputs)
            outputs = outputs.detach().squeeze().float(); labels = labels.float()
            loss += criterion(outputs, labels).cpu().numpy()
            # predicted = torch.max(F.softmax(outputs, dim=1), 1).indices
            # correct += (predicted == labels.max(1).indices).sum().div(torch.numel(predicted)).item()
            total += 1
    # print("Best trial test set accuracy for \"{}\": {}".format(trainScheme, correct/total))
    print("Best trial test set loss for \"{}\": {}".format(trainScheme, loss/total))


def main(num_samples=10, max_num_epochs=10, cpus_per_trial=6, gpus_per_trial=2):
    config = {
    # "lr": tune.loguniform(1e-6, 1e-2),
    "batch_size": tune.choice([64, 128, 256]),
    # "wd": tune.choice([0.0, 0.5e-3, 1e-2, 1e-1]),
    # "gamma": tune.uniform(0.1, 0.9),
    # "momentum": tune.uniform(0.1, 1.0),
    # "amsgrad": tune.choice([True, False])
    }
    print(modelName + " with SGD for OP side:"+str(side))
    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2
    )
    tuner = tune.Tuner(
         tune.with_resources(
            tune.with_parameters(train_pClas),
            resources={"cpu": cpus_per_trial, "gpu": gpus_per_trial}
        ),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        param_space=config,
    )
    results = tuner.fit()
    best_result = results.get_best_result("loss", "min")

    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation loss: {}".format(
        best_result.metrics["loss"]))
    if nTargets > 1:
        print("Best trial final validation accuracy: {}".format(
            best_result.metrics["accuracy"]))
    test_best_model(best_result)

if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--base_dir', type=str, help='Base directory')
    parser.add_argument('--modelName', type=str, help='Model name')
    parser.add_argument('--trainScheme', type=str, help='Training scheme')
    parser.add_argument('--nTargets', type=int, help='Number of targets')
    parser.add_argument('--side', type=int, help='Input lattice size')
    args = parser.parse_args()
    modelName = args.modelName; side = args.side; nTargets = args.nTargets
    base_dir = args.base_dir; dataDir = base_dir + "data/"
    # criterion = nn.MSELoss() if nTargets == 1 else nn.CrossEntropyLoss()
    criterion = nn.L1Loss()
    main(num_samples=6, max_num_epochs=30, gpus_per_trial=1)