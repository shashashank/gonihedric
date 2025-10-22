import torch as T
import os, time
from datetime import datetime
# import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import barrier, reduce

class Trainer:
    def __init__(
        self,
        model: T.nn.Module,
        train_data: T.utils.data.DataLoader,
        test_data: T.utils.data.DataLoader,
        trainLen: int,
        testLen: int,
        optimizer: T.optim.Optimizer,
        save_every: int,
        path: str,
        classFlag: bool,
        criterion
    ) -> None:
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model = model.to(self.gpu_id) 
        self.train_data = train_data
        self.test_data = test_data
        self.trainLen = trainLen
        self.testLen = testLen
        self.optimizer = optimizer
        self.save_every = save_every
        self.criterion = criterion
        self.epochs_run = 0
        self.best_loss = {"loss":float("inf"), "epoch":0}
        self.best_acc = {"acc":0, "epoch":0}
        self.classFlag = classFlag
        self.path = path
        if os.path.exists(self.path+"snapshot.pt"):
            print("Loading Snapshot")
            self._load_snapshot(self.path+"snapshot.pt")
        self.model = DDP(self.model, device_ids=[self.gpu_id])

    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        PATH = self.path+ "_" + str(epoch)+ ".pt"
        T.save(ckp, PATH)
        print(f"Training checkpoint saved at {PATH}",end="")

    def _save_snapshot(self, epoch):
        snapshot = {}
        snapshot["MODEL_STATE"] = self.model.module.state_dict()
        snapshot["EPOCHS_RUN"] = epoch
        T.save(snapshot, self.path+"snapshot.pt")
        print(f"\nTraining snapshot saved at snapshot.pt")

    def _load_snapshot(self, snapshot_path):
        snapshot = T.load(snapshot_path)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print("Resuming training from snapshot at Epoch {self.epochs_run}")

    def _run_trainbatch(self, source, targets):
        self.optimizer.zero_grad()
        # output = self.model(source)['out']
        output = self.model(source); output=T.squeeze(output)
        loss = self.criterion(output, targets)
        loss.backward(); self.optimizer.step()
        return loss.item(), output.detach()

    @T.no_grad()
    def _run_testbatch(self, source, targets):
        # output = self.model(source)['out'].detach()
        output = self.model(source).detach(); output=T.squeeze(output)
        loss = self.criterion(output, targets).item()
        return loss, output

    def _run_epoch(self, epoch, testFlag=False):
        running_loss = 0; running_corrects = 0
        if testFlag:
            self.model.eval()
            data = self.test_data; self.test_data.sampler.set_epoch(0)
            func = self._run_testbatch; length = self.testLen; string = "\ttest"
        else:
            self.model.train()
            data = self.train_data; self.train_data.sampler.set_epoch(epoch)
            func = self._run_trainbatch; length = self.trainLen; string = "train"

        for source, targets in data:
            source = source.to(self.gpu_id); targets = targets.to(self.gpu_id)
            tmpLoss, output = func(source, targets)
            with T.no_grad():
                running_loss += tmpLoss * source.size(0)
                if self.classFlag:
                    running_corrects += (T.max(output, 1).indices == targets).sum().div(targets.numel()).item()

        running_loss = T.tensor([running_loss], device=self.gpu_id)
        reduce(running_loss, dst=0, op=T.distributed.ReduceOp.SUM)
        if self.classFlag:
            running_corrects = T.tensor([running_corrects], device=self.gpu_id)
            reduce(running_corrects, dst=0, op=T.distributed.ReduceOp.SUM)

        # will log the aggregated metrics only on the 0th GPU. Make sure "train_dataset" is of
        #   type Dataset and not DataLoader to get the size of the full dataset and not of the local shard
        if self.gpu_id==0:
            totalLoss = running_loss / length
            if self.classFlag:
                totalAcc = (running_corrects.double() / length) * 100 
                print(string+f" loss = {float(totalLoss):.6f}; accuracy = {float(totalAcc):.2f}%", end="")
                self.logfile.write(f"\nEpoch {epoch}: {string} loss = {float(totalLoss):.6f}; accuracy = {float(totalAcc):.2f}%")
                if testFlag:
                    if totalAcc > self.best_acc["acc"]:
                        self.best_acc["acc"] = totalAcc
                        self.best_acc["epoch"] = epoch
                    if totalLoss < self.best_loss["loss"]:
                        self.best_loss["loss"] = totalLoss
                        self.best_loss["epoch"] = epoch
            else:
                print(string+f" loss = {float(totalLoss):.6f}", end="")
                self.logfile.write(f"\nEpoch {epoch}: {string} loss = {float(totalLoss):.6f}")
                if testFlag and totalLoss < self.best_loss["loss"]:
                        self.best_loss["loss"] = totalLoss
                        self.best_loss["epoch"] = epoch
        barrier()

    def train(self, max_epochs: int):
        now = lambda : datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.logfile = open(self.path+"progress.txt", "a")
        self.logfile.write(f"\nTraining Started at {now()}")

        startTime = time.time()
        for epoch in range(self.epochs_run, max_epochs):
            if self.gpu_id == 0: print(f"\n\nEpoch {epoch}")
            self._run_epoch(epoch, testFlag=False)
            if epoch % self.save_every == 0 and epoch != 0:
                self._run_epoch(epoch, testFlag=True)
                if self.gpu_id == 0:
                    self._save_snapshot(epoch)
                    self._save_checkpoint(epoch)
                barrier()
        endTime = time.time()

        if self.gpu_id == 0:
            print(f"\n\nTraining Complete")
            self.logfile.write(f"\nTraining Complete at {now()}")
            print(f"Best Loss: {self.best_loss['loss']} at Epoch {self.best_loss['epoch']}")
            self.logfile.write(f"\nBest Loss: {self.best_loss['loss']} at Epoch {self.best_loss['epoch']}")
            if self.classFlag:
                print(f"Best Accuracy: {self.best_acc['acc']} at Epoch {self.best_acc['epoch']}")
                self.logfile.write(f"\nBest Accuracy: {self.best_acc['acc']} at Epoch {self.best_acc['epoch']}")
            print(f"Time taken for training: {endTime-startTime}\n")
            self.logfile.write(f"\nTime taken for training: {endTime-startTime}\n")
            self.logfile.close()
        barrier()

def prepare_dist_dataloader(dataset: T.utils.data.Dataset, batch_size: int):
    return T.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset),
    )
def prepare_dataloader(dataset: T.utils.data.Dataset, batch_size: int):
    return T.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False
    )