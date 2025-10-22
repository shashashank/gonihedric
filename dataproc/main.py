import percDDP as pNN
import torch as T
import os, time
from datetime import timedelta, datetime
# import datetime as dt
import torchDatasets as ds
import networks as custNN
from torch.distributed import init_process_group, destroy_process_group

# base_dir = "/home/shkal/experiments/"
# base_dir = "/home/shashank/Code/percolation/"


def ddp_setup():
    init_process_group(backend="nccl", timeout=timedelta(seconds=60))
    T.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def manualSeed(seed:int):
    T.manual_seed(seed)
    T.cuda.manual_seed_all(seed)
    T.backends.cudnn.deterministic = True
    T.backends.cudnn.benchmark = True


def main(lr:float, weight_decay:float, batchSize:int, momentum:float,
         total_epochs:int, save_every:int, side:int, iter:int):
    model = custNN.Autoencoder([900,300, 100, 30, 10, 2], T.nn.Tanh())
    # snapshot = T.load(data_dir+"checkpoints/Autoencoder/2025-04-23 14:41:34/chkpt1_30__1300.pt")
    # model.load_state_dict(snapshot["MODEL_STATE"])
    custNN.initialize_weights(model)
    optimizer = T.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    chkptDir = data_dir+"checkpoints/Autoencoder/"+datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    os.makedirs(os.path.dirname(chkptDir+"/"), exist_ok=True)
    #write parameters to file
    with open(chkptDir+"/params.txt", "w") as f:
        f.write("lr: "+str(lr)+"\n")
        f.write("weight_decay: "+str(weight_decay)+"\n")
        f.write("batch_size: "+str(batchSize)+"\n")
        f.write("total_epochs: "+str(total_epochs)+"\n")
        f.write("save_every: "+str(save_every)+"\n")
        f.write("lattice_size: "+str(side)+"\n")
        f.write("trials: "+str(iter)+"\n")
        f.write("seed: "+str(args.seed)+"\n")
        f.write("momentum: "+str(momentum)+"\n")
    trainset = ds.CustomAutoencoderDataset(data_dir+"lattice", side)
    testset = ds.CustomAutoencoderDataset(data_dir+"latticeTest", side)
    trainLoader = pNN.prepare_dist_dataloader(trainset, batchSize)
    testLoader = pNN.prepare_dist_dataloader(testset, batchSize)
    criterion = T.nn.MSELoss()
    trainer = pNN.Trainer(model, train_data=trainLoader, test_data=testLoader,
                          trainLen = len(trainset), testLen = len(testset),
                          optimizer=optimizer, save_every=save_every,
                          path=chkptDir+"/chkpt"+str(iter)+"_"+str(side)+"_",
                          classFlag=False, criterion=criterion)
    trainer.train(total_epochs+1)
    

if __name__ == "__main__":
    # manualSeed(8206768308764056476)
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--momentum', type=float, help='Momentum')
    parser.add_argument('--weight_decay', type=float, help='Weight decay')
    parser.add_argument('--lattice_size', type=int, help='Input lattice size')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('--save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--trials', type=int, help='Number of trials to run')
    parser.add_argument('--base_dir', type=str, help='Base directory')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--momentum', type=int, help='momentum')
    args = parser.parse_args()
    base_dir = args.base_dir
    data_dir = base_dir + "data/"
    if args.seed is not None:
        manualSeed(args.seed)
    ddp_setup()
    startTime = time.time()
    for i in range(0, args.trials):
        main(args.lr, args.weight_decay, args.batch_size, args.momentum,
             args.total_epochs, args.save_every, args.lattice_size, i+1)
    endTime = time.time()
    destroy_process_group()
    if args.trials > 1:
        print(f"Time taken per training regiment ({args.total_epochs} epochs): {(endTime - startTime)/args.trials}")
