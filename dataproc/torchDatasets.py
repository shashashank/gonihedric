
import torch as T
import os
import numpy as np

def lineNumber(fileName):
    def csv_reader(file_name):
        for row in open(file_name, "r"):
            yield row
    csv_gen = csv_reader(fileName)
    row_count = 0
    for row in csv_gen:
        row_count += 1
    return row_count

def transforms(trainScheme:str, side:int, nTargets:int):
    if nTargets==1:
        if trainScheme=="percolation strength":
            target_transform = lambda x: x.to(T.float)/(side*side)
        elif trainScheme=="occupation probability":
            target_transform = lambda x: x.to(T.float)
        elif trainScheme=="average cluster size":
            target_transform = lambda x: x.to(T.float)/(side*side)
    else:
        target_transform=lambda x: T.Tensor.float(T.nn.functional.one_hot((x*10 - 1).long(), nTargets))
    return target_transform

# Custom Dataset: needs a data file and a config file in the format
#   dataFileData and dataFileConfig
class CustomDataset(T.utils.data.Dataset):
    def __init__(self, dataFile:str, side:int, yVal:str, transform=None, target_transform=None):
        self.file = dataFile + str(side);
        self.length = lineNumber(self.file+"op")
        self.side2 = side*side

        # seekSkip is the number of characters to skip to reach the next line
        intWidth = np.ceil(np.log10(self.side2))
        match yVal:
            case "occupation probability":
                self.labels = lambda x:T.tensor(np.array(x.readline().strip(), dtype=float), dtype=T.float)
                self.scheme = "op"; self.dataSeekSkip = 5+1
            case "cluster segmentation":
                self.labels = lambda x:T.tensor(np.fromstring(x.readline().strip(), dtype=int, sep=" "), dtype=T.int)
                self.scheme = "clFile"; self.dataSeekSkip = self.side2*(intWidth+1)+1
            case "number of clusters":
                self.labels = lambda x:T.tensor(np.array(x.readline().strip(), dtype=int), dtype=T.int)
                self.scheme = "numCl"; self.dataSeekSkip = intWidth+1
            case "largest cluster":
                self.labels = lambda x:T.tensor(np.array(list(x.readline().strip()), dtype=int), dtype=T.int)
                self.scheme = "ClSeg"; self.dataSeekSkip = self.side2+1
            case "percolation strength":
                self.labels = lambda x:T.tensor(np.array(x.readline().strip(), dtype=float), dtype=T.float)
                self.scheme = "maxClSz"; self.dataSeekSkip = intWidth+1
            case "average cluster size":
                self.labels = lambda x:T.tensor(np.array(x.readline().strip(), dtype=float), dtype=T.float)
                self.scheme = "avgClSz"; self.dataSeekSkip = intWidth+4
        self.configSeekSkip = self.side2+1 # config+1x\n
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with open(self.file + "lattice", 'r') as configs, open(self.file + self.scheme, 'r') as labels:
            if idx >= self.length:
                raise IndexError
            configs.seek(idx*self.configSeekSkip, os.SEEK_SET)
            labels.seek(idx*self.dataSeekSkip, os.SEEK_SET)
            label = self.labels(labels)
            config = T.tensor(np.array(list(configs.readline().strip()), dtype=int)).type(T.float)
            if self.transform:
                config = self.transform(config)
            if self.target_transform:
                label = self.target_transform(label)
            return (config, label)
        
class CustomAutoencoderDataset(T.utils.data.Dataset):
    def __init__(self, dataFile:str, side:int, transform=None, target_transform=None):
        self.file = dataFile;
        self.length = lineNumber(self.file+"data")
        self.side = side
        self.side2 = side*side

        # seekSkip is the number of characters to skip to reach the next line
        intWidth = np.ceil(np.log10(self.side2))
        self.dataSeekSkip = 17+1

        self.configSeekSkip = self.side2+1 # config+1x\n
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with open(self.file + "lattice", 'r') as configs, open(self.file + "data", 'r') as labels:
            if idx >= self.length:
                raise IndexError
            configs.seek(idx*self.configSeekSkip, os.SEEK_SET)
            labels.seek(idx*self.dataSeekSkip, os.SEEK_SET)
            label = T.tensor(np.fromstring(labels.readline().strip(), dtype=float, sep=" "), dtype=T.float)
            config = T.tensor(np.array(list(configs.readline().strip()), dtype=int)).type(T.float)
            if self.transform:
                config = config.reshape(1, self.side, self.side)
                config = self.transform(config)
            if self.target_transform:
                label = self.target_transform(label)
            return (config, label)
        
class CustomAutoencoderDataset3D(T.utils.data.Dataset):
    def __init__(self, dataFile:str, side:int, transform=None, target_transform=None):
        self.file = dataFile;

        self.string = "temps"
                # seekSkip is the number of characters to skip to reach the next line
        self.dataSeekSkip = 6+1 #7+1 also

        if os.path.isfile(self.file+"temps")==False:
            self.string = "params"
            self.dataSeekSkip = 13+1
                
        self.length = lineNumber(self.file+self.string)
        self.side = side
        self.side3 = side*side*side

        self.configSeekSkip = self.side3+1 # config+\n
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with open(self.file + "configs", 'r') as configs, open(self.file + self.string, 'r') as labels:
            if idx >= self.length:
                raise IndexError
            configs.seek(idx*self.configSeekSkip, os.SEEK_SET)
            labels.seek(idx*self.dataSeekSkip, os.SEEK_SET)
            label = T.tensor(np.fromstring(labels.readline().strip(), dtype=float, sep=" "), dtype=T.float)
            config = T.tensor(np.array(list(configs.readline().strip()), dtype=int)).type(T.float)
            config = config.reshape(1, self.side, self.side, self.side)
            if self.transform:
                config = self.transform(config)
            if self.target_transform:
                label = self.target_transform(label)
            return (config, label)