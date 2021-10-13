from torchvision.datasets import ImageFolder
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import torch
import numpy as np

rootdir = "../../output/"
transform = transforms.Compose([
    transforms.ToTensor()
    ])
ds = ImageFolder(rootdir, transform = transform)
ds_train, ds_val = torch.utils.data.random_split(ds, [len(ds) - 3000, 3000])
dl_train = DataLoader(ds_train, batch_size = 16)
dl_val = DataLoader(ds_val, batch_size = 16)
datadir = Path("./transfer/vision/cls")

def savenp(dl, split):
    datal = []
    labell = []

    for data, label in tqdm(dl) :
        datal.append(data)
        labell.append(label)

    datatensor = torch.cat(datal, dim = 0)
    labeltensor = torch.cat(labell,dim=0)

    datatensor = datatensor.permute(0,2,3,1).repeat(1,1,1,2) * 255.0


    print(datatensor.shape)

    print(labeltensor.shape)

    datanp = datatensor.numpy()
    labelnp = labeltensor.numpy()
    np.save(str(datadir/f"x_{split}.npy"),datanp)
    np.save(str(datadir/f"y_{split}.npy"),labelnp)
    print(str(datadir/f"x_{split}.npy"))

savenp(dl_train,"train")
savenp(dl_val,"test")

