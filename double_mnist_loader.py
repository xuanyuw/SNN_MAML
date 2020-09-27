import random
import h5py
import pickle
import torch
from io import BytesIO
from torchvision import transforms 
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler


pic_per_class = 1000


class DoubleMNIST(Dataset):
    def __init__(self, data_file, transform=None, is_paired_file=True):
        self.transform = transform
        if is_meta_file:
            try:
                self.sample = pickle.load(data_file)
            except PickleError:
                print('The file must be pickled first, please change is_paired_file to True and try again')
        else:
            if not h5py.is_hdf5(data_file):
                raise ValueError('Not hdf5 file')    
            self.paires = []
            self.sample = []
            f = h5py.File(data_file)
            dset = f[list(f.keys())[0]]
            for k in dset.keys():
                for img in dset.get(k):
                    if transform:
                        img = transform(img)
                    self.samples.append((k,img))
    def __len__(self):
        return len(self.sample)

    def __getitem__(self, idx):
        return self.sample(idx)      

class BatchRandomSampler(RandomSampler):
    def __init__(self, data_source, shots, batch_size, replacement=False, generator=None):
        self.data_source = data_source
        self.replacement = replacement
        self.shots = shots
        self.batch_size = batch_size
        self.generator = generator
    def num_

        




#%%
import h5py

f = h5py.File('data/doublemnist/train_data.hdf5', 'r')
list(f.keys())
# %%
dset = f['datasets']
type(dset)
# %%
list(dset.keys())
# %%
it = iter(dset.items())
grp = next(it)
# %%
grp[1][0]
# %%
import matplotlib.pyplot as plt
plt.imshow(grp[1][0])
# %%
import numpy as np

np.sqrt(544)
# %%
import io
import PIL.Image as Image

image = Image.open(io.BytesIO(grp[1][0]))
image.show()
# %%
dset.shape
# %%
grp[0]
# %%
len(grp[1])
# %%
dset.items()[0]
# %%
len(dset.get('30'))

# %%
testset = dset.get('30')
type(testset)
# %%
t = [None]*10
t[1].append((1, 2))
t[2] = [(2, 3), (4, 5)]
t
# %%

# %%
