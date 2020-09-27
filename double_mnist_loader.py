import random
import h5py
import pickle
import torch
from math import floor
from io import BytesIO
from torchvision import transforms 
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler


pic_per_class = 1000

def split_data(class_keys, pic_per_class, ways, support_shots, test_shots):
    n = floor((len(class_keys) * pic_per_class) / ways)
    label_pool = list(class_keys)*pic_per_class
    support_labels = []
    test_labels = []
    for i in range(n):
        samp = random.sample(label_pool, k=ways)
        support_labels.append(samp)
        tsamp = random.choices(samp, k=test_shots)
        test_labels.append(tsamp)

    indx_pool = list(range(pic_per_class)) * len(class_keys)
    test_pool = list(range(pic_per_class)) * len(class_keys)
    support_indx = []
    test_indx = []
    for j in range(n):
        samp = random.sample(indx_pool, k=support_shots)
        support_indx.append(samp)
        tsamp = random.choices(test_pool, k=test_shots)
    return {'support': list(zip(support_labels, support_indx)), 'test':list(zip(test_labels, test_indx))}

class DoubleMNIST(Dataset):
    def __init__(self, data_file, transform=None, is_paired_file=True):
        self.transform = transform
        if is_paired_file:
            try:
                self.samples = pickle.load(data_file)
            except pickle.PickleError:
                print('The file must be pickled first, please change is_paired_file to True and try again')
        else:
            if not h5py.is_hdf5(data_file):
                raise ValueError('Not hdf5 file')    
            self.paires = []
            self.samples = []
            f = h5py.File(data_file)
            dset = f[list(f.keys())[0]]
            for k in dset.keys():
                for img in dset.get(k):
                    if transform:
                        img = transform(img)
                    self.samples.append((k,img))
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples(idx)      



        




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
a = [['a', 'b'], ['c', 'd']]
b = [[1, 2], [3, 4]]
list(zip(a, b))
# %%
