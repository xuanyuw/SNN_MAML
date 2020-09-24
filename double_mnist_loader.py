import h5py
import torch
from io import BytesIO
from torchvision import transforms 
from torch.utils.data import DataLoader, Dataset


pic_per_class = 1000

class DoubleMNIST(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
    def __len__(self):
        return (len(dset), pic_per_class)
    def __getitem__(self, index):
        #index should be a tuple: (<label_index>,<image_index>)
        key = list(self.data.keys())[index[0]]
        raw_img = self.data.get(key)[index[1]]
        image = Image.open(io.BytesIO(raw_img))
        if self.transform:
            image = self.transform(image)
        return (key, image)
            
class 
        



#%%
import h5py

f = h5py.File('data/doublemnist/train_data.hdf5', 'r')
list(f.keys())
# %%
dset = f['datasets']
type(dset)
# %%
list(dset.keys())[0]
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
len(dset.get('00'))

# %%
