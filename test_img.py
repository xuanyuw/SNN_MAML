#%%
from torchmeta.datasets.helpers import doublemnist
from torchmeta.utils.data import BatchMetaDataLoader
from torchmeta.transforms import Categorical, ClassSplitter, Rotation
from torch.utils.data import DataLoader

test_dataset = doublemnist("data", ways=5, shots=1, test_shots=1,
                            meta_train=True, download=True)
#test_dataset = ClassSplitter(test_dataset, shuffle=True,
#                                num_train_per_class=5,
#                                num_test_per_class=15)
#test_dataloader = BatchMetaDataLoader(test_dataset, batch_size=2, num_workers=4)
test_dataloader = BatchMetaDataLoader(test_dataset, batch_size=10, num_workers=4)
#for batch in test_dataloader:
#    input, label = batch['test']
#    print('input shape = {}'.format(input.shape))
#    print('label shape = {}'.format(label.shape)

iter_loader = iter(test_dataloader)
input, label = next(iter_loader)['train']
print(tuple(input.shape))
print(type(input))
#%%
iter_loader = iter(test_dataloader)
next(iter_loader)
#%%
input, label = next(iter_loader)['test']

#print(tuple(input.shape))
print(label)
# %%
import numpy as np
import matplotlib.pyplot as plt

for i in range(2):
    for j in range(3):
        # this converts it from GPU to CPU and selects first image
        img = input.numpy()[i][j]
        #convert image back to Height,Width,Channels
        img = np.transpose(img, (1,2,0))
        #show the image
        plt.imshow(img)
        plt.show()  
# %%
