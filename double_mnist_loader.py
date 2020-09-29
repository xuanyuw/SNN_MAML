
import random
import h5py
import pickle
import torch
import io
from math import floor
import PIL.Image as Image
from torchvision import transforms 
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from tqdm import tqdm


pic_per_class = 1000

def split_data(class_keys, pic_per_class, ways, support_shots, test_shots):
    n = floor((len(class_keys) * pic_per_class) / ways)
    label_pool = list(class_keys)*pic_per_class
    support_labels = []
    test_labels = []
    print("separating labels")
    
    for i in tqdm(range(n)):
        samp = random.sample(label_pool, k=ways)
        support_labels.append(samp)
        tsamp = random.choices(samp, k=test_shots)
        test_labels.append(tsamp)
    indx_pool = list(range(pic_per_class)) * len(class_keys)
    test_pool = list(range(pic_per_class)) * len(class_keys)
    support_indx = []
    test_indx = []
    print('separating indices')
    for j in tqdm(range(n)):
        samp = random.sample(indx_pool, k=support_shots*ways)
        support_indx.append(samp)
        tsamp = random.choices(test_pool, k=test_shots)
        test_indx.append(tsamp)
    return {'support': list(zip(support_labels, support_indx)), 'test':list(zip(test_labels, test_indx))}

class DoubleMNIST(Dataset):
    # structure: ('support: [([labels], tensor(images)), (...) ...], 'test': [(...), ...])
    def __init__(self, data_file, ways, support_shots, test_shots, is_paired_file=True):
        if is_paired_file:
            try:
                self.samples = pickle.load(data_file)
            except pickle.PickleError:
                print('The file must be pickled first, please change is_paired_file to True and try again')
        else:
            if not h5py.is_hdf5(data_file):
                raise ValueError('Not a hdf5 file')    
            self.paires = []
            self.samples = []
            f = h5py.File(data_file)
            dset = f[list(f.keys())[0]]
            keys = dset.keys()
            indx_set = split_data(keys, pic_per_class, ways, support_shots, test_shots)
            print('load up support sets')
            s_li = []
            for i in tqdm(indx_set['support']):
            #for i in indx_set['support'][:10]:
                support_li = ()
                for j in range(len(i[1])):
                    lb = i[0][j]
                    img = dset.get(lb)[i[1][j]]
                    img = Image.open(io.BytesIO(img))
                    img_t = transforms.ToTensor()(img).unsqueeze_(0)
                    support_li = support_li + (img_t,)
                support_imgs = torch.cat(support_li)
                s_li.append({'support':(i[0], support_imgs)})
            print('load up test sets')
            for i in tqdm(range(len(indx_set['test']))):
            #for i in range(10):
                test_li = ()
                for j in range(len(indx_set['test'][i][0])):
                    lb = indx_set['test'][i][0][j]
                    img = dset.get(lb)[indx_set['test'][i][1][j]]
                    img = Image.open(io.BytesIO(img))
                    img_t = transforms.ToTensor()(img).unsqueeze_(0)
                    test_li = test_li + (img_t,)
                test_imgs = torch.cat(test_li)
                s_li[i].update({'test': (indx_set['test'][i][0], test_imgs)})
    
            self.samples = s_li
                    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]      

d = DoubleMNIST('data/doublemnist/train_data.hdf5', 5, 1, 1, False)

samp = d[0]

samp
