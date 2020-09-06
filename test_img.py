from torchmeta.datasets.helpers import doublemnist
from torchmeta.utils.data import BatchMetaDataLoader


test_dataset = doublemnist("data", ways=5, shots=1, test_shots=15, meta_train=True, download=True)
test_dataloader = BatchMetaDataLoader(test_dataset, batch_size=16, num_workers=4)

#for batch in test_dataloader:
#    input, label = batch['test']
#    print('input shape = {}'.format(input.shape))
#    print('label shape = {}'.format(label.shape)

iter_loader = iter(test_dataloader)
input, label = next(iter_loader)['val']
print(tuple(input.shape))