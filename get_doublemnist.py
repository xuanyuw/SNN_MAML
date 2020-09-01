from torchmeta.datasets.helpers import doublemnist

train_dataset = doublemnist('data', shots=10, ways=5, shuffle=True, meta_split = "train", test_shots=15, download=True)
test_dataset = doublemnist('data', shots=1, ways=5, shuffle=True, meta_split = "val", test_shots=15, download=True)