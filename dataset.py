import torch
import torchvision
from torchvision import transforms as transforms
from torch.utils.data import random_split

from lib.alibi import RandomizedLabelPrivacy
from lib.canary import fill_canaries

import numpy as np

class NoisedCIFAR(torch.utils.data.Dataset):
    def __init__(
        self,
        cifar: torch.utils.data.Dataset,
        num_classes: int,
        randomized_label_privacy: RandomizedLabelPrivacy,
    ):
        self.cifar = cifar
        self.rlp = randomized_label_privacy
        targets = [cifar.dataset.targets[index] for index in cifar.indices]
        self.soft_targets = [self._noise(t, num_classes) for t in targets]
        self.rlp.increase_budget()  # increase budget
        # calculate probability of label change
        num_label_changes = sum(
            label != torch.argmax(soft_target).item()
            for label, soft_target in zip(targets, self.soft_targets)
        )
        self.label_change = num_label_changes / len(targets)

    def _noise(self, label, n):
        onehot = torch.zeros(n).to()
        onehot[label] = 1
        rand = self.rlp.noise((n,))
        return onehot if rand is None else onehot + rand

    def __len__(self):
        return self.cifar.__len__()

    def __getitem__(self, index):
        image, label = self.cifar.__getitem__(index)
        return image, self.soft_targets[index], label

def load_CIFAR10(batch_size):
    CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR10_STD_DEV = (0.2023, 0.1994, 0.2010)
    train_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD_DEV)]) #transforms.RandomHorizontalFlip(), 
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD_DEV)])
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)
    labelNames = ["airplane", "automobile", "bird", "cat", "dear", "dog", "frog", "horse", "ship", "truck"]
    return train_loader, test_loader, labelNames

def load_CIFAR100(batch_size):
    CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
    CIFAR100_STD_DEV = (0.2675, 0.2565, 0.2761)
    train_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD_DEV)]) #transforms.RandomHorizontalFlip(), 
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD_DEV)])
    train_set = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    test_set = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def load_FedCIFAR10(batch_size, num_workers):
    CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR10_STD_DEV = (0.2023, 0.1994, 0.2010)
    train_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD_DEV)]) #transforms.RandomHorizontalFlip(), 
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD_DEV)])
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    len_train = len(train_set)
    len_client_data = {worker_id: int(len_train / num_workers) for worker_id in range(num_workers)}
    if sum(len_client_data.values()) != len_train:
        dif = len_train - sum(len_client_data.values())
        idxs = np.random.choice(num_workers, size=dif)
        for idx in idxs:
            len_client_data[idx] += 1
    
    rs = random_split(train_set, list(len_client_data.values()))

    train_loaders = [(torch.utils.data.DataLoader(dataset=rs[worker_id], batch_size=batch_size, shuffle=True), len(rs[worker_id]) / len_train) for worker_id in range(num_workers)]
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)
    labelNames = ["airplane", "automobile", "bird", "cat", "dear", "dog", "frog", "horse", "ship", "truck"]
    return train_loaders, test_loader, labelNames

def load_FedCIFAR10_LAPLACE(args, randomized_label_privacy):
    CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR10_STD_DEV = (0.2023, 0.1994, 0.2010)
    train_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD_DEV)]) #transforms.RandomHorizontalFlip(), 
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD_DEV)])
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    len_train = len(train_set)
    len_client_data = {worker_id: int(len_train / args.NUM_WORKERS) for worker_id in range(args.NUM_WORKERS)}
    if sum(len_client_data.values()) != len_train:
        dif = len_train - sum(len_client_data.values())
        idxs = np.random.choice(args.NUM_WORKERS, size=dif)
        for idx in idxs:
            len_client_data[idx] += 1
    
    rs = random_split(train_set, list(len_client_data.values()))
    rs = [rand_label_privacy_process(ds, randomized_label_privacy, args, 10) for ds in rs]

    train_loaders = [(torch.utils.data.DataLoader(dataset=rs[worker_id], batch_size=args.BATCH_SIZE, shuffle=True), len(rs[worker_id]) / len_train) for worker_id in range(args.NUM_WORKERS)]
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=args.BATCH_SIZE, shuffle=False)
    labelNames = ["airplane", "automobile", "bird", "cat", "dear", "dog", "frog", "horse", "ship", "truck"]
    return train_loaders, test_loader, labelNames

def rand_label_privacy_process(dataset, randomized_label_privacy, args, num_classes):
    if args.CANARY > 0 and args.CANARY < len(dataset):
        # capture debug info
        original_label_sum = sum([dataset.dataset.targets[index] for index in dataset.indices])
        original_last10_labels = [dataset[-i][1] for i in range(1, 11)]
        # inject canaries
        dataset = fill_canaries(
            dataset, num_classes, N=args.CANARY, seed=args.SEED
        )
        # capture debug info
        canary_label_sum = sum([dataset.dataset.targets[index] for index in dataset.indices])
        canary_last10_labels = [dataset[-i][1] for i in range(1, 11)]
        # verify presence
        if original_label_sum == canary_label_sum:
            raise Exception(
                "Canary infiltration has failed."
                f"\nOriginal label sum: {original_label_sum} vs"
                f" Canary label sum: {canary_label_sum}"
                f"\nOriginal last 10 labels: {original_last10_labels} vs"
                f" Canary last 10 labels: {canary_last10_labels}"
            )
    if args.NOISE_ONLY_ONCE:
        dataset = NoisedCIFAR(
            dataset, num_classes, randomized_label_privacy
        )
    return dataset

def load_data(args, randomized_label_privacy=None):
    if args.DATA_NAME == 'CIFAR10':
        return load_CIFAR10(args.BATCH_SIZE)
    elif args.DATA_NAME == 'CIFAR100':
        return load_CIFAR100(args.BATCH_SIZE)
    elif args.DATA_NAME == 'FedCIFAR10':
        return load_FedCIFAR10(args.BATCH_SIZE, num_workers=args.NUM_WORKERS)
    elif args.DATA_NAME == 'FedCIFAR10LAPLACE':
        return load_FedCIFAR10_LAPLACE(args, randomized_label_privacy)
    else:
        raise ValueError('Unknown data name: {}'.format(args.DATA_NAME))