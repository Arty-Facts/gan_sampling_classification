import medmnist.dataset
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from medmnist import INFO
import medmnist
import random
import numpy as np
from torch.utils.data import Sampler
import math

def seed_everything(seed=None):
    if seed is None:
        return 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

ALL_DATASETS = ['MNIST'
                ,'FashionMNIST'
                ,'SVHN'
                ,'CIFAR10'
                ,'CIFAR100'
                ,'PathMNIST'
                ,'DermaMNIST'
                # ,'OCTMNIST'
                ,'BloodMNIST'
                ,'BreastMNIST'
                # ,'PneumoniaMNIST'
                ,'RetinaMNIST'
                # ,'TissueMNIST'
                # ,'OrganAMNIST'
                ,'OrganCMNIST'
                # ,'OrganSMNIST'
                ]
class OneHotEncoding:
    def __init__(self, num_classes):
        self.num_classes = num_classes
    
    def __call__(self, label):
        one_hot = torch.zeros(self.num_classes)
        one_hot[label] = 1
        return one_hot
    
def get_label(dataset):
    if isinstance(dataset, (datasets.MNIST, datasets.FashionMNIST, datasets.CIFAR10, datasets.CIFAR100)):
        return torch.tensor(dataset.targets) if not isinstance(dataset.targets, torch.Tensor) else dataset.targets
    elif isinstance(dataset, datasets.SVHN):
        return torch.tensor(dataset.labels)
    elif isinstance(dataset, medmnist.dataset.MedMNIST):
        return torch.tensor(dataset.labels).reshape(-1)
    else:
        raise ValueError(f'Invalid dataset: {dataset}')

class CoolDataset():
    def __init__(self,name, dataset, image_shape, label_shape, channels, transform, lable_transform, class_names, reduce_level=None, seed=None):
        self.name = name
        self.dataset = dataset
        self.labels = get_label(dataset)
        self.image_shape = (channels, image_shape, image_shape)
        self.label_shape = label_shape
        self.has_labels = False
        self.transform=transform
        self.lable_transform=lable_transform
        self.class_names = class_names
        self.indices_per_class = [
            torch.where(self.labels == i)[0] for i in range(label_shape)
        ]

        if reduce_level is not None:
            seed_everything(seed)
            assert len(reduce_level) == len(self.indices_per_class), f"{name}: {len(reduce_level)} != {len(self.indices_per_class)}"
            indices_per_class = []
            for numb, indexs in zip(reduce_level, self.indices_per_class):
                indices = torch.randperm(len(indexs))[:numb]
                sampled = indexs[indices]
                indices_per_class.append(sampled)
            self.indices_per_class = indices_per_class
        self.lookup = torch.concatenate(self.indices_per_class)
        assert len(self.lookup) == len(torch.unique(self.lookup)), f"{name}: there are duplicate indexes after subsampling"
        self.data_per_class = [len(self.indices_per_class[i]) for i in range(label_shape)]
        self.sorted_data_per_class = sorted(
            zip(range(self.label_shape), self.data_per_class),
            key=lambda x: x[1], reverse=True
            )

    def __getitem__(self, index):
        index = self.lookup[index]
        img, label = self.dataset[index]
        return self.transform(img), self.lable_transform(label)
    
    def get_label(self, index):
        index = self.lookup[index]
        return self.lable_transform(self.labels[index])
    
    def get_random_index_for_label(self, label):
        index = torch.randint(0, len(self.indices_per_class[label]), (1,)) + sum([self.data_per_class[i] for i in range(label)])
        return index.item()
    
    def __len__(self):
        return len(self.lookup)
    
    def __repr__(self):
        return  f"{self.name} - " + ", ".join([f"{name}:{numb}" for name, numb in self.sorted_data_per_class])
    
class InfiniteClassSampler(Sampler):
    def __init__(self, dataset: CoolDataset, batch_size, balanced=True, seed=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.balanced = balanced
        self.num_classes = dataset.label_shape
        self.class_indices = dataset.indices_per_class
        self.data_per_class = dataset.data_per_class
        self.seed = seed

        seed_everything(seed)

    def __iter__(self):
        while True:
            if not self.balanced:
                # Random shuffle of dataset indices
                indices = torch.randperm(len(self.dataset))
                for i in range(0, len(indices), self.batch_size):
                    yield indices[i:i + self.batch_size].tolist()
            else:
                if self.batch_size > self.num_classes:
                    samples_per_class = self.batch_size // self.num_classes
                    for _ in range(len(self.dataset) // self.batch_size):  # can loop forever if needed
                        batch = []
                        for label, class_idxs in enumerate(self.class_indices):
                            if len(class_idxs) >= samples_per_class:
                                selected = torch.randperm(len(class_idxs))[:samples_per_class]  + sum([self.data_per_class[i] for i in range(label)])
                            else:
                                selected = torch.randint(0, len(class_idxs), (samples_per_class,))+ sum([self.data_per_class[i] for i in range(label)])
                            batch.extend(selected.tolist())
                        while len(batch) < self.batch_size:
                            label = random.randint(0, self.num_classes - 1)
                            idx = self.dataset.get_random_index_for_label(label)
                            batch.append(idx)
                        random.shuffle(batch)
                        yield batch
                else:
                    chosen_classes = random.sample(range(self.num_classes), self.batch_size)
                    batch = []
                    for label in chosen_classes:
                        class_idxs = self.class_indices[label]
                        idx = torch.randint(0, len(class_idxs), (1,)) + sum([self.data_per_class[i] for i in range(label)])
                        batch.append(idx)
                    yield batch



    def __len__(self):
        return 2 ** 31  # practically infinite

def get_dataset(dataset, data_path, reduce_level=None, sampling_strategy="none", seed=None, **kvargs):
    if sampling_strategy not in ["none", "class_balence"]:
        raise ValueError(f"Unsuported sampling_strategy {sampling_strategy}")
    if dataset == 'MNIST':
        channel = 1
        im_size = 28
        num_classes = 10
        ds_train = datasets.MNIST(data_path, train=True, download=True) # no augmentation
        ds_test = datasets.MNIST(data_path, train=False, download=True)
        class_names = [str(c) for c in range(num_classes)]
        transform = lambda x: transforms.functional.to_tensor(x) * 2 -1
        label_transform = OneHotEncoding(num_classes)
        if reduce_level == 1:
            reduce_level = torch.linspace(100, 1000, num_classes).long()
    elif dataset == 'FashionMNIST':
        channel = 1
        im_size = 28
        num_classes = 10
        ds_train = datasets.FashionMNIST(data_path, train=True, download=True) # no augmentation
        ds_test = datasets.FashionMNIST(data_path, train=False, download=True)
        class_names = ds_train.classes
        transform = lambda x: transforms.functional.to_tensor(x) * 2 -1
        label_transform = OneHotEncoding(num_classes)
        if reduce_level == 1:
            reduce_level = torch.linspace(100, 1000, num_classes).long()
    elif dataset == 'SVHN':
        channel = 3
        im_size = 32
        num_classes = 10
        ds_train = datasets.SVHN(data_path, split='train', download=True)  # no augmentation
        ds_test = datasets.SVHN(data_path, split='test', download=True)
        class_names = [str(c) for c in range(num_classes)]
        transform = lambda x: transforms.functional.to_tensor(x) * 2 -1
        label_transform = OneHotEncoding(num_classes)
        if reduce_level == 1:
            reduce_level = torch.linspace(100, 1000, num_classes).long()
    elif dataset == 'CIFAR10':
        channel = 3
        im_size = 32
        num_classes = 10
        ds_train = datasets.CIFAR10(data_path, train=True, download=True) # no augmentation
        ds_test = datasets.CIFAR10(data_path, train=False, download=True)
        class_names = ds_train.classes
        transform = lambda x: transforms.functional.to_tensor(x) * 2 -1
        label_transform = OneHotEncoding(num_classes)
        if reduce_level == 1:
            reduce_level = torch.linspace(100, 1000, num_classes).long()
    elif dataset == 'CIFAR100':
        channel = 3
        im_size = 32
        num_classes = 100
        ds_train = datasets.CIFAR100(data_path, train=True, download=True) # no augmentation
        ds_test = datasets.CIFAR100(data_path, train=False, download=True)
        class_names = ds_train.classes
        transform = lambda x: transforms.functional.to_tensor(x) * 2 -1
        label_transform = OneHotEncoding(num_classes)
        if reduce_level == 1:
            reduce_level = torch.linspace(100, 500, num_classes).long()
    elif dataset in ['PathMNIST'
                    ,'DermaMNIST'
                    # ,'OCTMNIST'
                    ,'BloodMNIST'
                    ,'BreastMNIST'
                    # ,'PneumoniaMNIST'
                    ,'RetinaMNIST'
                    # ,'TissueMNIST'
                    # ,'OrganAMNIST'
                    ,'OrganCMNIST'
                    # ,'OrganSMNIST'
                    ]:
        info = INFO[dataset.lower()]    
        channel = info['n_channels']
        im_size = kvargs.get("size", 28)
        num_classes = len(info["label"])
        DataClass = getattr(medmnist, info['python_class'])
        ds_train = DataClass(split='train',download=True, size=im_size, root=data_path)
        ds_test = DataClass(split='test',download=True, size=im_size, root=data_path)
        class_names = list(info["label"].values())
        transform = lambda x: (transforms.functional.to_tensor(x)  - 0.5) / 0.5
        label_transform = OneHotEncoding(num_classes)
        if reduce_level == 1:
            if dataset == 'PathMNIST':
                reduce_level = torch.tensor([3000, 5000, 6000, 7000, 2000, 8000, 1000, 4000, 10_000]).long()
            if dataset == 'BloodMNIST':
                reduce_level = torch.tensor([200, 1500, 400, 1000, 100, 300, 2000, 800]).long()
            if dataset == 'OrganCMNIST':
                reduce_level = torch.tensor([700, 300, 100, 200, 600, 800, 1500, 400, 500, 900, 1000]).long()
            if dataset == 'RetinaMNIST':
                reduce_level = torch.tensor([400, 100, 200, 150, 50]).long()
            if dataset == 'BreastMNIST':
                reduce_level = torch.tensor([100, 300]).long()
            if dataset == 'DermaMNIST':
                reduce_level = torch.tensor([200, 300, 600, 50, 700, 1000, 50]).long()
    else:
        raise ValueError(f'Invalid dataset: {dataset}')
    ds_test = CoolDataset(dataset+"_test", ds_test, im_size, num_classes, channel, transform, label_transform, class_names)
    ds_train = CoolDataset(dataset+"_train", ds_train, im_size, num_classes, channel, transform, label_transform, class_names, reduce_level, seed)
    return {
        'channel': channel,
        'im_size': im_size,
        'num_classes': num_classes,
        'train': ds_train,
        'test': ds_test,
        'class_names': class_names
    }
