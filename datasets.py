import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

def get_label(dataset):
    if isinstance(dataset, (datasets.MNIST, datasets.FashionMNIST, datasets.CIFAR10, datasets.CIFAR100)):
        return torch.tensor(dataset.targets)
    elif isinstance(dataset, datasets.SVHN):
        return torch.tensor(dataset.labels)
    else:
        raise ValueError(f'Invalid dataset: {dataset}')
class OneHotEncoding:
    def __init__(self, num_classes):
        self.num_classes = num_classes
    
    def __call__(self, label):
        one_hot = torch.zeros(self.num_classes)
        one_hot[label] = 1
        return one_hot
    
class CoolDataset():
    def __init__(self, dataset, image_shape, label_shape, channels, transform, lable_transform):
        self.dataset = dataset
        self.image_shape = (channels, image_shape, image_shape)
        self.label_shape = label_shape
        self.has_labels = False
        self.transform=transform
        self.lable_transform=lable_transform

    def __getitem__(self, index):
        img, label = self.dataset[index]
        return self.transform(img), self.lable_transform(label)
    def get_label(self, index):
        return self[index][1]

    def __len__(self):
        return len(self.dataset)
    
def get_dataset(dataset, data_path, transform=None):
    if dataset == 'MNIST':
        channel = 1
        im_size = 28
        num_classes = 10
        ds_train = datasets.MNIST(data_path, train=True, download=True, transform=transform) # no augmentation
        ds_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)
        class_names = [str(c) for c in range(num_classes)]
    elif dataset == 'FashionMNIST':
        channel = 1
        im_size = 28
        num_classes = 10
        ds_train = datasets.FashionMNIST(data_path, train=True, download=True, transform=transform) # no augmentation
        ds_test = datasets.FashionMNIST(data_path, train=False, download=True, transform=transform)
        class_names = ds_train.classes
    elif dataset == 'SVHN':
        channel = 3
        im_size = 32
        num_classes = 10
        ds_train = datasets.SVHN(data_path, split='train', download=True, transform=transform)  # no augmentation
        ds_test = datasets.SVHN(data_path, split='test', download=True, transform=transform)
        class_names = [str(c) for c in range(num_classes)]

    elif dataset == 'CIFAR10':
        channel = 3
        im_size = 32
        num_classes = 10
        ds_train = datasets.CIFAR10(data_path, train=True, download=True, transform=transform) # no augmentation
        ds_test = datasets.CIFAR10(data_path, train=False, download=True, transform=transform)
        class_names = ds_train.classes
        transform = lambda x: transforms.functional.to_tensor(x) * 2 -1
        label_transform = OneHotEncoding(num_classes)
    elif dataset == 'CIFAR100':
        channel = 3
        im_size = 32
        num_classes = 100
        ds_train = datasets.CIFAR100(data_path, train=True, download=True, transform=transform) # no augmentation
        ds_test = datasets.CIFAR100(data_path, train=False, download=True, transform=transform)
        class_names = ds_train.classes
    else:
        raise ValueError(f'Invalid dataset: {dataset}')
    ds_test = CoolDataset(ds_test, im_size, num_classes, channel, transform, label_transform)
    ds_train = CoolDataset(ds_train, im_size, num_classes, channel, transform, label_transform)
    return {
        'channel': channel,
        'im_size': im_size,
        'num_classes': num_classes,
        'train': ds_train,
        'test': ds_test,
        'label_train': get_label(ds_train.dataset),
        'label_test': get_label(ds_test.dataset),
        'class_names': class_names
    }
