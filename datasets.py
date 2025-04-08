import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from medmnist import INFO
import medmnist


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
ALL_DATASETS = ['MNIST'
                ,'FashionMNIST'
                ,'SVHN'
                ,'CIFAR10'
                ,'CIFAR100'
                ,'PathMNIST'
                ,'ChestMNIST'
                ,'DermaMNIST'
                ,'OCTMNIST'
                ,'BloodMNIST'
                ,'BreastMNIST'
                ,'PneumoniaMNIST'
                ,'RetinaMNIST'
                ,'TissueMNIST'
                ,'OrganAMNIST'
                ,'OrganCMNIST'
                ,'OrganSMNIST'
                ]
def get_dataset(dataset, data_path, **kvargs):
    if dataset == 'MNIST':
        channel = 1
        im_size = 28
        num_classes = 10
        ds_train = datasets.MNIST(data_path, train=True, download=True) # no augmentation
        ds_test = datasets.MNIST(data_path, train=False, download=True)
        class_names = [str(c) for c in range(num_classes)]
        transform = lambda x: transforms.functional.to_tensor(x) * 2 -1
        label_transform = OneHotEncoding(num_classes)
    elif dataset == 'FashionMNIST':
        channel = 1
        im_size = 28
        num_classes = 10
        ds_train = datasets.FashionMNIST(data_path, train=True, download=True) # no augmentation
        ds_test = datasets.FashionMNIST(data_path, train=False, download=True)
        class_names = ds_train.classes
        transform = lambda x: transforms.functional.to_tensor(x) * 2 -1
        label_transform = OneHotEncoding(num_classes)
    elif dataset == 'SVHN':
        channel = 3
        im_size = 32
        num_classes = 10
        ds_train = datasets.SVHN(data_path, split='train', download=True)  # no augmentation
        ds_test = datasets.SVHN(data_path, split='test', download=True)
        class_names = [str(c) for c in range(num_classes)]
        transform = lambda x: transforms.functional.to_tensor(x) * 2 -1
        label_transform = OneHotEncoding(num_classes)
    elif dataset == 'CIFAR10':
        channel = 3
        im_size = 32
        num_classes = 10
        ds_train = datasets.CIFAR10(data_path, train=True, download=True) # no augmentation
        ds_test = datasets.CIFAR10(data_path, train=False, download=True)
        class_names = ds_train.classes
        transform = lambda x: transforms.functional.to_tensor(x) * 2 -1
        label_transform = OneHotEncoding(num_classes)
    elif dataset == 'CIFAR100':
        channel = 3
        im_size = 32
        num_classes = 100
        ds_train = datasets.CIFAR100(data_path, train=True, download=True) # no augmentation
        ds_test = datasets.CIFAR100(data_path, train=False, download=True)
        class_names = ds_train.classes
        transform = lambda x: transforms.functional.to_tensor(x) * 2 -1
        label_transform = OneHotEncoding(num_classes)
    elif dataset in ['PathMNIST'
                    ,'ChestMNIST'
                    ,'DermaMNIST'
                    ,'OCTMNIST'
                    ,'BloodMNIST'
                    ,'BreastMNIST'
                    ,'PneumoniaMNIST'
                    ,'RetinaMNIST'
                    ,'TissueMNIST'
                    ,'OrganAMNIST'
                    ,'OrganCMNIST'
                    ,'OrganSMNIST'
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
        'class_names': class_names
    }
