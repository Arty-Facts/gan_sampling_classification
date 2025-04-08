import medmnist.dataset
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from medmnist import INFO
import medmnist


ALL_DATASETS = ['MNIST'
                ,'FashionMNIST'
                ,'SVHN'
                ,'CIFAR10'
                ,'CIFAR100'
                ,'PathMNIST'
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
    def __init__(self,name, dataset, image_shape, label_shape, channels, transform, lable_transform, class_names):
        self.name = name
        self.dataset = dataset
        self.lables = get_label(dataset)
        self.image_shape = (channels, image_shape, image_shape)
        self.label_shape = label_shape
        self.has_labels = False
        self.transform=transform
        self.lable_transform=lable_transform
        self.data_per_class = [(self.lables == i).sum() for i in range(label_shape)]
        self.class_names = class_names
        self.sorted_data_per_class = sorted(zip(range(self.label_shape), self.data_per_class), key=lambda x: x[1], reverse=True)

    def __getitem__(self, index):
        img, label = self.dataset[index]
        return self.transform(img), self.lable_transform(label)
    
    def get_label(self, index):
        return self.lable_transform(self.lables[index])

    def __len__(self):
        return len(self.dataset)
    
    def __repr__(self):
        return  f"{self.name} - " + ", ".join([f"{name}:{numb}" for name, numb in self.sorted_data_per_class])
    


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
    ds_test = CoolDataset(dataset+"_test", ds_test, im_size, num_classes, channel, transform, label_transform, class_names)
    ds_train = CoolDataset(dataset+"_train", ds_train, im_size, num_classes, channel, transform, label_transform, class_names)
    return {
        'channel': channel,
        'im_size': im_size,
        'num_classes': num_classes,
        'train': ds_train,
        'test': ds_test,
        'class_names': class_names
    }
