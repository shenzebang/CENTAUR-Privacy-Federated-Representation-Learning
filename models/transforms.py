from torchvision import transforms

trans_mnist_normalization = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

trans_emnist_normalization = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=0.1736, std=0.3317)])

trans_fashionmnist_normalization = transforms.Compose([transforms.ToTensor(),])

trans_cifar10_normalization = transforms.Compose([  transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])])
trans_cifar10_augmentation = transforms.Compose([   transforms.RandomCrop(32, padding=4),
                                                    transforms.RandomHorizontalFlip(),])

trans_cifar100_augmentation = transforms.Compose([  transforms.RandomCrop(32, padding=4),
                                                    transforms.RandomHorizontalFlip(),])

trans_cifar100_normalization = transforms.Compose([ transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                                              std=[0.267, 0.256, 0.276])])

