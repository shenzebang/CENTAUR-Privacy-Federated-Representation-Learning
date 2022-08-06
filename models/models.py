from torch import nn
import torch.nn.functional as F
from opacus.validators import ModuleValidator
import torchvision

class CNNCifar10(nn.Module):
    def __init__(self, args):
        super(CNNCifar10, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout(0.6)
        self.conv2 = nn.Conv2d(64, 64, 5)
        self.fc1 = nn.Linear(64 * 5 * 5, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, args.num_classes)
        self.cls = args.num_classes

        self.weight_keys = [['fc1.weight', 'fc1.bias'],
                            ['fc2.weight', 'fc2.bias'],
                            ['fc3.weight', 'fc3.bias'],
                            ['conv2.weight', 'conv2.bias'],
                            ['conv1.weight', 'conv1.bias'],
                            ]

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CNNCifar10_BN(nn.Module):  # with batch normalization
    def __init__(self, args):
        super(CNNCifar10_BN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.conv1_bn = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout(0.6)
        self.conv2 = nn.Conv2d(64, 64, 5)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc1_bn = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.fc2_bn = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, args.num_classes)
        self.cls = args.num_classes

        self.weight_keys = [['fc1.weight', 'fc1.bias'],
                            ['fc2.weight', 'fc2.bias'],
                            ['fc3.weight', 'fc3.bias'],
                            ['conv2.weight', 'conv2.bias'],
                            ['conv1.weight', 'conv1.bias'],
                            ['fc1_bn.weight', 'fc1_bn.bias', 'fc1_bn.running_mean', 'fc1_bn.running_var',
                             'fc1_bn.num_batches_tracked'],
                            ['fc2_bn.weight', 'fc2_bn.bias', 'fc2_bn.running_mean', 'fc2_bn.running_var',
                             'fc2_bn.num_batches_tracked'],
                            ['conv1_bn.weight', 'conv1_bn.bias', 'conv1_bn.running_mean', 'conv1_bn.running_var',
                             'conv1_bn.num_batches_tracked'],
                            ['conv2_bn.weight', 'conv2_bn.bias', 'conv2_bn.running_mean', 'conv2_bn.running_var',
                             'conv2_bn.num_batches_tracked']
                            ]

    def forward(self, x):
        x = self.pool(F.relu(self.conv1_bn(self.conv1(x))))
        x = self.pool(F.relu(self.conv2_bn(self.conv2(x))))
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = F.relu(self.fc2_bn(self.fc2(x)))
        x = self.fc3(x)
        return x

def get_model(args):
    if args.model == 'cnn' and 'cifar100' in args.dataset:
        raise NotImplementedError
    elif args.model == 'cnn' and 'cifar10' in args.dataset:
        if args.norm == 'None':
            net_glob = CNNCifar10(args=args)
        elif args.norm == 'batch_norm':
            net_glob = CNNCifar10_BN(args=args)
            net_glob = ModuleValidator.fix(net_glob)
    elif args.model == 'resnet' and 'cifar10' in args.dataset:
        net_glob = torchvision.models.resnet18(num_classes=args.num_classes)
        if not args.disable_dp:
            net_glob = ModuleValidator.fix(net_glob)
    elif args.model == 'mlp' and 'mnist' in args.dataset:  # both mnist and femnist
        raise NotImplementedError
    elif args.model == 'cnn' and 'femnist' in args.dataset:
        raise NotImplementedError
    elif args.model == 'mlp' and 'cifar' in args.dataset:
        raise NotImplementedError
    elif 'sent140' in args.dataset:
        if args.model == 'lstm':
            raise NotImplementedError
        if args.model == 'mlp':
            raise NotImplementedError
    elif args.model == 'bert' and 'harass' in args.dataset:
        raise NotImplementedError
    elif args.model == 'robert' and 'harass' in args.dataset:
        raise NotImplementedError
    else:
        exit('Error: unrecognized model')
    # print(net_glob)

    return net_glob