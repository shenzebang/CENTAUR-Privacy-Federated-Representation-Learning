import copy
from transformers import AutoTokenizer, RobertaTokenizer
import os
import numpy as np
from torch.utils.data import DataLoader, Dataset
import json
from filelock import FileLock
from torchvision import datasets
from typing import List

from models.transforms import trans_mnist, trans_cifar10_train, trans_cifar10_val, trans_cifar100_train, \
    trans_cifar100_val

import random
import torch

import pandas as pd


def get_transform(dataset_name, train=True, use_data_augmentation=False):
    if dataset_name == 'mnist':
        transform = trans_mnist
    elif dataset_name == 'cifar10':
        if train:
            transform = trans_cifar10_train if use_data_augmentation else trans_cifar10_val
        else:
            transform = trans_cifar10_val
    elif dataset_name == 'cifar100':
        if train:
            transform = trans_cifar100_train if use_data_augmentation else trans_cifar100_val
        else:
            transform = trans_cifar100_val
    else:
        raise NotImplementedError

    return transform


def get_dataset(dataset_name, train=True, transform=None):
    with FileLock(os.path.expanduser("~/.data.lock")):
        if dataset_name == 'mnist':
            dataset = datasets.MNIST('~/data/mnist/', train=train, download=True, transform=transform)
        elif dataset_name == 'cifar10':
            dataset = datasets.CIFAR10('~/data/cifar10', train=train, download=True, transform=transform)
        elif dataset_name == 'cifar100':
            dataset = datasets.CIFAR100('~/data/cifar100', train=train, download=True, transform=transform)
        else:
            raise NotImplementedError

    return dataset


def prepare_dataloaders(args):
    #   For now, only CIFAR10/CIFAR100 are implemented.
    num_users_orig = args.num_users

    # TODO: if args.num_users == 1, i.e. the centralized setting, the data allocation part can be simplified.

    ####################################################################################################################
    ###########################    CIFAR and MNIST   ###################################################################
    ####################################################################################################################
    if 'cifar' in args.dataset or args.dataset == 'mnist':
        '''
            1. Load the train/test datasets, create a user to data point map "dict_users_train"/"dict_users_test"
        '''
        ground_dataset_train, ground_dataset_test, dict_users_train, dict_users_test, dict_users_class = get_data(args)
        for idx in dict_users_train.keys(): np.random.shuffle(dict_users_train[idx])

        '''
            2. Prepare the train dataloader
        '''

        # Configure the transform that we want to use for training
        if args.dataset == 'cifar10':
            transform_multiplicity = trans_cifar10_train if args.data_augmentation else trans_cifar10_val
        elif args.dataset == 'cifar100':
            transform_multiplicity = trans_cifar100_train if args.data_augmentation else trans_cifar100_val
        else:
            raise NotImplementedError

        if args.data_augmentation:  # use DeepMind data augmentation
            '''
                In DP setting, directly add data augmentation to the training procedure deteriorates the utility.
                If "args.data_augmentation" is set to True, we will implement data augmentation as described in 
                the DeepMind paper. 
            '''
            print(
                "[ Using data augmentation implementation as described in the DeepMind paper! ]"
            )
        else:
            print(
                "[ Since there is no data augmentation, pre-process the dataset to improve the efficiency. ]"
            )
            # create a tensordataset
            iter_ground_dataset_train = iter(ground_dataset_train)
            imgs = []
            targets = []
            for img, target in iter_ground_dataset_train:
                imgs.append(transform_multiplicity(img))
                targets.append(target)

            # replace "ground_dataset_train" with the preprocessed TensorDataset for efficiency
            imgs = torch.stack(imgs, dim=0)
            targets = torch.tensor(targets)
            ground_dataset_train = torch.utils.data.TensorDataset(imgs, targets)
            transform_multiplicity = None  # disable transform_multiplicity since all data points are pre-processed

        # Wrap DatasetSplit with DatasetMultiplicity to produce multiple augmented images from a single image
        def make_dataset(_dataset_train, _dict_users_train_uid):
            return DatasetMultiplicity(
                    DatasetSplit(_dataset_train, _dict_users_train_uid),
                    transform_multiplicity,
                    args.data_augmentation_multiplicity
                    )

        train_dataloaders = []
        bs_too_large = False
        for uid in range(args.num_users):
            dataset_train_uid = make_dataset(ground_dataset_train, dict_users_train[uid])
            batch_size = min(args.batch_size, len(dataset_train_uid))
            bs_too_large = False if args.batch_size <= len(dataset_train_uid) and not bs_too_large else True
            train_dataloaders.append(DataLoader(dataset_train_uid,
                                        batch_size=batch_size,
                                        num_workers=0,
                                        # pin_memory=True,
                                        shuffle=True
                                        ))
        if bs_too_large:
            print(
                f"[ The train batch size is larger than the size of the local training dataset. ]"
            )

        '''
            3. Prepare the test dataloader
        '''
        # Configure the transform that we want to use for testing
        if args.dataset == 'cifar10':
            transform_test = trans_cifar10_val
        elif args.dataset == 'cifar100':
            transform_test = trans_cifar100_val
        else:
            raise NotImplementedError


        # Since there is no data augmentation, pre-process the dataset to improve the efficiency.
        iter_ground_dataset_test = iter(ground_dataset_test)
        imgs = []
        targets = []
        for img, target in iter_ground_dataset_test:
            imgs.append(transform_test(img))
            targets.append(target)

        imgs = torch.stack(imgs, dim=0)
        targets = torch.tensor(targets)
        ground_dataset_test = torch.utils.data.TensorDataset(imgs, targets)
        transform_test = None  # disable transform_multiplicity since all data points have been pre-processed

        make_dataset = lambda _dataset, _dict_users_uid: \
            DatasetMultiplicity(
                DatasetSplit(_dataset, _dict_users_uid),
                transform_test,
                0
            )

        test_dataloaders = []
        for uid in range(args.num_users):
            dataset_test_uid = make_dataset(ground_dataset_test, dict_users_test[uid])
            batch_size = min(args.test_batch_size, len(dataset_test_uid))
            test_dataloaders.append(DataLoader(dataset_test_uid,
                                batch_size=batch_size,
                                num_workers=0,
                                # pin_memory=True,
                                shuffle=False
                            ))



    ####################################################################################################################
    ###########################    harass   ############################################################################
    ####################################################################################################################
    elif args.dataset == 'harass':
        if args.model == 'bert':
            # Load the BERT tokenizer
            tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        elif args.model == 'robert':
            tokenizer = RobertaTokenizer.from_pretrained('roberta-base', truncation=True, do_lower_case=True)
        dataset_train, dataset_test, dict_users_train, dict_users_test = get_data(args, tokenizer=tokenizer)
        lens = []
        for idx in dict_users_train.keys():
            lens.append(len(dict_users_train[idx]))
        if args.arc == 'central':
            dict_users_train_orig = copy.deepcopy(dict_users_train)
            dict_users_test_orig = copy.deepcopy((dict_users_test))
            dataset_train_orig = dataset_train
            dataset_test_orig = dataset_test
            # dict_users_train = {0: [i for i in range(len(dataset_train))]}
            # dict_users_test = {0: [i for i in range(len(dataset_test))]}
            dict_users_train = {0: [item for idx in dict_users_train.keys() for item in dict_users_train[idx]]}
            dict_users_test = {0: [item for idx in dict_users_test.keys() for item in dict_users_test[idx]]}
            np.random.shuffle(dict_users_train[0])
            args.num_users_under = args.num_users
            args.num_users = 1
        else:
            for idx in dict_users_train.keys():
                np.random.shuffle(dict_users_train[idx])
    ####################################################################################################################
    ###########################    LEAF   ##############################################################################
    ####################################################################################################################
    else:  # leaf datasets
        if args.dataset == 'sent140':
            train_path = './leaf-master/data/' + args.dataset + '/data/train'
            test_path = './leaf-master/data/' + args.dataset + '/data/test'
        elif args.dataset == 'femnist':
            train_path = './leaf-master/data/' + args.dataset + '/data/train_' + str(num_users_orig) + '_' + str(
                args.shard_per_user)
            test_path = './leaf-master/data/' + args.dataset + '/data/test_' + str(num_users_orig) + '_' + str(
                args.shard_per_user)
        else:
            exit('Error: unrecognized dataset')

        clients, groups, dataset_train, dataset_test = read_data(train_path, test_path)
        # print([len(dataset_train[c]['x']) for c in dataset_train])
        # print([len(dataset_test[c]['x']) for c in dataset_test])
        lens = []
        for iii, c in enumerate(clients):
            lens.append(len(dataset_train[c]['x']))
        dict_users_train = list(dataset_train.keys())  # actually a list, while dataset_train is a dictionary
        dict_users_test = list(dataset_test.keys())
        sort_mask = sorted(range(len(lens)), key=lambda k: lens[k], reverse=True)
        sort_mask = sort_mask[:num_users_orig]
        lens = [lens[index] for index in sort_mask]
        dict_users_train = [dict_users_train[index] for index in sort_mask]  # actually a list
        dict_users_test = [dict_users_test[index] for index in sort_mask]
        lens_test = [len(dataset_test[c]['x']) for c in dict_users_test]
        clients = [clients[index] for index in sort_mask]
        print(lens)  # train
        print(lens_test)  # test

        for c in dataset_train.keys():
            dataset_train[c]['y'] = list(np.asarray(dataset_train[c]['y']).astype('int64'))
            dataset_test[c]['y'] = list(np.asarray(dataset_test[c]['y']).astype('int64'))

        if args.arc == 'central':
            dict_users_train_orig = copy.deepcopy(dict_users_train)
            dict_users_test_orig = copy.deepcopy(dict_users_test)
            sort_mask_orig = copy.deepcopy(sort_mask)
            sort_mask = [0]
            dict_users_train = ['central']
            dict_users_test = ['central']
            dataset_train_central = {'central': {}}
            dataset_test_central = {'central': {}}
            dataset_train_central['central']['x'] = sum([dataset_train[user]['x'] for user in dataset_train.keys()], [])
            dataset_test_central['central']['x'] = sum([dataset_test[user]['x'] for user in dataset_test.keys()], [])
            dataset_train_central['central']['y'] = sum([dataset_train[user]['y'] for user in dataset_train.keys()], [])
            dataset_test_central['central']['y'] = sum([dataset_test[user]['y'] for user in dataset_test.keys()], [])
            dataset_train_orig = copy.deepcopy(dataset_train)
            dataset_test_orig = copy.deepcopy(dataset_test)
            dataset_train = dataset_train_central
            dataset_test = dataset_test_central
            args.num_users_under = args.num_users
            args.num_users = 1


    return train_dataloaders, test_dataloaders


def get_data(args, tokenizer=None):
    # The "dataset.transform" will be set when creating the dataloader from the dataset
    with FileLock(os.path.expanduser("~/.data.lock")):
        if args.dataset == 'mnist':
            dataset_train = datasets.MNIST('~/data/mnist/', train=True, download=True)
            dataset_test = datasets.MNIST('~/data/mnist/', train=False, download=True)
            # sample users
            if args.iid:
                print('iid')
                dict_users_train = iid(dataset_train, args.num_users)
                dict_users_test = iid(dataset_test, args.num_users)
            else:
                dict_users_train, user_to_classes = noniid(dataset_train, args.num_users, args.shard_per_user, args.num_classes, args.sample_size_var)
                dict_users_test = noniid(dataset_test, args.num_users, args.shard_per_user, args.num_classes, args.sample_size_var, user_to_classes=user_to_classes)
        elif args.dataset == 'cifar10':
            dataset_train = datasets.CIFAR10('~/data/cifar10', train=True, download=True)
            dataset_test = datasets.CIFAR10('~/data/cifar10', train=False, download=True)
            if args.iid:
                dict_users_train = iid(dataset_train, args.num_users)
                dict_users_test = iid(dataset_test, args.num_users)
            else:
                dict_users_train, user_to_classes = noniid(dataset_train, args.num_users, args.shard_per_user, args.num_classes, args.sample_size_var)
                dict_users_test = noniid(dataset_test, args.num_users, args.shard_per_user, args.num_classes, args.sample_size_var, user_to_classes=user_to_classes)
        elif args.dataset == 'cifar100':
            dataset_train = datasets.CIFAR100('~/data/cifar100', train=True, download=True)
            dataset_test = datasets.CIFAR100('~/data/cifar100', train=False, download=True)
            if args.iid:
                dict_users_train = iid(dataset_train, args.num_users)
                dict_users_test = iid(dataset_test, args.num_users)
            else:
                dict_users_train, user_to_classes = noniid(dataset_train, args.num_users, args.shard_per_user, args.num_classes, args.sample_size_var)
                dict_users_test = noniid(dataset_test, args.num_users, args.shard_per_user, args.num_classes, args.sample_size_var, user_to_classes=user_to_classes)
        elif args.dataset == 'harass':
            df = pd.read_csv('data/Sexual_Harassment_Data/Harassment_Cleaned_tweets.csv')
            df.head()

            # Creating the dataset and dataloader for the neural network
            train_size = 0.8
            train_dataset = df.sample(frac=train_size, random_state=200)
            test_dataset = df.drop(train_dataset.index).reset_index(drop=True)
            train_dataset = train_dataset.reset_index(drop=True)

            # print("FULL Dataset: {}".format(df.shape))
            # print("TRAIN Dataset: {}".format(train_dataset.shape))
            # print("TEST Dataset: {}".format(test_dataset.shape))

            MAX_LEN = 256

            dataset_train = Triage(train_dataset, tokenizer, MAX_LEN)
            dataset_test = Triage(test_dataset, tokenizer, MAX_LEN)

            # dict_users_train = noniid_2classes(dataset_train, args.num_users)
            # print([len(dict_users_train[i]) for i in dict_users_train])
            # dict_users_test = noniid_2classes(dataset_test, args.num_users)
            # print([len(dict_users_test[i]) for i in dict_users_test])
            dict_users_train, user_to_classes = noniid_triage_2class(dataset_train, args.num_users, args.shard_per_user, args.num_classes)
            dict_users_test, user_to_classes = noniid_triage_2class(dataset_test, args.num_users, args.shard_per_user, args.num_classes,
                                                   user_to_classes=user_to_classes, testb=True)

            return dataset_train, dataset_test, dict_users_train, dict_users_test

        else:
            exit('Error: unrecognized dataset')

    return dataset_train, dataset_test, dict_users_train, dict_users_test, user_to_classes

def iid(dataset, num_users):
    """
    Sample I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    num_items_per_client = int(len(dataset) / num_users)
    image_idxs = [i for i in range(len(dataset))]

    for i in range(num_users):
        dict_users[i] = np.random.choice(image_idxs, num_items_per_client, replace=False)
        image_idxs = list(set(image_idxs) - set(dict_users[i]))

    return dict_users

def noniid(dataset, num_users, shard_per_user, num_classes, sample_size_var, user_to_classes = None):
    if sample_size_var > 0:
        return noniid_diff_size(dataset, num_users, shard_per_user, num_classes, user_to_classes)

    dict_users = {uid: [] for uid in range(num_users)}

    class_to_sample_ids = {
        _class: [] for _class in range(num_classes) # class_id : id list of samples that has class_id
    }


    # transform the dataset into a dictionary based on labels
    for i in range(len(dataset)):
        label = torch.tensor(dataset.targets[i]).item()
        if label < num_classes:
            class_to_sample_ids[label].append(i)

    shortest = min([len(class_to_sample_ids[_class]) for _class in class_to_sample_ids.keys()])

    if shortest == 0:
        raise ValueError('This dataset does not have sufficient number of classes!')

    # create test set
    if user_to_classes is not None:
        for uid in range(num_users):
            for _class in user_to_classes[uid]:
                dict_users[uid].extend(class_to_sample_ids[_class])
        return dict_users

    shard_per_class = int(shard_per_user * num_users / num_classes)
    class_to_shards = {}
    for label in class_to_sample_ids.keys():
        x = class_to_sample_ids[label]
        random.shuffle(x)

        """x = np.array(x[:usable])
        x = x.reshape((shard_per_class, -1))"""

        num_leftover = len(x) % shard_per_class
        leftover = x[-num_leftover:] if num_leftover > 0 else []
        x = np.array(x[:-num_leftover]) if num_leftover > 0 else np.array(x)
        x = x.reshape((shard_per_class, -1))

        x = list(x)

        # assign the leftovers to each shard until none left.
        for i, idx in enumerate(leftover): x[i] = np.concatenate([x[i], [idx]])

        class_to_shards[label] = x  # label to shards (content of each shard) mapping

    if user_to_classes is None:  # create a user-to-shard mapping
        success = False
        n_attempt = 99
        while not success:
            n_attempt += 1
            user_to_classes = list(range(num_classes)) * shard_per_class  # 2-D matrix representing all shards
            random.shuffle(user_to_classes)
            user_to_classes = np.array(user_to_classes).reshape((num_users, -1))

            # check if there is repeated class for a user
            success = check_repeat(user_to_classes)
            if n_attempt == 100:
                success = True
                print(
                    "Fail to ensure no repeated class for all users!"
                )

    # assign
    for uid in range(num_users):
        classes_uid = user_to_classes[uid]
        uid_to_shards = []
        for _class in classes_uid:  # label of each assigned shard
            if len(class_to_shards[_class]) == 0:
                raise ValueError("Insufficient data for sharding!")
            sid = np.random.choice(len(class_to_shards[_class]))  # take a shard from that class
            uid_to_shards.append(class_to_shards[_class].pop(sid))  # no replacement
        dict_users[uid] = list(np.concatenate(uid_to_shards))  # combine and flatten the shards

    return dict_users, user_to_classes

def check_repeat(user_to_classes: np.ndarray):
    num_users = user_to_classes.shape[0]
    num_shards_per_user = user_to_classes.shape[1]
    for uid in range(num_users):
        if len(np.unique(user_to_classes[uid, :])) != num_shards_per_user: # repeat identified
            return False
    return True

def noniid_diff_size(dataset, num_users, shard_per_user, num_classes, user_to_classes=[]):
    """
    non I.I.D parititioning of data over clients
    Sort the data by the digit label
    Divide each class into various sizes of shards
    """
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    idxs_dict = {}
    count = 0

    # transform the dataset into a dictionary based on labels
    for i in range(len(dataset)):
        label = torch.tensor(dataset.targets[i]).item()
        if label < num_classes and label not in idxs_dict.keys():
            idxs_dict[label] = []
        if label < num_classes:
            idxs_dict[label].append(i)
            count += 1

    shard_per_class = int(
        shard_per_user * num_users / num_classes)  # shard: a user with a class, shard_per_user: "classes" per user, shard_per_class: "users" per class

    for label in idxs_dict.keys():
        x = idxs_dict[label]
        avg_shard_size = int(np.floor(len(x) / shard_per_class))
        min_shard_size = int(np.ceil(avg_shard_size/2))

        # split data of each class into equal-sized shards. the remainders are taken out
        # when the class sizes are different, the shard sizes may vary too
        num_leftover = len(x) % shard_per_class
        leftover = x[-num_leftover:] if num_leftover > 0 else []
        x = np.array(x[:-num_leftover]) if num_leftover > 0 else np.array(x)
        x = x.reshape((shard_per_class, -1))
        x = list(x)

        # assign the leftovers to each shard until none left.
        for i, idx in enumerate(leftover):
            x[i] = np.concatenate([x[i], [idx]])

        for i in range(0, shard_per_class, 2):
            if i != (shard_per_class-1):  # not the last shard
                give_size = random.randint(0, len(x[i])-min_shard_size)
                x[i+1] = np.concatenate([x[i+1], x[i][:give_size]])
                x[i] = x[i][give_size:]

        x.sort(key=len)

        idxs_dict[label] = x  # label to shards (content of each shards) mapping

    if len(user_to_classes) == 0:  # create a user-to-shard mapping
        user_to_classes = list(range(shard_per_class*num_classes))
        random.shuffle(user_to_classes)
        user_to_classes = np.array(user_to_classes).reshape((num_users, -1))

    # divide and assign
    for i in range(num_users):
        rand_set_shards = user_to_classes[i]
        rand_set = []
        for shard in rand_set_shards:
            label = int(np.floor(shard/shard_per_class))
            shard_idx = shard % shard_per_class  # index of shard within a class
            rand_set.append(idxs_dict[label][shard_idx])
        dict_users[i] = np.concatenate(rand_set)  # combine and flatten the shards

    sizes = []
    for i in range(num_users):
        sizes.append(len(dict_users[i]))
    print(sizes)

    return dict_users, user_to_classes



def read_data(train_data_dir, test_data_dir):
    '''parses data in given train and test data directories
    assumes:
    - the data in the input directories are .json files with
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users

    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    '''
    clients = []
    groups = []
    train_data = {}
    test_data = {}

    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith('.json')]
    for f in train_files:
        file_path = os.path.join(train_data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        train_data.update(cdata['user_data'])

    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith('.json')]
    for f in test_files:
        file_path = os.path.join(test_data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        test_data.update(cdata['user_data'])

    clients = list(train_data.keys())

    return clients, groups, train_data, test_data

class DatasetSplit(Dataset):
    def __init__(self, dataset: Dataset, sample_ids: List[int], name=None):
        self.dataset = dataset
        self.sample_ids = sample_ids
        self.name = name

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, item):
        if self.name is None:
            image, label = self.dataset[self.sample_ids[item]]
        elif 'femnist' in self.name:
            image = torch.reshape(torch.tensor(self.dataset['x'][item]), (1, 28, 28))
            label = torch.tensor(self.dataset['y'][item])
        elif 'sent140' in self.name:
            image = self.dataset['x'][item]
            label = self.dataset['y'][item]
        elif 'harass' in self.name:
            return self.dataset[self.sample_ids[item]]
        else:
            image, label = self.dataset[self.sample_ids[item]]
        return image, label

class DatasetSplit2Device(Dataset):
    def __init__(self, d_split: DatasetSplit, device='cpu'):
        assert d_split.name == None # only CIFAR10/CIFAR100/MNIST are supported for now

        self.images, self.labels = self.move_to_device(d_split, device)
        self.len = len(d_split)

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        return self.images[item], self.labels[item]

    def move_to_device(self, d_split: DatasetSplit, device):
        image_and_label = [d_split.dataset[d_split.sample_ids[i]] for i in range(len(d_split))]
        images = torch.stack([image for image, _ in image_and_label])
        labels = torch.stack([torch.tensor(label) for _, label in image_and_label])

        return images.to(device), labels.to(device)

class DatasetMultiplicity(Dataset):
    def __init__(self, d_split: DatasetSplit, transform, multiplicity: int = 0):
        assert d_split.name == None  # only CIFAR10/CIFAR100/MNIST are supported for now
        self.d_split = d_split
        self.transform = transform
        self.multiplicity = multiplicity
        self._disable_multiplicity = False

    def __len__(self):
        return len(self.d_split)

    def __getitem__(self, item):
        # assume that the original dataset (CIFAR10/CIFAR100/MNIST) performs no transform!
        img, target = self.d_split[item]

        if not self._disable_multiplicity and self.multiplicity > 1:
            # create multiple images from a single raw image with data augmentation
            imgs = torch.stack([self.transform(img) for _ in range(self.multiplicity)], dim=0)
            targets = torch.tensor(target).repeat(self.multiplicity)
            return imgs, targets
        else:
            if self.transform is not None:
                img, target = self.transform(img), torch.tensor(target)
            return img, target

    def disable_multiplicity(self):
        self._disable_multiplicity = True

    def enable_multiplicity(self):
        self._disable_multiplicity = False

class Triage(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def get_labels(self):
        l = []
        for i in range(self.len):
            title = str(self.data.Text[i])
            title = " ".join(title.split())
            inputs = self.tokenizer.encode_plus(
                title,
                None,
                add_special_tokens=True,
                max_length=self.max_len,
                padding='max_length',
                return_token_type_ids=True,
                truncation=True
            )

            l.append(self.data.Label[i])

        return l

    def __getitem__(self, index):
        title = str(self.data.Text[index])
        title = " ".join(title.split())
        inputs = self.tokenizer.encode_plus(
            title,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.data.Label[index], dtype=torch.long)
        }

    def __len__(self):
        return self.len

def noniid_triage_2class(dataset, num_users, shard_per_user, num_classes, user_to_classes=[], testb=False):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    idxs_dict = {}
    count = 0

    # transform the dataset into a dictionary based on labels
    labels = dataset.get_labels()
    for i in range(len(dataset)):
        label = labels[i]
        if label < num_classes and label not in idxs_dict.keys():
            idxs_dict[label] = []
        if label < num_classes:
            idxs_dict[label].append(i)
            count += 1

    # each user has 1 shard from one class, then other shards from the other class
    shard_per_class = {}
    shard_per_class[0] = int((shard_per_user - 1) * np.ceil(num_users / 2) + np.floor(num_users / 2))
    shard_per_class[1] = int((shard_per_user - 1) * np.floor(num_users / 2) + np.ceil(num_users / 2))

    samples_per_user = int(count / num_users)
    # whether to sample more test samples per user
    if (samples_per_user < 20) and testb:
        repeat = int(np.ceil(20 / samples_per_user))
    else:
        repeat = 1

    for label in idxs_dict.keys():
        x = idxs_dict[label]

        # split data of each classes into equal-sized shards. the remainders are taken out
        # when the class sizes are different, the shard sizes may vary too
        num_leftover = len(x) % shard_per_class[label]
        leftover = x[-num_leftover:] if num_leftover > 0 else []
        x = np.array(x[:-num_leftover]) if num_leftover > 0 else np.array(x)
        x = x.reshape((shard_per_class[label], -1))
        x = list(x)

        # assign the leftovers to each shard until none left.
        for i, idx in enumerate(leftover):
            x[i] = np.concatenate([x[i], [idx]])

        idxs_dict[label] = x  # label to shards (content of each shards) mapping

    if len(user_to_classes) == 0:
        user_to_classes = []
        for user in range(num_users):
            shards = []
            if user < np.floor(num_users / 2):
                shards.append(0)
                for i in range(shard_per_user - 1):
                    shards.append(1)
            else:
                shards.append(1)
                for i in range(shard_per_user - 1):
                    shards.append(0)
            random.shuffle(shards)
            user_to_classes.append(shards)
        random.shuffle(user_to_classes)

    user_to_classes = np.array(user_to_classes)

    # divide and assign
    for i in range(num_users):
        if repeat > 1:
            rand_set_label = list(user_to_classes[i]) * repeat  # more shards from each assigned class
        else:
            rand_set_label = user_to_classes[i]
        rand_set = []
        for label in rand_set_label:  # label of each assigned shard
            idx = np.random.choice(len(idxs_dict[label]), replace=False)  # take a shard from that class, no replacement for the inner loop
            if (repeat > 1) and testb:
                rand_set.append(idxs_dict[label][idx])  # have replacement for the outer loop
            else:
                rand_set.append(idxs_dict[label].pop(idx))  # no replacement for the outer loop
        dict_users[i] = np.concatenate(rand_set)  # combine and flatten the shards

    return dict_users, user_to_classes
