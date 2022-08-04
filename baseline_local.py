# Modified from: https://github.com/pliang279/LG-FedAvg/blob/master/main_fed.py
# credit goes to: Paul Pu Liang

# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

# This program implements FedRep under the specification --alg fedrep, as well as Fed-Per (--alg fedper), LG-FedAvg (--alg lg),
# FedAvg (--alg fedavg) and FedProx (--alg prox)

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

import copy
import warnings
from collections import OrderedDict
from typing import List

import numpy as np
import ray
import torch
import torch.optim as optim
from opacus import PrivacyEngine
from torch import nn
from torch.utils.data import DataLoader

from Models.models import get_model
from common_utils import accuracy, seed_all, make_private
from data_utils import prepare_dataloaders
from options import args_parser

warnings.filterwarnings("ignore")

class Client:
    def __init__(self, cid, model: nn.Module, train_dataloader: DataLoader, test_dataloader: DataLoader, args):
        self.cid = cid # client id
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.args = args
        if not args.disable_dp:
            self.privacy_engine = PrivacyEngine(secure_mode=args.secure_rng)
        else:
            self.privacy_engine = None
            self.optimizer_noise_multiplier = 0

        if args.verbose:
            print(
                f"Client {cid} has {len(train_dataloader.dataset)} training samples"
            )

    def step(self):
        train_loss, train_top1_acc = self.train()

        test_loss, test_top1_acc = self.test()

        return (train_loss, train_top1_acc, test_loss, test_top1_acc)

    def train(self):
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(),
                              lr=args.lr,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay
                              )
        if not args.disable_dp:
            model, optimizer, train_dataloader = make_private(self.args, self.privacy_engine, self.model, optimizer,
                                                              self.train_dataloader)
        else:
            model = self.model
            train_dataloader = self.train_dataloader

        for local_epoch in range(self.args.local_epochs):
            losses = []
            top1_acc = []
            for _batch_idx, (data, target) in enumerate(train_dataloader):
                data, target = data.to(self.args.device), target.to(self.args.device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                model.zero_grad()
                losses.append(loss.item())
                preds = np.argmax(output.detach().cpu().numpy(), axis=1)
                labels = target.detach().cpu().numpy()
                acc = accuracy(preds, labels)
                top1_acc.append(acc)

            if args.verbose:
                print(f"Client {self.cid}, Local Epoch: {local_epoch} \t Loss: {np.mean(losses):.6f}"
                      f"\t Acc@1: {np.mean(top1_acc) * 100:.6f} "
                      )

        if not args.disable_dp:
            sdd = model.state_dict()
            sdd = OrderedDict([(key[8:], sdd[key]) for key in sdd.keys()])
            self.model.load_state_dict(sdd)
        # otherwise, model is self.model so it is already updated

        train_loss = np.mean(losses, keepdims=True)
        train_top1_acc = np.mean(top1_acc, keepdims=True)

        return train_loss, train_top1_acc

    def test(self):
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        test_loss = np.zeros((1,))
        correct = np.zeros((1,))
        with torch.no_grad():
            for data, target in self.test_dataloader:
                data, target = data.to(self.args.device), target.to(self.args.device)
                output = self.model(data)
                test_loss += criterion(output, target).item()  # sum up batch loss
                pred = output.argmax(
                    dim=1, keepdim=True
                )  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        return (test_loss / len(self.test_dataloader.dataset), correct / len(self.test_dataloader.dataset), )

def aggregate_model(args, global_model, clients: List[Client]):
    return global_model # no global model in the local training setting

def broadcast(args, global_model, clients: List[Client]):
    return 0 # no need to broadcase the global model to the clients in the local training setting

@ray.remote(num_gpus=.5)
def ray_dispatch(client: Client):
    result = client.step()
    print(f"Client {client.cid} finished!")
    return result

def train(args, global_model, clients: List[Client]):
    broadcast(args, global_model, clients)
    # loss_and_accs = [client.step() for client in clients]
    result_id = [ray_dispatch.remote(client) for client in clients]
    results = ray.get(result_id)
    train_losses = [result[0] for result in results]
    train_top1_accs = [result[1] for result in results]
    test_losses = [result[2] for result in results]
    test_top1_accs = [result[3] for result in results]

    global_average_train_loss = np.mean(np.concatenate(train_losses))
    global_average_train_top1_acc = np.mean(np.concatenate(train_top1_accs))
    global_average_test_loss = np.mean(np.concatenate(test_losses))
    global_average_test_top1_acc = np.mean(np.concatenate(test_top1_accs))


    return global_model, global_average_train_loss, global_average_train_top1_acc, global_average_test_loss, global_average_test_top1_acc


if __name__ == '__main__':
    # parse args
    args = args_parser()
    seed_all(args.seed)
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    run_results = []
    for _ in range(args.n_runs):
        # initialize the model
        global_model = get_model(args)
        train_dataloaders, test_dataloaders, _ = prepare_dataloaders(args)

        # Define N = args.num_users.
        # There are N users, each one should have (model, optimizer, lr_scheduler, train_dataloader, test_dataloader).
        # No need for privacy engine since we are in the local training setting
        models = [copy.deepcopy(global_model) for _ in range(args.num_users)]
        clients = [Client(idx, model, trdlr, tedlr, args) for idx, (model, trdlr, tedlr) in enumerate(zip(models, train_dataloaders, test_dataloaders))]


        _, global_average_train_loss, global_average_train_top1_acc, global_average_test_loss, global_average_test_top1_acc \
            = train(args, global_model, clients)

        print(f"Train Loss: {global_average_train_loss:.6f}"
              f"\t Train Acc@1: {global_average_train_top1_acc * 100:.6f} "
              )
        print(f"Test Loss: {global_average_test_loss:.6f}"
              f"\t Test Acc@1: {global_average_test_top1_acc * 100:.6f} "
              )

        # No need for global test in the local training setting
        # final_test_result = test_global(args, global_model, global_test_dataloader)
        run_result = [global_average_train_loss, global_average_train_top1_acc, global_average_test_loss, global_average_test_top1_acc]
        run_results.append(run_result)

    # if len(run_results) > 1:
    #     print(
    #         "Accuracy averaged over {} runs: {:.2f}% ± {:.2f}%".format(
    #             len(run_results), np.mean(run_results) * 100, np.std(run_results) * 100
    #         )
    #     )

    # repro_str = (
    #     f"mnist_{args.lr}_{args.sigma}_"
    #     f"{args.max_per_sample_grad_norm}_{args.batch_size}_{args.epochs}"
    # )
    # torch.save(run_results, f"run_results_{repro_str}.pt")
    #
    # if args.save_model:
    #     torch.save(model.state_dict(), f"mnist_cnn_{repro_str}.pt")


