import warnings


import os

import numpy as np
import torch
import torch.optim as optim
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm

from models.models import get_model
from utils.common_utils import accuracy, seed_all, make_private
from options import args_parser
from utils.data_utils import get_dataset, get_transform
from collections import OrderedDict


warnings.filterwarnings("ignore")

def test_configuration(args, model_state):
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # Init DataLoader
    transform = get_transform(args.dataset, train=False)
    dataset_test = get_dataset(args.dataset, train=False, transform=transform)

    test_loader = torch.utils.data.DataLoader(dataset_test,
                                              batch_size=args.test_batch_size,
                                              shuffle=True,
                                              num_workers=1,
                                              pin_memory=True,
                                              )

    # Init the model
    model = get_model(args).to(device)

    # Load the trained model_state
    model.load_state_dict(model_state)

    # Test the performance
    return test(model, device, test_loader)

def test(model: nn.Module, device, test_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )
    return correct / len(test_loader.dataset)

def train(args, model, device, train_loader, optimizer, privacy_engine, epoch):
    model.train()
    criterion = nn.CrossEntropyLoss()
    losses = []
    top1_acc = []

    with BatchMemoryManager(
            data_loader=train_loader,
            max_physical_batch_size=args.MAX_PHYSICAL_BATCH_SIZE,
            optimizer=optimizer
    ) as memory_safe_data_loader:

        for _batch_idx, (data, target) in enumerate(memory_safe_data_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()
            acc = accuracy(preds, labels)
            top1_acc.append(acc)

    if epoch % 10 == 9:
        if not args.disable_dp:
            epsilon, best_alpha = privacy_engine.accountant.get_privacy_spent(
                delta=args.delta
            )
            print(
                f"Train Epoch: {epoch} \t"
                f"Loss: {np.mean(losses):.6f} "
                f"Acc@1: {np.mean(top1_acc) * 100:.6f} "
                f"(ε = {epsilon:.2f}, δ = {args.delta}) for α = {best_alpha}"
            )
        else:
            print(f"Train Epoch: {epoch} \t Loss: {np.mean(losses):.6f}"
                  f"\t Acc@1: {np.mean(top1_acc) * 100:.6f} "
                  )

    return np.mean(losses), np.mean(top1_acc)


def main(args, checkpoint_dir=None):
    if args.seed != 0:
        seed_all(args.seed)
        if args.verbose:
            print(
                f"Seed is set to {args.seed} to ensure reproducibility"
            )
    else:
        if args.verbose:
            print(
                f"No seed is manually set."
            )

    # Init DataLoader
    transform = get_transform(args.dataset, train=True, use_data_augmentation=args.data_augmentation)
    dataset_train = get_dataset(args.dataset, train=True, transform=transform)

    if args.verbose:
        print(
            f"Training on dataset {args.dataset}."
        )

    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    train_loader = torch.utils.data.DataLoader(dataset_train,
                                               batch_size=args.batch_size,
                                               num_workers=1,
                                               pin_memory=True,
                                               )

    # Init model
    model = get_model(args).to(device)
    if args.verbose:
        print(
            f"Using model {type(model)}."
        )
    # Init optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.lr / args.dp_clip, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    if args.verbose:
        print(
            f"Using optimizer {type(optimizer)} wit learning rate {args.lr / args.dp_clip}"
        )

    if checkpoint_dir:
        checkpoint = os.path.join(checkpoint_dir, "checkpoint")
        model_state, optimizer_state = torch.load(checkpoint)
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
        if args.verbose:
            print(
                f"Load checkpoint (model_state, optimizer_state) from file {checkpoint}."
            )

    # Init the privacy engine (or not)
    if not args.disable_dp:
        privacy_engine = PrivacyEngine(secure_mode=args.secure_rng)
        model, optimizer, train_loader = make_private(args, privacy_engine, model, optimizer, train_loader)
        if args.verbose:
            print(
                f"Initialized PrivacyEngine from opacus to ensure ({args.epsilon}, {args.delta})-DP.\n"
                f"Using sigma={optimizer.noise_multiplier} and C={args.dp_clip}"
            )
    else:
        privacy_engine = None
        if args.verbose:
            print(
                f"DP is disabled. No privacy engine is created."
            )

    # Train
    for epoch in range(1, args.epochs + 1):
        loss, top1_acc = train(args, model, device, train_loader, optimizer, privacy_engine, epoch)

    # Save the model as a check point


    # Return the accuracy and models
    sdd = model.state_dict()
    if not args.disable_dp:
        # Privacy Engine will add prefix to the key of the state_dict.
        # Remove the prefix to ensure compatibility.
        sdd = OrderedDict([(key[8:], sdd[key]) for key in sdd.keys()])

    return loss, top1_acc, sdd, optimizer.state_dict()



if __name__ == '__main__':
    args = args_parser()
    '''
    ####################################################################################################################
        If this is the main file, call <main> with "args" as it is.
    ####################################################################################################################    

    ####################################################################################################################
        If using ray.tune for hyper parameter tuning, <main> will be wrapped to produce <main_tune>.

            In <main_tune>, the first input is "config", which contains the hyper parameters to be tuned by ray.tune.
                1.  According to the "config" variable, the corresponding argument in "args" will be changed.
                2.  The procedure <main> will then be called with the altered "args".
                3.  The outputs (loss, accuracy) of <main> will be returned using ray.tune.report.
    ####################################################################################################################            
    '''
    loss, top1_acc, model_state, _ = main(args)


    '''
    ####################################################################################################################
        If this is the main file, call <test_configuration> to test the trained model with "args" as it is.
    ####################################################################################################################
        
        
    ####################################################################################################################
        If using ray.tune for hyper parameter tuning, <test_configuration> will be wrapped to produce <test_best_model>.

            In <test_best_model>, the input is "best_trial", which contains information about the best hyper parameters 
            returned by ray.tune.
                1.  According to the "best_trial.checkpoint.value", the "model_state" will be loaded; "args" will be 
                    altered according to the "best_trial.config".
                2.  The procedure <test_configuration> will be called, with the altered "args" and the loaded 
                    "model_state".
    ####################################################################################################################
    '''
    test_configuration(args, model_state)
