import numpy as np
import torch
import random
from opacus import PrivacyEngine, GradSampleModule
from opacus.accountants.utils import get_noise_multiplier
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import Optimizer
from typing import List
from collections import OrderedDict
import itertools
import os

def accuracy(preds: np.ndarray, labels: np.ndarray):
    return (preds == labels).mean()

def seed_all(seed:int = 10):
    if not seed:
        seed = 10

    print("[ Using Seed : ", seed, " ]")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def make_private(args, privacy_engine: PrivacyEngine, model: nn.Module, optimizer: Optimizer, dataloader: DataLoader, noise_multiplier: float = -1):
    if privacy_engine is None: # do nothing if the privacy engine is void
        return model, optimizer, dataloader

    if noise_multiplier < 0:
        noise_multiplier = get_noise_multiplier(
            target_epsilon=args.epsilon,
            target_delta=args.delta,
            sample_rate=dataloader.batch_size / len(dataloader.dataset),
            epochs=args.epochs * args.local_ep
        )

    model, optimizer, dataloader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=dataloader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=args.dp_clip,
    )
    return model, optimizer, dataloader

def activate_in_keys(model: nn.Module, representation_keys: List[str]):
    for name, param in model.named_parameters():
        if name in representation_keys:
            param.requires_grad = True
        else:
            param.requires_grad = False

def deactivate_in_keys(model: nn.Module, representation_keys: List[str]):
    for name, param in model.named_parameters():
        if name not in representation_keys:
            param.requires_grad = True
        else:
            param.requires_grad = False

def get_keys(args, global_model):
    if args.alg == 'PPSGD':
        if 'cifar' in args.dataset:
            if args.model == 'cnn':
                global_keys = [global_model.weight_keys[i] for i in [0, 1, 2, 4, 5]]
                representation_keys = [global_model.weight_keys[i] for i in [0, 1, 4, 5]]
                representation_keys = list(itertools.chain.from_iterable(representation_keys))
                global_keys = list(itertools.chain.from_iterable(global_keys))
                all_keys = list(itertools.chain.from_iterable(global_model.weight_keys))
                local_keys = [key for key in all_keys if key not in global_keys]
            else:
                raise NotImplementedError
        elif 'mnist' in args.dataset:
            if args.model == 'mlp':
                global_keys = [global_model.weight_keys[i] for i in [0, 1, 2, 3]]
                representation_keys = [global_model.weight_keys[i] for i in [0, 1, 2]]
                representation_keys = list(itertools.chain.from_iterable(representation_keys))
                global_keys = list(itertools.chain.from_iterable(global_keys))
                all_keys = list(itertools.chain.from_iterable(global_model.weight_keys))
                local_keys = [key for key in all_keys if key not in global_keys]
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        # there is no fine_tune_key for PPSGD
        return global_keys, local_keys, [], representation_keys

    if args.alg == 'DP_FedRep':
        if 'cifar' in args.dataset:
            if args.model == 'cnn':
                global_keys = [global_model.weight_keys[i] for i in [0, 1, 3, 4]]
                global_keys = list(itertools.chain.from_iterable(global_keys))
                all_keys = list(itertools.chain.from_iterable(global_model.weight_keys))
                local_keys = [key for key in all_keys if key not in global_keys]
                representation_keys = [global_model.weight_keys[i] for i in [0, 1, 3, 4]]
                representation_keys = list(itertools.chain.from_iterable(representation_keys))
            else:
                raise NotImplementedError
        elif 'mnist' in args.dataset:
            if args.model == 'mlp':
                global_keys = [global_model.weight_keys[i] for i in [0, 1, 2]]
                global_keys = list(itertools.chain.from_iterable(global_keys))
                all_keys = list(itertools.chain.from_iterable(global_model.weight_keys))
                local_keys = [key for key in all_keys if key not in global_keys]
                representation_keys = [global_model.weight_keys[i] for i in [0, 1, 2]]
                representation_keys = list(itertools.chain.from_iterable(representation_keys))
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        # there is no fine_tune_key for DP_FedRep
        return global_keys, local_keys, [], representation_keys


    if args.alg == 'DP_FedAvg_ft' or args.alg == 'PMTL':
        if 'cifar' in args.dataset:
            if args.model == 'cnn':
                representation_keys = [global_model.weight_keys[i] for i in [0, 1, 3, 4]]
                representation_keys = list(itertools.chain.from_iterable(representation_keys))
                all_keys = list(itertools.chain.from_iterable(global_model.weight_keys))
                head_keys = [key for key in all_keys if key not in representation_keys]
            else:
                raise NotImplementedError
        elif 'mnist' in args.dataset:
            if args.model == 'mlp':
                representation_keys = [global_model.weight_keys[i] for i in [0, 1, 2]]
                representation_keys = list(itertools.chain.from_iterable(representation_keys))
                all_keys = list(itertools.chain.from_iterable(global_model.weight_keys))
                head_keys = [key for key in all_keys if key not in representation_keys]
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        # there is no local_keys for DP_FedAvg_ft
        return all_keys, [], head_keys, representation_keys

    if args.alg == 'Local':
        if 'cifar' in args.dataset:
            if args.model == 'cnn':
                all_keys = list(itertools.chain.from_iterable(global_model.weight_keys))
                representation_keys = [global_model.weight_keys[i] for i in [0, 1, 3, 4]]
                representation_keys = list(itertools.chain.from_iterable(representation_keys))
            else:
                raise NotImplementedError
        elif 'mnist' in args.dataset:
            if args.model == 'mlp':
                all_keys = list(itertools.chain.from_iterable(global_model.weight_keys))
                representation_keys = [global_model.weight_keys[i] for i in [0, 1, 2]]
                representation_keys = list(itertools.chain.from_iterable(representation_keys))
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        # there is no global_keys or fine_tune_keys for local training
        return [], all_keys, [], representation_keys


    # global_keys = []
    # local_keys = []
    # if 'cifar' in args.dataset:
    #     if args.model == 'cnn':
    #         global_keys = [global_model.weight_keys[i] for i in [0, 1, 3, 4]]
    #     if args.model == 'mlp':
    #         global_keys = [global_model.weight_keys[i] for i in [0, 1, 2]]
    #
    #     global_keys = list(itertools.chain.from_iterable(global_keys))
    #     all_keys = list(itertools.chain.from_iterable(global_model.weight_keys))
    #     local_keys = [key for key in all_keys if key not in global_keys]
    # elif 'mnist' in args.dataset:
    #     if args.model == 'cnn':
    #         global_keys = [global_model.weight_keys[i] for i in [0, 1, 3, 4]]
    #     if args.model == 'mlp':
    #         global_keys = [global_model.weight_keys[i] for i in [0, 1, 2]]
    #
    #     global_keys = list(itertools.chain.from_iterable(global_keys))
    #     all_keys = list(itertools.chain.from_iterable(global_model.weight_keys))
    #     local_keys = [key for key in all_keys if key not in global_keys]
    #
    # elif 'sent140' in args.dataset:
    #     if args.model == 'lstm':
    #         global_keys = global_model[:-4]
    #     if args.model == 'mlp':
    #         global_keys = [global_model[i] for i in [0, 1, 2, 3]]
    #     raise NotImplementedError
    # elif 'harass' in args.dataset:
    #     global_keys = global_model[:-2]
    #     raise NotImplementedError
    # else:
    #     global_keys = global_model[:-2]
    #     raise NotImplementedError



    # return global_keys, local_keys


def fix_DP_model_keys(args, model: nn.Module):
    sd = model.state_dict()
    if not args.disable_dp and args.dp_type == "local-level-DP":
        # Privacy Engine will add prefix to the key of the state_dict.
        # Remove the prefix to ensure compatibility.
        sd = OrderedDict([(key[8:], sd[key]) for key in sd.keys()])

    return sd

def set_cuda(args):
    '''
        Use "args.gpu" to determine the gpu setting.

    '''
    os.environ['CUDA_LAUNCH_BLOCKING']='1'

    args.n_gpus = torch.cuda.device_count()
    print(
        f"[ There are {args.n_gpus} GPUs available. ]"
    )
    return args.n_gpus


class CudaMemoryPrinter:
    def __init__(self):
        self.idx = 0

    def print(self):
        current_memory, total_memory = torch.cuda.mem_get_info(0)
        print(
            f"{self.idx}. {(total_memory - current_memory)/1024**2}"
        )
        self.idx += 1


def aggregate_grad_sample(model, n_multiplicity: int):
    if n_multiplicity > 1 and type(model) is GradSampleModule: # If type(model) is nn.Module, do nothing.
    # A single image produces multiple samples with data augmentation. Average gradients from a single image.
        component_modules = model._module._modules
        for cm_key in component_modules.keys():
            params = component_modules[cm_key]._parameters
            if len(params) != 0:
            # This is a trainable module. If len is 0, this is not trainable, .e.g. pooling layer or dropout layer
                for f_key in params.keys():
                    # TODO: make sure there is no BUG.
                    if params[f_key].grad is not None and hasattr(params[f_key], "grad_sample"):
                    # requires_grad is True! Else, this parameter is not to be updated
                        grad_sample = params[f_key].grad_sample
                        grad_sample = grad_sample.view(-1, n_multiplicity, *grad_sample.shape[1:])
                        params[f_key].grad_sample = torch.mean(grad_sample, dim=1) # 1 is the multiplicity dimension.

def flat_multiplicty_data(data: torch.Tensor, target: torch.Tensor):
    if len(data.shape) == 5:
        data = data.view(data.shape[0] * data.shape[1], *data.shape[2:])
        target = target.view(target.shape[0] * target.shape[1], *target.shape[2:])
        return data, target
    elif len(data.shape) == 4 or len(data.shape) == 3:
        return data, target
    else:
        raise ValueError(
            "data.shape should be 5 (data augmentation with multiplicity), 4 (batch), or 3 (single)!"
        )

def check_args(args):
    '''
        Check the args to prevent undesired settings
    '''
    if args.data_augmentation is False and args.data_augmentation_multiplicity > 1:
        print(
            "[ WARNING: No data augmentation is performed, but data augmentation multiplicity is set larger than 1! ]",
            "[ Automatically set the multiplicity to 0. ]"
        )
        args.data_augmentation_multiplicity = 0

    # if args.use_ray and args.frac_participate < 1:
    #     print(
    #         "[ WARNING: The is a bug in the partial participating case with ray backend  ]",
    #         "[ Automatically set args.use_ray to False. ]"
    #     )
    #     args.use_ray = False

def restore_from_checkpoint(args, global_model, checkpoint_dir=None):
    if checkpoint_dir is not None:
        print(
            "Unless starting from a pretrained model, "
            "when training from scratch, "
            "loading checkpoint may not make much sense since the privacy accountant is not loaded."
        )
        checkpoint = os.path.join(checkpoint_dir, "checkpoint")
        model_state = torch.load(checkpoint)
        global_model.load_state_dict(model_state)
        if args.verbose:
            print(
                f"Load checkpoint (model_state) from file {checkpoint}."
            )


def trimmed_mean(x, dim, cut_percentage=.1):
    n = x.shape[dim]
    n_cut = int(n * cut_percentage)
    x = torch.topk(x, n - n_cut, dim=dim, largest=True).values
    x = torch.topk(x, n - 2 * n_cut, dim=dim, largest=False).values
    return torch.mean(x, dim=dim)

AGGR_OPS = {
    "mean"  : torch.mean,
    "median": lambda x, dim: torch.median(x, dim).values,
    "trimmed_mean": trimmed_mean
}


def server_update_with_clip(sd: OrderedDict, sds_global_diff: List[OrderedDict], keys: List[str], representation_keys: List[str],
                            clip_threshold=-1, global_lr=1, noise_level=0, aggr='mean', print_diff_norm=False):
    '''
        Only the key in "keys" will be updated. If "keys" is empty, all keys will be updated.
    '''
    # if cut_percentage > .49:
    #     raise ValueError("The cut percentage is over 49\%!")

    if len(keys) == 0: keys = sd.keys()

    n_clients = len(sds_global_diff)

    with torch.autograd.no_grad():
        if clip_threshold <= 0: # The server performs no clip.
            aggr_op = AGGR_OPS[aggr]
            for key in keys:
                sds_global_diff_key = [sd_global_diff[key] for sd_global_diff in sds_global_diff]
                sd[key] = sd[key] + global_lr * aggr_op(torch.stack(sds_global_diff_key, dim=0), dim=0)
            snr = -1
        else: # The server performs clip.
            norm_diff_clients = [ torch.ones(1) ] * n_clients
            # 1. Calculate the norm of differences
            for cid, sd_global_diff in enumerate(sds_global_diff):
                norm_diff_cid_square = [torch.norm(sd_global_diff[key]) ** 2 for key in keys]
                norm_diff_clients[cid] = torch.sqrt(torch.sum(torch.stack(norm_diff_cid_square)))

            if print_diff_norm:
                norm_diff_std, norm_diff_mean = torch.std_mean(torch.stack(norm_diff_clients))
                print(f"[Norm diff mean: {norm_diff_mean: .5f}, norm diff std: {norm_diff_std: .5f}]")
            else:
                norm_diff_std, norm_diff_mean = torch.zeros([]), torch.zeros([])
            # 2. Rescale the diffs
            rescale_clients = [1 if norm_diff_client<clip_threshold else clip_threshold/norm_diff_client
                                 for norm_diff_client in norm_diff_clients]
            for rescale_client, sd_global_diff in zip(rescale_clients, sds_global_diff):
                for key in keys:
                    sd_global_diff[key] = sd_global_diff[key] * rescale_client

            # 3. update the global model
            for key in keys:
                white_noise = noise_level * torch.randn(sd[key].size(), device=sd[key].device) if noise_level > 0 else 0
                sds_global_diff_key = [sd_global_diff[key] for sd_global_diff in sds_global_diff]
                sd[key] = sd[key] + global_lr * (torch.mean(torch.stack(sds_global_diff_key, dim=0), dim=0) + white_noise / n_clients)

            # 4. compute the signal noise ratio
            total_numel = 0
            signal = torch.zeros(1)
            for key in representation_keys:
                sds_global_diff_key = torch.stack([sd_global_diff[key] for sd_global_diff in sds_global_diff])
                total_numel += torch.numel(sds_global_diff_key)
                signal = signal + torch.sum(torch.abs(sds_global_diff_key))

            signal_per_dim = signal / total_numel

            snr = signal_per_dim / noise_level * n_clients

    return sd, snr.numpy(), norm_diff_mean[None].numpy(), norm_diff_std[None].numpy()

class Results:
    def __init__(self):
        self.train_losses = []
        self.train_accs = []
        self.validation_losses = []
        self.validation_accs = []
        self.test_losses = []
        self.test_accs = []

    def add(self, result):
        train_loss = result["train loss"]
        train_acc = result["train acc"]
        validation_loss = result["validation loss"]
        validation_acc = result["validation acc"]
        test_loss = result["test loss"]
        test_acc = result["test acc"]
        self.train_losses.append(train_loss)
        self.train_accs.append(train_acc)
        self.validation_losses.append(validation_loss)
        self.validation_accs.append(validation_acc)
        self.test_losses.append(test_loss)
        self.test_accs.append(test_acc)

    def mean(self):
        return (torch.mean(torch.stack(self.train_losses)),
                torch.mean(torch.stack(self.train_accs)),
                torch.mean(torch.stack(self.validation_losses)),
                torch.mean(torch.stack(self.validation_accs)),
                torch.mean(torch.stack(self.test_losses)),
                torch.mean(torch.stack(self.test_accs)))

class Logger:
    def __init__(self):
        self.train_losses_history = []
        self.train_accs_history = []
        self.validation_losses_history = []
        self.validation_accs_history = []
        self.test_losses_history = []
        self.test_accs_history = []

        self.train_losses_current = []
        self.train_accs_current = []
        self.validation_losses_current = []
        self.validation_accs_current = []
        self.test_losses_current = []
        self.test_accs_current = []
        
        self.current_epoch = 0

        self.snrs = []
        self.gradient_norm_means = []
        self.gradient_norm_stds = []

    def log(self, stats_dict_all, epoch):
        if epoch == self.current_epoch:
            self.train_losses_current.append(stats_dict_all["train loss"])
            self.train_accs_current.append(stats_dict_all["train acc"])
            self.validation_losses_current.append(stats_dict_all["validation loss"])
            self.validation_accs_current.append(stats_dict_all["validation acc"])
            self.test_losses_current.append(stats_dict_all["test loss"])
            self.test_accs_current.append(stats_dict_all["test acc"])
        else:
            if epoch != self.current_epoch + 1:
                raise ValueError("The stats should be logger sequentially!")

            # This is a new epoch
            self.current_epoch = epoch
            # Store the stats of the current epoch in hisotry
            self.train_losses_history.append(torch.cat(self.train_losses_current, dim=0))
            self.train_accs_history.append(torch.cat(self.train_accs_current, dim=0))
            self.validation_losses_history.append(torch.cat(self.validation_losses_current, dim=0))
            self.validation_accs_history.append(torch.cat(self.validation_accs_current, dim=0))
            self.test_losses_history.append(torch.cat(self.test_losses_current, dim=0))
            self.test_accs_history.append(torch.cat(self.test_accs_current, dim=0))

            # Clear the current stats
            self._reset()

            # Store the input stats
            self.log(stats_dict_all, epoch)
            
    def log_snr(self, snr: np.ndarray):
        self.snrs.append(snr)


    def log_gradient_norm(self, norm_mean: np.ndarray, norm_std: np.ndarray):
        self.gradient_norm_means.append(norm_mean)
        self.gradient_norm_stds.append(norm_std)


    def _reset(self):
        self.train_losses_current = []
        self.train_accs_current = []
        self.validation_losses_current = []
        self.validation_accs_current = []
        self.test_losses_current = []
        self.test_accs_current = []
        
    def report(self, epoch):
        if len(self.train_losses_current) != 0:
            self.train_losses_history.append(torch.cat(self.train_losses_current, dim=0))
            self.train_accs_history.append(torch.cat(self.train_accs_current, dim=0))
            self.validation_losses_history.append(torch.cat(self.validation_losses_current, dim=0))
            self.validation_accs_history.append(torch.cat(self.validation_accs_current, dim=0))
            self.test_losses_history.append(torch.cat(self.test_losses_current, dim=0))
            self.test_accs_history.append(torch.cat(self.test_accs_current, dim=0))

            self._reset()

        return self.train_losses_history[epoch], self.train_accs_history[epoch], self.validation_losses_history[epoch],\
                    self.validation_accs_history[epoch], self.test_losses_history[epoch], self.test_accs_history[epoch]

    def report_snr(self):
        return np.concatenate(self.snrs)

    def report_gradient_norm(self):
        return np.concatenate(self.gradient_norm_means), np.concatenate(self.gradient_norm_stds)

    def save_snr(self, save_directory, save_name):
        snrs = self.report_snr()
        file_name = save_directory + save_name
        with open(file_name, 'wb') as f:
            np.save(f, snrs)

    def save_gradient_norm(self, save_directory, save_name):
        g_mean, g_std = self.report_gradient_norm()
        file_name = save_directory + save_name
        with open(file_name, 'wb') as f:
            np.save(f, np.stack([g_mean, g_std]))