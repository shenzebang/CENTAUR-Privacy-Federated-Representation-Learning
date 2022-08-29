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

def get_representation_keys(args, global_model):
    representation_keys = []
    if 'cifar' in args.dataset:
        if args.model == 'cnn':
            representation_keys = [global_model.weight_keys[i] for i in [0, 1, 3, 4]]
        if args.model == 'mlp':
            representation_keys = [global_model.weight_keys[i] for i in [0, 1, 2]]
    elif 'mnist' in args.dataset:
        if args.model == 'cnn':
            representation_keys = [global_model.weight_keys[i] for i in [0, 1, 3, 4]]
        if args.model == 'mlp':
            representation_keys = [global_model.weight_keys[i] for i in [0, 1, 2]]
    elif 'sent140' in args.dataset:
        if args.model == 'lstm':
            representation_keys = global_model[:-4]
        if args.model == 'mlp':
            representation_keys = [global_model[i] for i in [0, 1, 2, 3]]
    elif 'harass' in args.dataset:
        representation_keys = global_model[:-2]
    else:
        representation_keys = global_model[:-2]


    if 'sent140' not in args.dataset and 'harass' not in args.dataset:
        representation_keys = list(itertools.chain.from_iterable(representation_keys))

    return representation_keys


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


def server_update_with_clip(sd: OrderedDict, sd_clients: List[OrderedDict], keys: List[str],
                            clip_threshold=-1, global_lr=1, noise_level=0, aggr='mean'):
    '''
        Only the key in "keys" will be updated. If "keys" is empty, all keys will be updated.
    '''
    # if cut_percentage > .49:
    #     raise ValueError("The cut percentage is over 49\%!")

    if len(keys) == 0: keys = sd.keys()

    if clip_threshold <= 0: # The server performs no clip.
        aggr_op = AGGR_OPS[aggr]
        for key in keys:
            param_clients_key = [sd_client[key] for sd_client in sd_clients]
            sd[key] = sd[key] * (1 - global_lr) + global_lr * aggr_op(torch.stack(param_clients_key, dim=0), dim=0)
    else: # The server performs clip.
        diff_clients = [ {} for _ in range(len(sd_clients)) ]
        norm_diff_clients = [ torch.ones(1) ] * len(sd_clients)
        # 1. Calculate the norm of differences
        for cid, sd_client in enumerate(sd_clients):
            diff_cid = {key: sd_client[key] - sd[key] for key in keys}
            norm_diff_cid_square = [torch.norm(diff_cid[key]) ** 2 for key in keys]
            norm_diff_clients[cid] = torch.sqrt(torch.sum(torch.stack(norm_diff_cid_square)))
            diff_clients[cid] = diff_cid

        # 2. Rescale the diffs
        rescale_clients = [1 if norm_diff_client<clip_threshold else clip_threshold/norm_diff_client
                             for norm_diff_client in norm_diff_clients]
        for rescale_client, diff_client in zip(rescale_clients, diff_clients):
            for key in keys:
                diff_client[key] = diff_client[key] * rescale_client

        # 3. update the global model
        for key in keys:
            white_noise = noise_level * torch.randn(sd[key].size(), device=sd[key].device) if noise_level > 0 else 0
            # white_noise = 0
            diff_clients_key = [diff_client[key] for diff_client in diff_clients]
            sd[key] = sd[key] + global_lr * (torch.mean(torch.stack(diff_clients_key, dim=0), dim=0) + white_noise / len(sd_clients))

    return sd