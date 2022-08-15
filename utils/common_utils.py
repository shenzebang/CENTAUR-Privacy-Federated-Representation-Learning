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

def make_private(args, privacy_engine: PrivacyEngine, model: nn.Module, optimizer: Optimizer, dataloader: DataLoader):
    if privacy_engine is None: # do nothing if the privacy engine is void
        return model, optimizer, dataloader

    noise_multiplier = get_noise_multiplier_from_args(args, dataloader)
    model, optimizer, dataloader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=dataloader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=args.dp_clip,
    )
    # model, optimizer, dataloader = privacy_engine.make_private_with_epsilon(module=model,
    #             optimizer=optimizer,
    #             data_loader=dataloader,
    #             max_grad_norm=args.dp_clip,
    #             target_epsilon=args.epsilon,
    #             target_delta=args.delta,
    #             epochs=args.epochs
    #         )
    return model, optimizer, dataloader

def get_noise_multiplier_from_args(args, dataloader):
    sample_rate = dataloader.batch_size / len(dataloader.dataset)
    noise_multiplier = get_noise_multiplier(
        target_epsilon=args.epsilon,
        target_delta=args.delta,
        sample_rate=sample_rate,
        epochs=args.epochs * args.local_ep
    )
    return noise_multiplier

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
    if not args.disable_dp:
        # Privacy Engine will add prefix to the key of the state_dict.
        # Remove the prefix to ensure compatibility.
        sd = OrderedDict([(key[8:], sd[key]) for key in sd.keys()])

    return sd

def set_cuda(args):
    '''
        Use "args.gpu" to determine the gpu setting.

    '''
    os.environ['CUDA_LAUNCH_BLOCKING']='1'

    #### TODO: THIS IS NOT WORKING! ####

    # gpus = args.gpu.split('-')
    # negative_gpu = False
    # for gpu in gpus:
    #     if int(gpu) < 0:
    #         negative_gpu = True
    #
    # if negative_gpu:
    #     print(
    #         f"Using CPU only"
    #     )
    #     os.environ["CUDA_VISIBLE_DEVICES"] = ""
    #     args.n_gpus = 0
    #     return 0
    #
    # n_gpus = len(gpus)
    # gpus = ",".join(gpus)
    # print(
    #     f"Using GPUs {gpus}"
    # )
    # os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    # args.n_gpus = n_gpus
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

    if args.use_ray and args.frac_participate < 1:
        print(
            "[ WARNING: The is a bug in the partial participating case with ray backend  ]",
            "[ Automatically set args.use_ray to False. ]"
        )
        args.use_ray = False

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