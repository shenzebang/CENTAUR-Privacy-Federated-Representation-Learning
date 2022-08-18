# Torch-Privacy-Federated-Learning
This repository contains the code for one of our coming papers.

---
## Dependence

To install the other dependencies: `pip install -r requirements.txt`.

## Data

The CIFAR10, CIFAR100 and MNIST datasets are downloaded automatically by the `torchvision` package. 

## Usage 

We provide scripts that has been tested to produce the results stated in our paper (TO BE DONE!).
Please see them under the foler `script`.

In the following, we explain several important flags.
### Explanation of flags

- Model related
  - `args.model`
- Experiment setting related
  - `args.num_users`
  - `args.shard_per_user`
  - `args.dataaset`
  - `args.validation_ratio`

- Training related
  - `args.lr`
  - `args.global_lr`
  - `args.batch_size`
  - `args.local_ep`
  - `args.frac_participate`
  - `args.epochs`

- Privacy related

  - `args.dp_type`

  - `args.epsilon`
  - `args.delta`
  - `args.dp_clip`
  - `args.MAX_PHYSICAL_BATCH_SIZE`

### Parallel computing with multiple GPUs

Currently, we use [ray](https://github.com/ray-project/ray) to parallel the computations of client update. 

- The overall switch is `args.use_ray`. Without sending this flag, `ray` is disabled and the client updates will be conducted sequentially. Note that when there is no CUDA device available, `ray` will also be automatically disabled. 
- The flag `args.ray_gpu_fraction` controls the number of ray workers a single GPU can host. For example when there are 4 gpus available and `args.ray_gpu_fraction` is set to 0.3, then there will be in total 12 ray workers (floor(1/0.3) = 3, and 3 * 4 = 12).
- **Caveat:** Right now, `ray` is not compatible with the partial participation setting, i.e. `args.frac_participate` is less than 1. Hence, when `args.frac_participate` is less than 1, `args.use_ray` will be automatically set to false to disable the ray backend.

## Summary of results
We summarize the experiment results as follows. 

In the following table, the number of users N is fixed as 100. S (short for shard) stands for the maximum number classes a client can hold.
For CIFAR10, the parameter $\delta$ of DP is fixed as $10^{-5}$. 

| Datasets | CIFAR10 (S=2) |
| ----------- | ----------- |
| DP-FedRep | 0 |
| DP-FedAvg-ft | 0 |
