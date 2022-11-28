# Torch-Privacy-Federated-Learning

This repository contains the code for one of our coming papers.

---

## Dependence

To install the dependencies: `pip install -r requirements.txt`.

## Data

The CIFAR10, CIFAR100 and MNIST datasets are downloaded automatically by the `torchvision` package.

## Usage

We provide scripts that has been tested to produce the results stated in our paper.
Please find them under the foler `script/user-level-DP`.

In the following, we explain several important flags.

### Explanation of flags

- Model related

  - `args.model`
- Experiment setting related

  - `args.num_users`: The number of clients, denoted by $N$.
  - `args.shard_per_user`: The maximum number of classes a single client can host, denoted by $S$.
  - `args.dataset`
  - `args.validation_ratio`: The ratio of the training dataset withheld for the purpose of validation.
- Training related

  - `args.lr`: The learning rate of local round on the clients, denoted by $\eta_l$.
  - `args.global_lr`: The learning rate of global round on the server, denoted by $\eta_g$. To better understand its meanining, let $\theta_i^{t+1}$ be the model returned from client $i$ after finishing the local updates on the $t^{th}$ round and let $C_g$ denotes the global update clipping threshold ($C_g<\infty$ in the `user-level-DP` setting and $C_g = \infty$ in the `local-level-DP` setting). $\mathcal{S}^t$ denotes the subset of clients that are active in the $t^{th}$ round. $\sigma$ stands for the noise multiplier (determined by the target DP configuration) and $W^t$ is an element-wise standard Gaussian noise matrix with appropariate size.

    $$
    \theta^{t+1} = \theta^{t} + \eta_g \cdot\left( \frac{1}{|\mathcal{S}^t|} \sum_{i \in \mathcal{S}^t} \mathrm{clip}(\theta_i^{t+1} - \theta^t; C_g) + \frac{\sigma C_g}{|\mathcal{S}^t|} W^t\right)

    $$

    A special case is `args.global_lr` set to 1, in which case the server simply averages the models returned from the clients after local updates. Note that $\mathrm{clip}(\cdot; C)$ denotes the clipping operator with parameter $C$.
  - `args.batch_size`: The batch size of local round on the clients.
  - `args.local_ep`: The number of epochs in a single localround on the clients.
  - `args.frac_participate`: The fraction of users that will participate per global round, i.e. $\frac{|\mathcal{S}^t|}{N}$.
  - `args.epochs`: The number of global epochs that **a single client will participate**. The total number of global epochs is hence `args.epochs`/`args.frac_participate`.
- Privacy related

  - `args.dp_type`: Either `user-level-DP` or `local-level-DP`.
  - `args.epsilon`: DP parameter $\epsilon$.
  - `args.delta`: DP parameter $\delta$.
  - `args.dp_clip`: The clipping threshold, i.e. the value $C_g$ in the `user-level-DP` setting or the value $C_l$ in the `local-level-DP` setting.
  - `args.MAX_PHYSICAL_BATCH_SIZE`: The per-sample gradient computation required in the `local-level-DP` setting is memory consuming and hence  `args.batch_size` can not be set too large due the CUDA memory limitation. [Opacus](https://github.com/pytorch/opacus) provides a variant of the dataloader class that supports simulating the large-sized (logical) batch with a number of small-sized (physical) batch. Hence, this wrap class allows a large (logical) batch size to be used even when the physical CUDA memory is limited. The parameter `args.MAX_PHYSICAL_BATCH_SIZE` controls the physical batch size in this wrap class. **This wrap class is used only in the `local-level-DP` setting.**

### Parallel computing with multiple GPUs

Currently, we use [ray](https://github.com/ray-project/ray) to parallel the computations of client update.

- The overall switch is `args.use_ray`. Without sending this flag, `ray` is disabled and the client updates will be conducted sequentially. Note that when there is no CUDA device available, `ray` will also be automatically disabled.
- The flag `args.ray_gpu_fraction` controls the number of ray workers a single GPU can host. For example when there are 4 gpus available and `args.ray_gpu_fraction` is set to 0.3, then there will be in total 12 ray workers ($floor(1/0.3) = 3$, and $3 * 4 = 12$).
- **Caveat:** Right now, `ray` is not compatible with the partial participation setting, i.e. `args.frac_participate` is less than 1. Hence, when `args.frac_participate` is less than 1, `args.use_ray` will be automatically set to false to disable the ray backend.


