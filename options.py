#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # checkpoint arguments
    parser.add_argument('--save_checkpoint', action="store_true")
    parser.add_argument('--save_freq', type=int, default=10, help="frequency of saving checkpoints")

    # federated arguments

    parser.add_argument('--repeat_class', type=bool, default=False, help='whether allow a user to have shards from the same class')

    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")
    parser.add_argument('--grad_norm', action='store_true', help='use_gradnorm_avging')

    parser.add_argument('--local_updates', type=int, default=1000000, help="maximum number of local updates")
    parser.add_argument('--m_tr', type=int, default=500, help="maximum number of samples/user to use for training")
    parser.add_argument('--m_ft', type=int, default=500, help="maximum number of samples/user to use for fine-tuning")


    
    # algorithm-specific hyperparameters
    parser.add_argument('--lr_g', type=float, default=0.1, help="global learning rate for SCAFFOLD")
    parser.add_argument('--mu', type=float, default='0.1', help='FedProx parameter mu')
    parser.add_argument('--gmf', type=float, default='0', help='FedProx parameter gmf')
    parser.add_argument('--alpha_apfl', type=float, default='0.75', help='APFL parameter alpha')
    parser.add_argument('--alpha_l2gd', type=float, default='1', help='L2GD parameter alpha')
    parser.add_argument('--lambda_l2gd', type=float, default='0.5', help='L2GD parameter lambda')
    parser.add_argument('--lr_in', type=float, default='0.0001', help='PerFedAvg inner loop step size')
    parser.add_argument('--bs_frac_in', type=float, default='0.8', help='PerFedAvg fraction of batch used for inner update')
    parser.add_argument('--lam_ditto', type=float, default='1', help='Ditto parameter lambda')
    parser.add_argument('--dim_clip_perc', type=float, default='1.0', help='percentile for per-dimension clipping bound')

    # other arguments

    parser.add_argument('--iid', type=bool, default=False, help='whether i.i.d or not')

    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')

    parser.add_argument('--load_fed', type=str, default='n', help='define pretrained federated model path')
    parser.add_argument('--results_save', type=str, default='runA', help='define fed results save folder')
    parser.add_argument('--save_every', type=int, default=50, help='how often to save models')
    parser.add_argument('--trim_frac', type=float, default=0.2, help='trimmed fraction for trimmed mean')
    parser.add_argument('--dir', type=str, default='default', help='saved results directory')
    parser.add_argument('--model_dir', type=str, default='DoNotSave', help='saved models directory')
    parser.add_argument('--sample_size_var', type=float, default=0, help='positive value for varying sample sizes')

    parser.add_argument(
        "--print_diff_norm",
        action="store_true",
        default=False,
        help="print the mean and std of the local difference norm",
    )

    #########


    # DP configuration
    parser.add_argument('--dp_type', type=str, default="local-level-DP", choices=['local-level-DP', 'user-level-DP'])
    parser.add_argument('--dp_clip', type=float, default=1, help='clipping norm for DP')
    parser.add_argument('--epsilon', type=float, default=-1, help='privacy budget')
    parser.add_argument('--delta', type=float, default=1e-6, help='privacy guarantee probability')
    parser.add_argument('--noise_multiplier', type=float, default=-1,
                        help='If noise_multiplier < 0, set the noise multiplier of the PrivacyEngine automatically'
                             'according to the DP budget, else overwrite the noise multiplier manually.'
                        )
    parser.add_argument(
        "--disable-dp",
        action="store_true",
        default=False,
        help="Disable privacy training and just train with vanilla implementation",
    )
    parser.add_argument(
        "--secure-rng",
        action="store_true",
        default=False,
        help="Enable Secure RNG to have trustworthy privacy guarantees. Comes at a performance cost",
    )

    # Train configuration
    parser.add_argument('--epochs', type=int, default=1, help="rounds of training")
    parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=1, help="the number of local epochs: E")
    parser.add_argument('--frac_participate', type=float, default=1.,
                        help="fraction of clients that participates per sub-step")
    parser.add_argument('--local_epochs', type=int, default=1, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10, help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--global_lr', type=float, default=1., help="update rate on the server")
    parser.add_argument('--momentum', type=float, default=0.9, help="SGD momentum (default: 0.9)")
    parser.add_argument('--lr_decay', type=float, default=0., help="learning rate decay per round")
    parser.add_argument('--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--batch_size', type=int, default=128, help="train batch size")
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--aggr', type=str, default='mean', help="name of dataset", choices=['mean', 'median', 'trimmed_mean'])

                                                                                                 # Test configuration
    parser.add_argument('--test_batch_size', type=int, default=128, help="test batch size")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--print_freq', type=int, default=10)

    # Dataset configuration
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset", choices=['cifar10',
                                                                                                 'cifar100',
                                                                                                 'emnist_l',
                                                                                                 'emnist_d',
                                                                                                 'fashionmnist'])
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges")
    parser.add_argument('--validation_ratio', type=float, default=.1)

    # Experiment configuration
    parser.add_argument('--alg', type=str, default='DP_FedRep', choices=['DP_FedRep', 'DP_FedAvg_ft', 'PPSGD', 'Local', 'PMTL'])
    parser.add_argument('--num_users', type=int, default=100, help="number of users: n")
    parser.add_argument('--shard_per_user', type=int, default=2, help="classes per user")
    parser.add_argument('--shard_size', type=int, default=200, help="size per shard")

    parser.add_argument('--n_runs', type=int, default=5, help="number of independent trials")
    parser.add_argument(
        "--data_augmentation",
        action="store_true",
        default=False,
        help="Enable data augmentation for vision tasks",
    )

    parser.add_argument(
        "--data_augmentation_multiplicity",
        type=int,
        default=16,
    )


    # model configuration
    parser.add_argument('--model', type=str, default='mlp', help='model name', choices=['mlp', 'cnn', 'resnet'])
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--norm', type=str, default='None', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")
    parser.add_argument('--num_layers_keep', type=int, default=1, help='number layers to keep')

    # Backend configuration
    parser.add_argument('--ray_gpu_fraction', type=float, default=.3)
    parser.add_argument('--MAX_PHYSICAL_BATCH_SIZE', type=int, default=400, help="used in batch_memory_manager")
    parser.add_argument('--use_ray', action='store_true')


    # Algorithm-specific configurations
    ### DP-FedRep
    parser.add_argument('--lr-rep', type=float, default=0.1, help="learning rate")
    parser.add_argument('--lr-head', type=float, default=0.01, help="learning rate")
    parser.add_argument('--local_head_ep', type=int, default=1,
                        help="the number of local epochs for the head for FedRep")

    ### DP-FedAvg-ft
    parser.add_argument('--ft-ep', type=int, default=1,
                        help="the number of local epochs for fine tuning the head")

    ### PPSGD
    parser.add_argument('--lr_l', type=float, default=0.1, help="learning rate of the local model")

    ### PMTL
    parser.add_argument('--PMTL_lambda', type=float, default=0.1, help="weight of the regularization")


    ### Ray.tune configuration
    parser.add_argument('--gpus_per_trial', type=float, default=1.,)

    args = parser.parse_args()
    return args
