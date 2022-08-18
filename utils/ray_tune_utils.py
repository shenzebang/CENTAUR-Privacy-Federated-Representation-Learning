from ray import tune

def get_search_space_from_args(args):
    '''
         For different algorithm, there are different hyper parameters.
         The search space of these hyperparmeters are set here.
    '''
    if args.alg == 'DP_FedRep':
        search_space = {
            "lr": tune.grid_search([.05, .1, .15,]),
            # "lr": tune.grid_search([.1]),
            # "lr_head": tune.grid_search([.01, .02, .04, .08]),
            # tune.choice([.05, .1, .2]), #tune.sample_from(lambda _: 2 ** np.random.randint(-6, -3)),
            "C": tune.grid_search([1]),  # tune.sample_from(lambda _: 1),
            # "C": tune.grid_search([.25, .5, 1]),
            "epochs": tune.grid_search([200, 300, 400, 500]),  # tune.loguniform(1e-4, 1e-1),
            "local_ep": tune.grid_search([1, 2, 4, 8, 16]),  # tune.loguniform(1e-4, 1e-1),
            "batch size": tune.grid_search([4000])
        }
    elif args.alg == 'DP_FedAvg_ft':
        search_space = {
            "lr": tune.grid_search([.05, .1, .15,]),
            # "lr": tune.grid_search([.1]),
            # "lr_head": tune.grid_search([.01, .02, .04, .08]),
            # tune.choice([.05, .1, .2]), #tune.sample_from(lambda _: 2 ** np.random.randint(-6, -3)),
            # "C": tune.grid_search([.25, .5, 1]),  # tune.sample_from(lambda _: 1),
            "C": tune.grid_search([1]),
            "epochs": tune.grid_search([200, 300, 400, 500]),  # tune.loguniform(1e-4, 1e-1),
            "local_ep": tune.grid_search([1, 2, 4, 8, 16]),  # tune.loguniform(1e-4, 1e-1),
            "batch size": tune.grid_search([4000])
        }
    else:
        raise NotImplementedError

    return search_space

def update_args_with_config(args, config):
    '''
         For different algorithm, there are different hyper parameters.
         Given a configuration of the hyper parameters, the fields in "args" is adjusted accordingly.
    '''
    if args.alg == 'DP_FedRep':
        args.lr = config['lr']
        args.dp_clip = config['C']
        args.epochs = config['epochs']
        args.local_ep = config['local_ep']
        args.batch_size = config['batch size']
    elif args.alg == 'DP_FedAvg_ft':
        args.lr = config['lr']
        args.dp_clip = config['C']
        args.epochs = config['epochs']
        args.batch_size = config['batch size']
        args.local_ep = config['local_ep']
    else:
        raise NotImplementedError