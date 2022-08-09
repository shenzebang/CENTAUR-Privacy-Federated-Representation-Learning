import warnings

from torchsummary import summary

from utils.common_utils import *
from utils.data_utils import prepare_dataloaders
from methods.DP_FedAvg_ft import Client as DP_FedAvg_ft_Client
from methods.DP_FedAvg_ft import Server as DP_FedAvg_ft_Server
from methods.DP_FedRep import Client as DP_FedRep_Client
from methods.DP_FedRep import Server as DP_FedRep_Server
from methods.PPSGD import Client as PPSGD_Client
from methods.PPSGD import Server as PPSGD_Server
from models.models import get_model
from options import args_parser
from utils.ray_remote_worker import *
from ray import tune


ALGORITHMS = {
    "DP_FedAvg_ft"  : (DP_FedAvg_ft_Server, DP_FedAvg_ft_Client),
    "DP_FedRep"     : (DP_FedRep_Server, DP_FedRep_Client),
    "PPSGD"         : (PPSGD_Server, PPSGD_Client),
}



warnings.filterwarnings("ignore")

def main(args, is_ray_tune = False, checkpoint_dir=None):
    '''
         "checkpoint_dir" will be used to ensure future compatibility with other ray.tune schedulers.
    '''
    if args.seed != 0:
        seed_all(args.seed)
        if args.verbose:
            print(
                f"[ Seed is set to {args.seed} to ensure reproducibility. ]"
            )
    else:
        if args.verbose:
            print(
                f"[ No seed is manually set. ]"
            )

    device = torch.device(f'cuda' if torch.cuda.is_available() and args.gpu != -1 else 'cpu')


    # Determine the algorithm
    print(
        f"[ Running Algorithm {args.alg}. ]"
    )
    (Server, Client) = ALGORITHMS[args.alg]



    # Init Dataloaders
    train_dataloaders, test_dataloaders = prepare_dataloaders(args)


    # Init model
    global_model = get_model(args).to(device)
    summary(global_model, input_size=(3, 32, 32))
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

    # get the representation keys
    representation_keys = get_representation_keys(args, global_model)

    # Init Clients
    clients = [Client(idx, args, representation_keys, traindlr, testdlr, global_model, device) for idx, (traindlr, testdlr) in
               enumerate(zip(train_dataloaders, test_dataloaders))]

    # Init Server
    if args.n_gpus > 0 and args.use_ray:
        RemoteWorker = ray.remote(num_gpus=args.ray_gpu_fraction)(Worker)
        n_remote_workers = int(1 / args.ray_gpu_fraction) * args.n_gpus
        print(
            f"[ Creating {n_remote_workers} remote workers altogether. ]"
        )
        remote_workers = [RemoteWorker.remote(args.n_gpus, wid) for wid in range(n_remote_workers)]
    else:
        print(
            f"[ No remote workers is created. Clients are evaluated sequentially. ]"
        )
        remote_workers = None

    server = Server(args, global_model, representation_keys, clients, remote_workers)


    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    # Run experiment
    for epoch in range(args.epochs):
        train_loss, train_acc, test_loss, test_acc = server.step(epoch)
        if is_ray_tune:
            with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint")
                torch.save(server.model.state_dict(), path)

            tune.report(
                train_loss  = train_loss.item(),
                train_acc   = train_acc.item(),
                test_loss   = test_loss.item(),
                test_acc    = test_acc.item()
            )

        train_losses.append(train_loss.item())
        train_accs.append(train_acc.item())
        test_losses.append(test_loss.item())
        test_accs.append(test_acc.item())

    # return results
    return train_losses, train_accs, server.model.state_dict()




if __name__ == '__main__':
    args = args_parser()

    n_gpus = set_cuda(args)

    if args.use_ray and n_gpus > 0:
        ray.init(num_gpus=n_gpus, log_to_driver=False)
    '''
    ####################################################################################################################
        If this is the main file, call <main> with "args" as it is and "is_ray_tune" is set to False.
    ####################################################################################################################    

    ####################################################################################################################
        If using ray.tune for hyper parameter tuning, <main> will be wrapped to produce <main_tune> and "is_ray_tune" is
        set to True.

            In <main_tune>, the first input is "config", which contains the hyper parameters to be tuned by ray.tune.
                1.  According to the "config" variable, the corresponding argument in "args" will be changed.
                2.  The procedure <main> will then be called with the altered "args".
                3.  The outputs (loss, accuracy) of <main> will be returned using ray.tune.report.
    ####################################################################################################################            
    '''
    loss, top1_acc, model_state = main(args)

    # '''
    # ####################################################################################################################
    #     If this is the main file, call <test_configuration> to test the trained model with "args" as it is.
    # ####################################################################################################################
    #
    #
    # ####################################################################################################################
    #     If using ray.tune for hyper parameter tuning, <test_configuration> will be wrapped to produce <test_best_model>.
    #
    #         In <test_best_model>, the input is "best_trial", which contains information about the best hyper parameters
    #         returned by ray.tune.
    #             1.  According to the "best_trial.checkpoint.value", the "model_state" will be loaded; "args" will be
    #                 altered according to the "best_trial.config".
    #             2.  The procedure <test_configuration> will be called, with the altered "args" and the loaded
    #                 "model_state".
    # ####################################################################################################################
    # '''
    # test_configuration(args, model_state)