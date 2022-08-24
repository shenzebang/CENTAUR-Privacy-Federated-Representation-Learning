import ray

from main import main
from options import args_parser
from utils.ray_tune_utils import *
from utils.common_utils import set_cuda
from utils.common_utils import check_args

def main_tune(config, args, checkpoint_dir=None):
    '''
        "config" contains the hyper parameters that will be tuned by ray.tune.
    '''
    # Alter "args" according to "best_trial.config"
    update_args_with_config(args, config)

    # TODO: Disable ray backend for now since ray is not compatible with partial participation.
    args.use_ray = False

    main(args, is_ray_tune=True, checkpoint_dir=checkpoint_dir)

# def test_best_model(best_trial, args):
#     '''
#         "config" contains the hyper parameters tuned by ray.tune.
#     '''
#
#     # Alter "args" according to "best_trial.config"
#     update_args_with_config(args, best_trial.config)
#
#     print(
#         f"[ Reporting the best configuration for {args.alg} on {args.dataset} with N={args.num_users} S={args.shard_per_user}]"
#     )
#     print(best_trial.config)
#
#     # Load the corresponding model_state
#     checkpoint_path = os.path.join(best_trial.checkpoint.value, "checkpoint")
#
#     model_state, _ = torch.load(checkpoint_path)
#
#     # Test the performance
#     # test_configuration(args, model_state)

def ray_tune(args, num_samples=1, gpus_per_trial=1):
    search_space = get_search_space_from_args(args)
    # scheduler = ASHAScheduler(
    #     max_t=max_num_epochs,
    #     grace_period=1,
    #     reduction_factor=2)
    scheduler = None
    result = tune.run(
        tune.with_parameters(main_tune, args=args),
        resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
        config=search_space,
        metric="validation_acc",
        mode="max",
        num_samples=num_samples,
        scheduler=scheduler,
        log_to_file=True
    )

    best_trial = result.get_best_trial("validation_acc", "max", "all")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))

    # if ray.util.client.ray.is_connected():
    #     # If using Ray Client, we want to make sure checkpoint access
    #     # happens on the server. So we wrap `test_best_model` in a Ray task.
    #     # We have to make sure it gets executed on the same node that
    #     # ``tune.run`` is called on.
    #     from ray.util.ml_utils.node import force_on_current_node
    #     remote_fn = force_on_current_node(ray.remote(test_best_model))
    #     ray.get(remote_fn.remote(best_trial))
    # else:
    #     test_best_model(best_trial, args)


if __name__ == '__main__':
    args = args_parser()

    n_gpus = set_cuda(args)

    check_args(args)

    if n_gpus > 0:
        ray.init(num_gpus=n_gpus, log_to_driver=False)

        ray_tune(args, num_samples=1, gpus_per_trial=args.gpus_per_trial)

    else:
        print(
            "Please assign at least one GPU! EXIT due to lack of resource!"
        )