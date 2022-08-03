import numpy as np
import torch

import os

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler

from options import args_parser
from baseline_centralized import main, test_configuration



def main_tune(config, args):
    '''
        "config" contains the hyper parameters that will be tuned by ray.tune.
    '''
    args.lr = config['lr']
    args.dp_clip = config['C']
    args.epochs = config['epochs']
    args.train_batch_size = config['batch size']

    loss, accuracy, model_state, optimizer_state = main(args)

    with tune.checkpoint_dir(step=args.epochs) as checkpoint_dir:
        path = os.path.join(checkpoint_dir, "checkpoint")
        torch.save(
            (model_state, optimizer_state), path)

    tune.report(loss=loss, accuracy=accuracy)

def test_best_model(best_trial, args):
    '''
        "config" contains the hyper parameters tuned by ray.tune.
    '''

    # Alter "args" according to "best_trial.config"
    args.lr = best_trial.config['lr']
    args.dp_clip = best_trial.config['C']
    args.epochs = best_trial.config['epochs']
    args.train_batch_size = best_trial.config['batch size']

    # Load the corresponding model_state
    checkpoint_path = os.path.join(best_trial.checkpoint.value, "checkpoint")

    model_state, _ = torch.load(checkpoint_path)

    # Test the performance
    test_configuration(args, model_state)

def ray_tune(args, num_samples=10, max_num_epochs=10, gpus_per_trial=2):
    config = {
        "lr": tune.grid_search([.05, .1, .15, .2]),#tune.choice([.05, .1, .2]), #tune.sample_from(lambda _: 2 ** np.random.randint(-6, -3)),
        "C": tune.grid_search([.25, .5, 1]),   #tune.sample_from(lambda _: 1),
        "epochs": tune.grid_search([400, 500, 600]), #tune.loguniform(1e-4, 1e-1),
        "batch size": tune.grid_search([4000])
    }
    # scheduler = ASHAScheduler(
    #     max_t=max_num_epochs,
    #     grace_period=1,
    #     reduction_factor=2)
    scheduler = None
    result = tune.run(
        tune.with_parameters(main_tune, args=args),
        resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
        config=config,
        metric="accuracy",
        mode="max",
        num_samples=num_samples,
        scheduler=scheduler
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))

    if ray.util.client.ray.is_connected():
        # If using Ray Client, we want to make sure checkpoint access
        # happens on the server. So we wrap `test_best_model` in a Ray task.
        # We have to make sure it gets executed on the same node that
        # ``tune.run`` is called on.
        from ray.util.ml_utils.node import force_on_current_node
        remote_fn = force_on_current_node(ray.remote(test_best_model))
        ray.get(remote_fn.remote(best_trial))
    else:
        test_best_model(best_trial, args)


if __name__ == '__main__':
    args = args_parser()

    ray_tune(args, num_samples=1, max_num_epochs=2, gpus_per_trial=.3)