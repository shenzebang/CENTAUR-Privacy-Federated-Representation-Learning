from matplotlib import pyplot
import numpy as np
from utils.common_utils import Logger


def plot_per_client_stats(run: int, stats: np.ndarray, stats_name: str, save_dir=None):
    if stats.ndim != 1:
        raise ValueError("The stats to be plotted should be a one dimensional array!")
    n_client = stats.shape[0]
    pyplot.figure()
    pyplot.plot(np.arange(n_client), stats)
    pyplot.ylabel(stats_name)
    pyplot.xlabel("client index")
    pyplot.show()

    if save_dir is not None:
        pyplot.savefig(f'{save_dir}/{stats_name}_vs_clients_run_{run}.pdf')

def plot_snr(run: int, snrs: np.ndarray, save_dir=None):
    if snrs.ndim != 1:
        raise ValueError("The stats to be plotted should be a one dimensional array!")
    n_steps = snrs.shape[0]
    pyplot.figure()
    pyplot.plot(np.arange(n_steps), snrs)
    pyplot.ylabel("signal to noise ratio")
    pyplot.xlabel("steps")
    pyplot.show()

    if save_dir is not None:
        pyplot.savefig(f'{save_dir}/snr_run_{run}.pdf')

def plot_stats_in_logger(run: int, logger: Logger, epoch: int, plot_save_dir=None):
    train_losses, train_accs, validation_losses, validation_accs, test_losses, test_accs = logger.report(epoch)
    plot_per_client_stats(run, train_losses, "training loss", plot_save_dir)
    plot_per_client_stats(run, train_accs, "training accuracy", plot_save_dir)
    plot_per_client_stats(run, validation_losses, "validation loss", plot_save_dir)
    plot_per_client_stats(run, validation_accs, "validation accuracy", plot_save_dir)
    plot_per_client_stats(run, test_losses, "testing loss", plot_save_dir)
    plot_per_client_stats(run, test_accs, "testing accuracy", plot_save_dir)

    # plot_snr(run, logger.report_snr(), plot_save_dir)

