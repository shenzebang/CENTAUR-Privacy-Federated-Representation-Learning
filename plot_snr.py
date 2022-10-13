import numpy as np
from matplotlib import pyplot


def plot_snr(n_steps, snrs):
    pyplot.figure()
    colors = ['r', 'g', 'b', 'c']
    for snr, color in zip(snrs, colors):
        pyplot.plot(np.arange(n_steps), snr, color)
    pyplot.ylabel("signal to noise ratio")
    pyplot.xlabel("steps")
    pyplot.show()


methods = ["DP_FedAvg_ft", "DP_FedRep", "PMTL", "PPSGD"]
datasets = ["emnist_d"]


for dataset in datasets:
    snrs = []
    for method in methods:
        snrs.append(np.load(f"log/snrs/snr_{method}_{dataset}.np"))
    n_steps = snrs[0].shape[0]
    plot_snr(n_steps, snrs)

