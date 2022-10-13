import numpy as np
from matplotlib import pyplot



methods = ["DP_FedRep", "DP_FedAvg_ft", "PMTL", "PPSGD"]
means = {
    "DP_FedAvg_ft": [57.25, 75.60, 84.69, 86.85, 87.09, 87.14],
    "DP_FedRep":    [65.64, 77.40, 91.80, 92.79, 92.98, 92.96],
    "PMTL":         [57.40, 75.69, 84.76, 88.81, 88.65, 88.12],
    "PPSGD":        [57.35, 75.59, 89.03, 88.96, 88.99, 88.97]
}

stds = {
    "DP_FedAvg_ft": [1.15, 0.35, 0.92, 1.11, 1.37, 0.63],
    "DP_FedRep":    [1.09, 1.23, 0.24, 0.25, 0.20, 0.24],
    "PMTL":         [1.07, 0.21, 0.94, 2.08, 0.34, 0.59],
    "PPSGD":        [1.26, 1.13, 1.46, 1.93, 2.13, 2.16]
}

names = {
    "DP_FedAvg_ft": "DP-FedAvg-ft",
    "DP_FedRep":    "CENTAUR",
    "PMTL":         "PMTL-ft",
    "PPSGD":        "PPSGD"
}


epsilons = np.array([.125, .25, .5, 1, 2, 4])



pyplot.figure()

colors = ['r', 'g', 'b', 'c']
for method, color in zip(methods, colors):
    mean_method = np.array(means[method])
    std_method  = np.array(stds[method])
    pyplot.errorbar(epsilons, mean_method, std_method, label=names[method])

mean_local = np.ones_like(epsilons) * 90.67
std_local = np.ones_like(epsilons) * 0.46
pyplot.errorbar(epsilons, mean_local, std_local, label="Local")


pyplot.ylabel("utility")
pyplot.xlabel("privacy")
pyplot.legend()
pyplot.show()