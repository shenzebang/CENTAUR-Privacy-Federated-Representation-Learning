import numpy as np
from matplotlib import pyplot


lines = [
    ('solid', 'r'),
    ('dashed', 'r')
]

def plot_grad(n_steps, norms, i, description):
    pyplot.figure()
    
    pyplot.plot(np.arange(n_steps), norms[0][0], linestyle = 'solid', linewidth = 1.5, label = 'Global layers')
    pyplot.fill_between(np.arange(n_steps), norms[0][0] + norms[0][1], norms[0][0] - norms[0][1], alpha = 0.5)

    pyplot.plot(np.arange(n_steps), norms[1][0], linestyle = 'solid', linewidth = 1.5, label = 'Representation layers')
    pyplot.fill_between(np.arange(n_steps), norms[1][0] + norms[1][1], norms[1][0] - norms[1][1], alpha = 0.5)
    

    pyplot.ylabel("gradient norm")
    pyplot.xlabel("steps")
    pyplot.title(description)
    pyplot.legend(fontsize=8)
    pyplot.savefig(f"ICLR2023_CR_Large_Epoch/log/gradient_norm/gradient_norm_{method}_{dataset}_R{i}.png")


methods = ["DP_FedAvg_ft"]
datasets = ["cifar10"]
num_runs = 3
setup = "N1000_S2"

for dataset in datasets:
    for i in range(num_runs):
        norms = []
        for method in methods:
            norms.append(np.load(f"ICLR2023_CR_Large_Epoch/log/gradient_norm/{method}_{dataset}_{setup}_R{i}.np"))
            norms.append(np.load(f"ICLR2023_CR_Large_Epoch/log/gradient_norm/{method}_{dataset}_{setup}_R{i}_rep.np"))
        n_steps = norms[0][0].shape[0]
        plot_grad(n_steps, norms, i, description = f"{method} {dataset} {setup} run_{i}")

