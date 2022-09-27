## Local Level DP

In the following tables, the number of users $N$ is fixed as 100. $S$ (short for shard) stands for the maximum number classes a client can hold.
For CIFAR10 and CIFAR100, the parameter $\delta$ of DP is fixed as $10^{-5}$.


| Datasets     | CIFAR10 ($S=2$) | Hyperparameters | Tuning script |
| -------------- | ----------------- | ----------------- | ----------------- |
| DP-FedRep ($\epsilon=1$)    | 81.64%               | Current best trial: 1ad10_00004 with validation_acc=0.809822265625 and parameters={'"lr": 0.05, "C": 1, "epochs": 200, "local_ep": 2, "batch size": 4000}. Result logdir: /home/ubuntu/ray_results/main_tune_2022-08-22_11-18-10          | `script/local-level-DP/CIFAR10/N100_S2/run_tune_DP-FedRep.sh`|
| DP-FedAvg-ft ($\epsilon=1$) | 79.36%              | Current best trial: 2e804_00026 with validation_acc=0.793619140625 and parameters={'lr': 0.1, 'C': 1, 'epochs': 400, 'local_ep': 1, 'batch size': 500}. Result logdir: /home/ubuntu/ray_results/main_tune_2022-09-09_07-44-06          |`script/local-level-DP/CIFAR10/N100_S2/tune/DP-FedAvg.sh`                   |




## User Level DP

|Algorithm|  Local_only | DP-FedRep |  DP-FedAvg | PMTL | PPSGD |
|---------|-------------|-----------|------------|------|-------|
| N500S2  |79.36 (0.49)|78.54 (0.54)|72.12 (0.77)|      |74.46, ...|
| N1000S2 | 
| N500S5  |
| N1000S5 |