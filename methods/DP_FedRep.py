import copy
import gc
import warnings

import torch
from opacus.utils.batch_memory_manager import wrap_data_loader
from torch import optim

from utils.data_utils import prepare_ft_dataloader
from utils.common_utils import *
from utils.ray_remote_worker import *
from methods.api import Server, Client, Results


class ClientDPFedRep(Client):

    def step(self, epoch: int):
        # 1. Fine tune the head
        # _, _ = self._train_head()
        model_old = self.model
        model_new = copy.deepcopy(self.model)
        _, _ = self._fine_tune_over_head(model_new, self.local_keys)

        # 2. Calculate the performance of the representation from the previous iteration
        #    The performance is the
        validation_loss, validation_acc, test_loss, test_acc = self.test(model_new)

        # 3. Update the representation
        # train_loss, train_acc = self._train_representation() if epoch >=0 else (torch.tensor(0.), torch.tensor(0.))
        train_loss, train_acc = self._train_over_keys(model_new, self.global_keys) \
                                    if epoch >= 0 else (torch.tensor(0.), torch.tensor(0.))

        # return the accuracy, the updated head, and the representation difference
        return self.report(model_old, model_new, train_loss, train_acc, validation_loss, validation_acc, test_loss,
                           test_acc)


class ServerDPFedRep(Server):

    def step(self, epoch: int):
        '''
            A single server step consists of 1/args.frac_participate sub-steps
        '''
        sub_step_users = self.divide_into_subgroups()
        results_mega_step = Results()
        for clients in sub_step_users:

            gc.collect()
            torch.cuda.empty_cache()

            # 1. Server broadcast the global model
            self.broadcast(clients)
            # 2. Server orchestrates the clients to perform local updates
            results_dict_sub_step = self.local_update(clients, epoch)
            # This step is to ensure the compatibility with the ray backend.
            if self.args.use_ray:
                for client, sd_local, PE in zip(clients, results_dict_sub_step["sds_local"], results_dict_sub_step["PEs"]):
                    sd = client.model.state_dict()
                    for key in sd_local.keys():
                        sd[key] = sd_local[key]
                    client.model.load_state_dict(sd)
                    if client.idx == 0: client.PE = PE

            head_norms = []
            for client in clients:
                head_norm = 0
                sd_client = client.model.state_dict()
                for key in client.local_keys:
                    head_norm += torch.norm(sd_client[key]) ** 2
                head_norms.append(torch.sqrt(head_norm))
            head_std, head_mean = torch.std_mean(torch.stack(head_norms))
            print(f"head norm is {head_mean}({head_std})")

            # 3. Server aggregate the local updates
            self.aggregate(results_dict_sub_step["sds_global_diff"])
            results_mega_step.add(results_dict_sub_step)

            if self.accountant is not None:
                self.accountant.step(
                    noise_multiplier=self.noise_multiplier, sample_rate=self.args.frac_participate
                )

        return self.report(epoch, results_mega_step)

