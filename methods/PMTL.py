import copy
import warnings
import gc

import torch
from opacus.utils.batch_memory_manager import wrap_data_loader
from torch import optim

from utils.data_utils import prepare_ft_dataloader

from utils.common_utils import *
from utils.ray_remote_worker import *
from methods.api import Server, Client, Results


class ClientPMTL(Client):

    def step(self, epoch: int):
        model_head = copy.deepcopy(self.model)
        _, _ = self._fine_tune_over_head(model_head, self.fine_tune_keys)

        validation_loss, validation_acc, test_loss, test_acc = self.test(model_head)

        del model_head

        model_old = self.model
        model_new = copy.deepcopy(model_old)
        def regularization(model: nn.Module):
            sd = model.state_dict()
            sd_old = model_old.state_dict()
            reg = torch.sum(torch.stack([torch.norm(sd[key] - sd_old[key])**2 for key in self.global_keys]))

            return reg * self.args.PMTL_lambda


        train_loss, train_acc = self._train_over_keys(model_new, self.global_keys, regularization=regularization) \
                                if epoch >= 0 else (torch.tensor(0.), torch.tensor(0.))

        # return the accuracy and the model difference
        return self.report(model_old, model_new, train_loss, train_acc, validation_loss, validation_acc, test_loss,
                           test_acc)

class ServerPMTL(Server):
    def _get_local_and_global_keys(self):
        return [], self.fine_tune_keys + self.representation_keys

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
            for client, PE in zip(clients, results_dict_sub_step["PEs"]):
                if client.idx == 0: client.PE = PE
            # 3. Server aggregate the local updates
            self.aggregate(results_dict_sub_step["sds_global_diff"])
            results_mega_step.add(results_dict_sub_step)

            if self.accountant is not None:
                self.accountant.step(
                    noise_multiplier=self.noise_multiplier, sample_rate=self.args.frac_participate
                )


        return self.report(epoch, results_mega_step)
