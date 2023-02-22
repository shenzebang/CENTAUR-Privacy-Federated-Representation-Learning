import copy
import warnings

from methods.api import Server, Client, Results
from utils.common_utils import *
from utils.ray_remote_worker import *



class ClientPPSGD(Client):

    def step(self, epoch: int):
        # 1. Fine tune the head
        # _, _ = self._train_head()
        model_old = self.model
        model_new = copy.deepcopy(self.model)
        _, _ = self._fine_tune_over_head(model_new, self.local_keys)

        # 2. Calculate the performance of the representation from the previous iteration
        statistics = {}
        statistics_validation_testing = self.test(model_new)
        statistics.update(statistics_validation_testing)

        # 3. Update the representation
        # train_loss, train_acc = self._train_representation() if epoch >=0 else (torch.tensor(0.), torch.tensor(0.))
        statistics_training = self._train_over_keys(model_new, self.global_keys) \
                                    if epoch >= 0 else (np.zeros([]), np.zeros([]))
        statistics.update(statistics_training)

        # return the accuracy, the updated head, and the representation difference
        return self.report(model_old, model_new, statistics)


class ServerPPSGD(Server):

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
                for client, sd_local, PE in zip(clients, results_dict_sub_step["sd_local"], results_dict_sub_step["PE"]):
                    sd = client.model.state_dict()
                    for key in sd_local.keys():
                        sd[key] = sd_local[key]
                    client.model.load_state_dict(sd)
                    if client.idx == 0: client.PE = PE
            # 3. Server aggregate the local updates
            self.aggregate(results_dict_sub_step["sd_global_diff"])
            results_mega_step.add(results_dict_sub_step)

            if self.accountant is not None:
                self.accountant.step(
                    noise_multiplier=self.noise_multiplier, sample_rate=self.args.frac_participate
                )

        return self.report(epoch, results_mega_step)

