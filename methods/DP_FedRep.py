import copy
import gc
import warnings

from opacus.utils.batch_memory_manager import wrap_data_loader
from torch import optim

from utils.data_utils import prepare_ft_dataloader
from utils.common_utils import *
from utils.ray_remote_worker import *
from methods.api import Server, Client, Results
warnings.filterwarnings("ignore")

class ClientDPFedRep(Client):

    def _train_representation(self):
        '''
            The privacy engine is maintained by the server to ensure the compatibility with ray backend
        '''

        activate_in_keys(self.model, self.representation_keys)

        self.model.train()
        optimizer = optim.SGD(self.model.parameters(),
                              lr=self.args.lr,
                              momentum=self.args.momentum,
                              weight_decay=self.args.weight_decay
                              )
        model, optimizer, train_loader = make_private(self.args, self.PE, self.model, optimizer, self.train_dataloader, self.noise_multiplier)


        losses = []
        top1_acc = []

        if self.PE is not None and self.train_dataloader.batch_size > self.args.MAX_PHYSICAL_BATCH_SIZE:
            train_loader = wrap_data_loader(
                    data_loader=train_loader,
                    max_batch_size=self.args.MAX_PHYSICAL_BATCH_SIZE,
                    optimizer=optimizer
            )

        for rep_epoch in range(self.args.local_ep):
            for _batch_idx, (data, target) in enumerate(train_loader):
                data, target = flat_multiplicty_data(data.to(self.device), target.to(self.device))
                output = model(data)
                loss = self.criterion(output, target)
                loss.backward()
                aggregate_grad_sample(model, self.args.data_augmentation_multiplicity)
                optimizer.step()
                optimizer.zero_grad()
                model.zero_grad()
                losses.append(loss.item())

                preds = np.argmax(output.detach().cpu().numpy(), axis=1)
                labels = target.detach().cpu().numpy()
                acc = accuracy(preds, labels)
                top1_acc.append(acc)
        # del optimizer

        # Using PE to privitize the model will change the keys of model.state_dict()
        # This subroutine restores the keys to the non-DP model
        # self.model.load_state_dict(fix_DP_model_keys(self.args, model))

        return torch.tensor(np.mean(losses)), torch.tensor(np.mean(top1_acc))

    def _train_head(self):
        '''
            Optimize over the local head
        '''
        deactivate_in_keys(self.model, self.representation_keys)
        self.model.train()

        optimizer = optim.SGD(self.model.parameters(),
                              lr=self.args.lr_head,
                              momentum=self.args.momentum,
                              weight_decay=self.args.weight_decay
                              )

        losses = []
        top1_acc = []
        ft_dataloader = prepare_ft_dataloader(self.args, self.device, self.model, self.train_dataloader.dataset.d_split)
        for head_epoch in range(self.args.local_head_ep):
            for _batch_idx, (data, target) in enumerate(ft_dataloader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data, head=True)
                loss = self.criterion(output, target)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                losses.append(loss.item())

                preds = np.argmax(output.detach().cpu().numpy(), axis=1)
                labels = target.detach().cpu().numpy()
                acc = accuracy(preds, labels)
                top1_acc.append(acc)

        del ft_dataloader

        return torch.tensor(np.mean(losses)), torch.tensor(np.mean(top1_acc))

    def _get_local_and_global_keys(self):
        return self.fine_tune_keys, self.representation_keys

    def step(self, epoch: int):
        # 1. Fine tune the head
        # _, _ = self._train_head()
        model_old = self.model
        model_new = copy.deepcopy(self.model)
        _, _ = self._fine_tune_over_head(model_new, self.fine_tune_keys)

        # 2. Calculate the performance of the representation from the previous iteration
        #    The performance is the
        validation_loss, validation_acc, test_loss, test_acc = self.test(model_new)

        # 3. Update the representation
        # train_loss, train_acc = self._train_representation() if epoch >=0 else (torch.tensor(0.), torch.tensor(0.))
        train_loss, train_acc = self._train_over_keys(model_new, self.representation_keys) \
                                    if epoch >= 0 else (torch.tensor(0.), torch.tensor(0.))

        # return the accuracy, the updated head, and the representation difference
        return self.report(model_old, model_new, train_loss, train_acc, validation_loss, validation_acc, test_loss,
                           test_acc)


class ServerDPFedRep(Server):

    def _get_local_and_global_keys(self):
        return self.fine_tune_keys, self.representation_keys

    def broadcast(self, clients):
        for client in clients:
            sd_client_old = client.model.state_dict()
            client.model = copy.deepcopy(self.model)
            sd_client_new = client.model.state_dict()
            for key in sd_client_new.keys():
                # The local head should not be broadcast
                if key not in self.representation_keys:
                    sd_client_new[key] = copy.deepcopy(sd_client_old[key])
            client.model.load_state_dict(sd_client_new)

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
            # 3. Server aggregate the local updates
            self.aggregate(results_dict_sub_step["sds_global_diff"])
            results_mega_step.add(results_dict_sub_step)

            if self.accountant is not None:
                self.accountant.step(
                    noise_multiplier=self.noise_multiplier, sample_rate=self.args.frac_participate
                )

        return self.report(epoch, results_mega_step)

