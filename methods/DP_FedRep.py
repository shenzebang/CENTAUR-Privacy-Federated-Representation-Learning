import copy
import random
import warnings

from opacus.utils.batch_memory_manager import wrap_data_loader
from torch import optim

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
        model, optimizer, train_loader = make_private(self.args, self.PE, self.model, optimizer, self.train_dataloader)


        losses = []
        top1_acc = []

        if self.PE is not None:
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
        self.model.load_state_dict(fix_DP_model_keys(self.args, model))

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
        self.train_dataloader.dataset.disable_multiplicity()
        for head_epoch in range(self.args.local_head_ep):
            for _batch_idx, (data, target) in enumerate(self.train_dataloader):
                data, target = flat_multiplicty_data(data.to(self.device), target.to(self.device))
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                losses.append(loss.item())

                preds = np.argmax(output.detach().cpu().numpy(), axis=1)
                labels = target.detach().cpu().numpy()
                acc = accuracy(preds, labels)
                top1_acc.append(acc)
        self.train_dataloader.dataset.enable_multiplicity()
        return torch.tensor(np.mean(losses)), torch.tensor(np.mean(top1_acc))


    def step(self):
        # 1. Fine tune the head
        _, _ = self._train_head()

        # 2. Calculate the performance of the representation from the previous iteration
        #    The performance is the
        test_loss, test_acc = self.test(self.model)

        # 3. Update the representation
        train_loss, train_acc = self._train_representation()

        # return the accuracy and the updated representation
        result_dict = {
            "train loss":   train_loss,
            "train acc":    train_acc,
            "test loss":    test_loss,
            "test acc":     test_acc,
            "sd":           self.model.state_dict(),
            "PE":           self.PE
        }

        if self.args.verbose:
            print(
                f"Client {self.idx} finished."
            )
        return result_dict

class ServerDPFedRep(Server):
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

    def aggregate(self, sds_client: List[OrderedDict]):
        '''
            Only the simplest average aggregation is implemented
        '''
        sd = self.model.state_dict()
        for key in sd.keys():
            # Only update the representation
            if key in self.representation_keys:
                sd[key] = torch.mean(torch.stack([sd_client[key] for sd_client in sds_client], dim=0), dim=0)
        self.model.load_state_dict(sd)

    def step(self, epoch: int):
        '''
            A single server step consists of 1/args.frac_participate sub-steps
        '''
        sub_step_users = self.divide_into_subgroups()
        results_mega_step = Results()
        for clients in sub_step_users:
            # 1. Server broadcast the global model
            self.broadcast(clients)
            # 2. Server orchestrates the clients to perform local updates
            results_dict_sub_step = self.local_update(clients)
            # This step is to ensure the compatibility with the ray backend.
            if self.args.use_ray:
                for client, sd, PE in zip(clients, results_dict_sub_step["sds"], results_dict_sub_step["PEs"]):
                    client.model.load_state_dict(sd)
                    client.PE = PE
            # 3. Server aggregate the local updates
            self.aggregate(results_dict_sub_step["sds"])
            results_mega_step.add(results_dict_sub_step)

        return self.report(epoch, results_mega_step)

