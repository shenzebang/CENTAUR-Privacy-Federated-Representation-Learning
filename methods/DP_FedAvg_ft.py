import copy
import warnings
import gc
from opacus.utils.batch_memory_manager import wrap_data_loader
from torch import optim

from utils.data_utils import prepare_ft_dataloader

from utils.common_utils import *
from utils.ray_remote_worker import *
from methods.api import Server, Client, Results
warnings.filterwarnings("ignore")

class ClientDPFedAvgFT(Client):

    def _train(self):
        '''
            The privacy engine is maintained by the server to ensure the compatibility with ray backend
        '''
        # deactivate over an empty set == activate all the variables
        deactivate_in_keys(self.model, [])

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

        # Using PE to privatize the model will change the keys of model.state_dict()
        # This subroutine restores the keys to the non-DP model
        self.model.load_state_dict(fix_DP_model_keys(self.args, model))

        return torch.tensor(np.mean(losses)), torch.tensor(np.mean(top1_acc))

    def _fine_tune_head(self, model_head: nn.Module):
        '''
            Optimize over the local head of a copy of the model.
            The result is not incorporated into "self.model"
        '''
        deactivate_in_keys(model_head, self.representation_keys)
        model_head.train()

        optimizer = optim.SGD(model_head.parameters(),
                              lr=self.args.lr_head,
                              momentum=self.args.momentum,
                              weight_decay=self.args.weight_decay
                              )

        ft_dataloader = prepare_ft_dataloader(self.args, self.device, self.model, self.train_dataloader.dataset.d_split)
        losses = []
        top1_acc = []
        for head_epoch in range(self.args.ft_ep):
            for _batch_idx, (data, target) in enumerate(ft_dataloader):
                data, target = data.to(self.device), target.to(self.device)
                output = model_head(data, head=True)
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
        return [], self.fine_tune_keys + self.representation_keys

    def step(self, epoch: int):
        # 1. Fine tune the head of a copy
        model_head = copy.deepcopy(self.model)
        # _, _ = self._fine_tune_head(model_head)
        _, _ = self._fine_tune_over_head(model_head, self.fine_tune_keys)

        # 2. Calculate the performance of the representation from the previous iteration
        #    Only the fine tuned model is tested
        validation_loss, validation_acc, test_loss, test_acc = self.test(model_head)

        del model_head

        # 3. Update the representation
        # train_loss, train_acc = self._train() if epoch >= 0 else (torch.tensor(0.), torch.tensor(0.))
        model_old = self.model
        model_new = copy.deepcopy(model_old)
        train_loss, train_acc = self._train_over_keys(model_new, self.fine_tune_keys+self.representation_keys) \
                                if epoch >= 0 else (torch.tensor(0.), torch.tensor(0.))

        # return the accuracy and the model difference
        return self.report(model_old, model_new, train_loss, train_acc, validation_loss, validation_acc, test_loss,
                           test_acc)

class ServerDPFedAvgFT(Server):
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
