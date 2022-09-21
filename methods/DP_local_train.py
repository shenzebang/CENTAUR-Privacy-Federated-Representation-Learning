import warnings

from opacus.utils.batch_memory_manager import wrap_data_loader
from torch import optim

from methods.api import Server, Client, Results
from utils.common_utils import *
from utils.ray_remote_worker import *

warnings.filterwarnings("ignore")

class ClientLocalOnly(Client):
    def _get_local_and_global_keys(self):
        return [], self.fine_tune_keys + self.representation_keys

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


    def step(self, step: int):
        # train_loss, train_acc = self._train() if step >= 0 else (torch.tensor(0.), torch.tensor(0.))
        train_loss, train_acc = self._train_over_keys(self.model, self.fine_tune_keys + self.representation_keys) \
                                if step >= 0 else (torch.tensor(0.), torch.tensor(0.))

        validation_loss, validation_acc, test_loss, test_acc = self.test(self.model)

        return self.report(self.model, self.model, train_loss, train_acc, validation_loss, validation_acc, test_loss, test_acc, loss_acc_only=True)


class ServerLocalOnly(Server):

    def _get_local_and_global_keys(self):
        return [], self.fine_tune_keys + self.representation_keys

    def broadcast(self, clients: List[Client]):
        '''
            Local training only. No broadcast step.
        '''
        pass

    def aggregate(self, sds_client: List[OrderedDict]):
        '''
            Local training only. No aggregation step.
        '''
        pass

    def step(self, epoch: int):
        # Server orchestrates the clients to perform local updates
        results = Results()
        results.add(self.local_update(self.clients, epoch))

        # return results
        return self.report(epoch, results)

    def check_args(self):
        '''
            Make sure the configurations in "self.args" matches the purpose of local training
        '''
        if self.args.epochs != 1:
            print(
                f"This is a local training method with {self.args.num_users} users."
                "The number of global epochs is automatically set to 1."
            )
            self.args.epochs = 1
