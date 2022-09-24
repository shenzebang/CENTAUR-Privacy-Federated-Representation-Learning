from methods.api import Server, Client, Results
from utils.common_utils import *
from utils.ray_remote_worker import *



class ClientLocalOnly(Client):

    def step(self, step: int):
        # train_loss, train_acc = self._train() if step >= 0 else (torch.tensor(0.), torch.tensor(0.))
        test_freq_local = 10 if self.idx % 50 == 0 else 0
        train_loss, train_acc = self._train_over_keys(self.model, self.local_keys, test_freq_local=test_freq_local) \
                                if step >= 0 else (torch.tensor(0.), torch.tensor(0.))

        validation_loss, validation_acc, test_loss, test_acc = self.test(self.model)

        return self.report(self.model, self.model, train_loss, train_acc, validation_loss, validation_acc, test_loss, test_acc, loss_acc_only=True)


class ServerLocalOnly(Server):

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
