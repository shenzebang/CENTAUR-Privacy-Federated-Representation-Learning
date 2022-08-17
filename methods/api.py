import warnings

import torch

from utils.common_utils import *
from utils.ray_remote_worker import *
import copy
warnings.filterwarnings("ignore")

class Results:
    def __init__(self):
        self.train_losses = []
        self.train_accs = []
        self.validation_losses = []
        self.validation_accs = []
        self.test_losses = []
        self.test_accs = []

    def add(self, result):
        train_loss = result["train loss"]
        train_acc = result["train acc"]
        validation_loss = result["validation loss"]
        validation_acc = result["validation acc"]
        test_loss = result["test loss"]
        test_acc = result["test acc"]
        self.train_losses.append(train_loss)
        self.train_accs.append(train_acc)
        self.validation_losses.append(validation_loss)
        self.validation_accs.append(validation_acc)
        self.test_losses.append(test_loss)
        self.test_accs.append(test_acc)

    def mean(self):
        return (torch.mean(torch.stack(self.train_losses)),
                torch.mean(torch.stack(self.train_accs)),
                torch.mean(torch.stack(self.validation_losses)),
                torch.mean(torch.stack(self.validation_accs)),
                torch.mean(torch.stack(self.test_losses)),
                torch.mean(torch.stack(self.test_accs)))

class Client:
    def __init__(self,
                 idx: int,
                 args,
                 representation_keys: List[str],
                 train_dataloader: DataLoader,
                 validation_dataloader: DataLoader,
                 test_dataloader: DataLoader,
                 model: nn.Module,
                 device: torch.device
                 ):
        self.idx = idx
        self.args = args
        self.model = copy.deepcopy(model)
        self.representation_keys = representation_keys
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.test_dataloader = test_dataloader
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.PE = None
        if not args.disable_dp:
            self.PE = PrivacyEngine(secure_mode=args.secure_rng)

    def test(self, model_test: nn.Module):
        model_test.eval()
        with torch.autograd.no_grad():
            validation_losses = []
            validation_top1_acc = []

            for _batch_idx, (data, target) in enumerate(self.validation_dataloader):
                data, target = data.to(self.device), target.to(self.device)
                output = model_test(data)
                loss = self.criterion(output, target)
                validation_losses.append(loss.item())

                preds = np.argmax(output.detach().cpu().numpy(), axis=1)
                labels = target.detach().cpu().numpy()
                acc = accuracy(preds, labels)
                validation_top1_acc.append(acc)


        with torch.autograd.no_grad():
            test_losses = []
            test_top1_acc = []

            for _batch_idx, (data, target) in enumerate(self.test_dataloader):
                data, target = data.to(self.device), target.to(self.device)
                output = model_test(data)
                loss = self.criterion(output, target)
                test_losses.append(loss.item())

                preds = np.argmax(output.detach().cpu().numpy(), axis=1)
                labels = target.detach().cpu().numpy()
                acc = accuracy(preds, labels)
                test_top1_acc.append(acc)

        return torch.tensor(np.mean(validation_losses)), torch.tensor(np.mean(validation_top1_acc)), \
               torch.tensor(np.mean(test_losses)), torch.tensor(np.mean(test_top1_acc))

    def step(self):
        raise NotImplementedError

class Server:
    def __init__(self, args, model: nn.Module, representation_keys: List[str], clients: List[Client], remote_workers: List[Worker]):
        self.args = args
        self.model = model
        self.representation_keys = representation_keys
        self.clients = clients
        self.remote_workers = remote_workers

    def broadcast(self, clients: List[Client]):
        raise NotImplementedError

    def aggregate(self, sds_client: List[OrderedDict]):
        raise NotImplementedError

    def local_update(self, clients: List[Client]):
        '''
            Server orchestrates the clients to perform local updates.
            The current implementation did not use ray backend.
        '''
        results = compute_with_remote_workers(self.remote_workers, clients)

        result_dict = {
            "train loss": torch.mean(torch.stack([result["train loss"] for result in results])),
            "train acc": torch.mean(torch.stack([result["train acc"] for result in results])),
            "validation loss": torch.mean(torch.stack([result["validation loss"] for result in results])),
            "validation acc": torch.mean(torch.stack([result["validation acc"] for result in results])),
            "test loss": torch.mean(torch.stack([result["test loss"] for result in results])),
            "test acc": torch.mean(torch.stack([result["test acc"] for result in results])),
            "sds": [result["sd"] for result in results],
            "PEs": [result["PE"] for result in results]
        }
        return result_dict

    def step(self, epoch: int):
        raise NotImplementedError

    def report(self, epoch, results: Results):
        train_loss, train_acc, validation_loss, validation_acc, test_loss, test_acc = results.mean()
        if epoch % self.args.print_freq == 0 or epoch > self.args.epochs - 5:
            if not self.args.disable_dp:
                epsilon, best_alpha = self.clients[0].PE.accountant.get_privacy_spent(
                    delta=self.args.delta
                )
                print(
                    f"On {self.args.dataset} using {self.args.alg} with {self.args.frac_participate * 100}\% par. rate, "
                    f"Train Epoch: {epoch} \t"
                    f"Loss: {train_loss:.6f} "
                    f"Acc@1: {train_acc * 100:.6f} "
                    f"(ε = {epsilon:.2f}, δ = {self.args.delta}) for α = {best_alpha}"
                )
                print(
                    f"On {self.args.dataset} using {self.args.alg} with {self.args.frac_participate * 100}\% par. rate, "
                    f"Validation Epoch: {epoch} \t"
                    f"Loss: {validation_loss:.6f} "
                    f"Acc@1: {validation_acc * 100:.6f} "
                    f"(ε = {epsilon:.2f}, δ = {self.args.delta}) for α = {best_alpha}"
                )
                print(
                    f"On {self.args.dataset} using {self.args.alg} with {self.args.frac_participate * 100}\% par. rate, "
                    f"Train Epoch: {epoch} \t"
                    f"Test loss: {test_loss:.6f} "
                    f"Test acc@1: {test_acc * 100:.6f} "
                    f"(ε = {epsilon:.2f}, δ = {self.args.delta}) for α = {best_alpha}"
                )
            else:
                print(
                    f"On {self.args.dataset} using {self.args.alg} with {self.args.frac_participate * 100}\% par. rate, "
                    f"Train Epoch: {epoch} \t Loss: {train_loss:.6f}"
                    f"\t Acc@1: {train_acc * 100:.6f} "
                      )
                print(
                    f"On {self.args.dataset} using {self.args.alg} with {self.args.frac_participate * 100}\% par. rate, "
                    f"Validation Epoch: {epoch} \t Loss: {validation_loss:.6f}"
                    f"\t Acc@1: {validation_acc * 100:.6f} "
                )
                print(
                    f"On {self.args.dataset} using {self.args.alg} with {self.args.frac_participate * 100}\% par. rate, "
                    f"Test Epoch: {epoch} \t Loss: {test_loss:.6f}"
                    f"\t Acc@1: {test_acc * 100:.6f} "
                      )
        return train_loss, train_acc, validation_loss, validation_acc, test_loss, test_acc

    def divide_into_subgroups(self):
        if self.args.frac_participate < 1:
            # i. Shuffle the clients
            random.shuffle(self.clients)

            # ii. split the clients into subgroups
            num_sub_steps = int(1 / self.args.frac_participate)

            user_per_sub_step = [int(self.args.num_users / num_sub_steps)] * num_sub_steps
            for i in range(self.args.num_users % num_sub_steps):
                user_per_sub_step[i] += 1

            sub_step_users = []; p = 0
            for sub_step, num_users in enumerate(user_per_sub_step):
                sub_step_users.append(self.clients[p: p+num_users])
                p += num_users

            return sub_step_users

        else:
            return [self.clients]

