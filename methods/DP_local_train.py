import copy
import warnings

from opacus.utils.batch_memory_manager import wrap_data_loader
from torch import optim

from utils.common_utils import *
from utils.ray_remote_worker import *

warnings.filterwarnings("ignore")

class Client:
    def __init__(self,
                 idx: int,
                 args,
                 representation_keys: List[str],
                 train_dataloader: DataLoader,
                 test_dataloader: DataLoader,
                 model: nn.Module,
                 device: torch.device
                 ):
        self.idx = idx
        self.args = args
        self.model = copy.deepcopy(model)
        self.representation_keys = representation_keys
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.device = device
        self.criterion = nn.CrossEntropyLoss()


    def _train(self, PE: PrivacyEngine):
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
        model, optimizer, train_loader = make_private(self.args, PE, self.model, optimizer, self.train_dataloader)


        losses = []
        top1_acc = []

        if PE is not None:
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

    def test(self, model_test: nn.Module):
        model_test.eval()

        with torch.autograd.no_grad():
            losses = []
            top1_acc = []

            for _batch_idx, (data, target) in enumerate(self.test_dataloader):
                data, target = data.to(self.device), target.to(self.device)
                output = model_test(data)
                loss = self.criterion(output, target)
                losses.append(loss.item())

                preds = np.argmax(output.detach().cpu().numpy(), axis=1)
                labels = target.detach().cpu().numpy()
                acc = accuracy(preds, labels)
                top1_acc.append(acc)

        return torch.tensor(np.mean(losses)), torch.tensor(np.mean(top1_acc))


    def step(self, PE: PrivacyEngine):
        train_loss, train_acc = self._train(PE)

        test_loss, test_acc = self.test(self.model)

        # return the accuracy and the updated representation
        result = {
            "train loss":   train_loss,
            "train acc":    train_acc,
            "test loss":    test_loss,
            "test acc":     test_acc,
            "sd":           self.model.state_dict(),
            "PE":           PE
        }
        if self.args.verbose:
            print(
                f"Client {self.idx} finished."
            )
        return result

class Server:
    def __init__(self, args, model: nn.Module, representation_keys: List[str], clients: List[Client], remote_workers):
        self.args = args
        self.model = model
        self.representation_keys = representation_keys
        self.clients = clients
        self.PEs = [None] * args.num_users
        if not args.disable_dp:
            self.PEs = [PrivacyEngine(secure_mode=args.secure_rng) for _ in range(args.num_users)]

        self.remote_workers = remote_workers

        self.check_args()

    def broadcast(self):
        '''
            Local training only. No broadcast step.
        '''
        pass

    def aggregate(self, sds_client: List[OrderedDict]):
        '''
            Local training only. No aggregation step.
        '''
        pass

    def local_update(self):
        '''
            Server orchestrates the clients to perform local updates.
            The current implementation did not use ray backend.
        '''

        results = compute_with_remote_workers(self.remote_workers, self.clients, self.PEs)

        result = {
            "train loss":   torch.mean(torch.stack([result["train loss"] for result in results])),
            "train acc":    torch.mean(torch.stack([result["train acc"] for result in results])),
            "test loss":    torch.mean(torch.stack([result["test loss"] for result in results])),
            "test acc":     torch.mean(torch.stack([result["test acc"] for result in results])),
            "sds":          [result["sd"] for result in results],
            "PEs":          [result["PE"] for result in results]
        }
        return result

    def step(self, epoch: int):
        # Server orchestrates the clients to perform local updates
        results = self.local_update()

        del results["sds"]
        torch.cuda.empty_cache()

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


    def report(self, epoch, results):
        train_loss = results["train loss"]
        train_acc = results["train acc"]
        test_loss = results["test loss"]
        test_acc = results["test acc"]
        if not self.args.disable_dp:
            epsilon, best_alpha = self.PEs[0].accountant.get_privacy_spent(
                delta=self.args.delta
            )
            print(
                f"Train Epoch: {epoch} \t"
                f"Loss: {train_loss:.6f} "
                f"Acc@1: {train_acc * 100:.6f} "
                f"(ε = {epsilon:.2f}, δ = {self.args.delta}) for α = {best_alpha}"
            )
            print(
                f"Train Epoch: {epoch} \t"
                f"Test loss: {test_loss:.6f} "
                f"Test acc@1: {test_acc * 100:.6f} "
                f"(ε = {epsilon:.2f}, δ = {self.args.delta}) for α = {best_alpha}"
            )
        else:
            print(f"Train Epoch: {epoch} \t Loss: {train_loss:.6f}"
                  f"\t Acc@1: {train_acc * 100:.6f} "
                  )
            print(f"Test Epoch: {epoch} \t Loss: {test_loss:.6f}"
                  f"\t Acc@1: {test_acc * 100:.6f} "
                  )
        return results["train loss"], results["train acc"], results["test loss"], results["test acc"]