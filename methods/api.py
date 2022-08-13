import warnings

from utils.common_utils import *
from utils.ray_remote_worker import *
import copy
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
        self.PE = None
        if not args.disable_dp:
            self.PE = PrivacyEngine(secure_mode=args.secure_rng)

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

    def step(self):
        raise NotImplementedError

class Server:
    def __init__(self, args, model: nn.Module, representation_keys: List[str], clients: List[Client], remote_workers: List[Worker]):
        self.args = args
        self.model = model
        self.representation_keys = representation_keys
        self.clients = clients
        self.remote_workers = remote_workers

    def broadcast(self):
        raise NotImplementedError

    def aggregate(self, sds_client: List[OrderedDict]):
        raise NotImplementedError

    def local_update(self, clients: List[Client]):
        '''
            Server orchestrates the clients to perform local updates.
            The current implementation did not use ray backend.
        '''
        results = compute_with_remote_workers(self.remote_workers, clients)

        result = {
            "train loss": torch.mean(torch.stack([result["train loss"] for result in results])),
            "train acc": torch.mean(torch.stack([result["train acc"] for result in results])),
            "test loss": torch.mean(torch.stack([result["test loss"] for result in results])),
            "test acc": torch.mean(torch.stack([result["test acc"] for result in results])),
            "sds": [result["sd"] for result in results],
            "PEs": [result["PE"] for result in results]
        }
        return result

    def step(self, epoch: int):
        raise NotImplementedError

    def report(self, epoch, results):
        train_loss = results["train loss"]
        train_acc = results["train acc"]
        test_loss = results["test loss"]
        test_acc = results["test acc"]
        if not self.args.disable_dp:
            epsilon, best_alpha = self.clients[0].PE.accountant.get_privacy_spent(
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

