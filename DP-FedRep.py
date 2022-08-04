import copy
import warnings

import ray
from opacus.utils.batch_memory_manager import BatchMemoryManager
from torch import optim

from Models.models import get_model
from common_utils import *
from data_utils import prepare_dataloaders
from options import args_parser

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
        self.model = model
        self.representation_keys = representation_keys
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.device = device
        self.criterion = nn.CrossEntropyLoss()


    def _train_representation(self, PE: PrivacyEngine):
        '''
            The privacy engine is maintained by the server to ensure the compatibility with ray backend
        '''

        activate_in_keys(self.model, self.representation_keys)

        self.model.train()
        optimizer = optim.SGD(self.model.parameters(),
                              lr=args.lr_rep,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay
                              )
        model, optimizer, train_loader = make_private(args, PE, self.model, optimizer, self.train_dataloader)


        losses = []
        top1_acc = []

        if PE is not None:
            with BatchMemoryManager(
                    data_loader=train_loader,
                    max_physical_batch_size=args.MAX_PHYSICAL_BATCH_SIZE,
                    optimizer=optimizer
            ) as memory_safe_data_loader:
                for rep_epoch in range(self.args.local_rep_ep):
                    for _batch_idx, (data, target) in enumerate(memory_safe_data_loader):
                        data, target = data.to(self.device), target.to(self.device)
                        output = model(data)
                        loss = self.criterion(output, target)
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                        model.zero_grad()
                        losses.append(loss.item())

                        preds = np.argmax(output.detach().cpu().numpy(), axis=1)
                        labels = target.detach().cpu().numpy()
                        acc = accuracy(preds, labels)
                        top1_acc.append(acc)
        else:
            for rep_epoch in range(self.args.local_rep_ep):
                for _batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(self.device), target.to(self.device)
                    output = model(data)
                    loss = self.criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    losses.append(loss.item())

                    preds = np.argmax(output.detach().cpu().numpy(), axis=1)
                    labels = target.detach().cpu().numpy()
                    acc = accuracy(preds, labels)
                    top1_acc.append(acc)
        # del optimizer

        # Using PE to privitize the model will change the keys of model.state_dict()
        # This subroutine restores the keys to the non-DP model
        self.model.load_state_dict(fix_DP_model_keys(args, model))

        return torch.tensor(np.mean(losses)), torch.tensor(np.mean(top1_acc))

    def _train_head(self):
        '''
            Optimize over the local head
        '''
        deactivate_in_keys(self.model, self.representation_keys)
        self.model.train()

        optimizer = optim.SGD(self.model.parameters(),
                              lr=args.lr_head,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay
                              )

        losses = []
        top1_acc = []
        local_head_ep = max(self.args.local_ep - self.args.local_rep_ep, 1)
        if self.args.local_ep == self.args.local_rep_ep:
            print("Number of local head epochs is ZERO!")
        for head_epoch in range(local_head_ep):
            for _batch_idx, (data, target) in enumerate(self.train_dataloader):
                data, target = data.to(self.device), target.to(self.device)
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

        return torch.tensor(np.mean(losses)), torch.tensor(np.mean(top1_acc))

    def test(self):
        self.model.eval()

        with torch.autograd.no_grad():
            losses = []
            top1_acc = []

            for _batch_idx, (data, target) in enumerate(self.test_dataloader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                losses.append(loss.item())

                preds = np.argmax(output.detach().cpu().numpy(), axis=1)
                labels = target.detach().cpu().numpy()
                acc = accuracy(preds, labels)
                top1_acc.append(acc)

        return torch.tensor(np.mean(losses)), torch.tensor(np.mean(top1_acc))


    def step(self, PE: PrivacyEngine):
        # 1. Fine tune the head
        _, _ = self._train_head()

        # 2. Calculate the performance of the representation from the previous iteration
        #    The performance is the
        test_loss, test_acc = self.test()

        # 3. Update the representation
        train_loss, train_acc = self._train_representation(PE)

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
    def __init__(self, args, model: nn.Module, representation_keys: List[str], clients: List[Client]):
        self.args = args
        self.model = model
        self.representation_keys = representation_keys
        self.clients = clients
        self.PEs = [None] * args.num_users
        if not args.disable_dp:
            self.PEs = [PrivacyEngine(secure_mode=args.secure_rng) for _ in range(args.num_users)]


    def broadcast(self):
        for client in self.clients:
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

    def local_update(self):
        '''
            Server orchestrates the clients to perform local updates.
            The current implementation did not use ray backend.
        '''
        results = [client.step(PE) for client, PE in zip(self.clients, self.PEs)]

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
        # 1. Server broadcast the global model
        self.broadcast()
        # 2. Server orchestrates the clients to perform local updates
        results = self.local_update()
        # This step is to ensure the compatibility with the ray backend.
        for client, sd in zip(self.clients, results["sds"]):
            client.model.load_state_dict(sd)
        self.PEs = results["PEs"]
        # 3. Server aggregate the local updates
        self.aggregate(results["sds"])


        train_loss = results["train loss"]
        train_acc = results["train acc"]
        test_loss = results["test loss"]
        test_acc = results["test acc"]
        if not args.disable_dp:
            epsilon, best_alpha = self.PEs[0].accountant.get_privacy_spent(
                delta=args.delta
            )
            print(
                f"Train Epoch: {epoch} \t"
                f"Loss: {train_loss:.6f} "
                f"Acc@1: {train_acc * 100:.6f} "
                f"(ε = {epsilon:.2f}, δ = {args.delta}) for α = {best_alpha}"
            )
            print(
                f"Train Epoch: {epoch} \t"
                f"Test loss: {test_loss:.6f} "
                f"Test acc@1: {test_acc * 100:.6f} "
                f"(ε = {epsilon:.2f}, δ = {args.delta}) for α = {best_alpha}"
            )
        else:
            print(f"Train Epoch: {epoch} \t Loss: {train_loss:.6f}"
                  f"\t Acc@1: {train_acc* 100:.6f} "
                  )
            print(f"Test Epoch: {epoch} \t Loss: {test_loss:.6f}"
                  f"\t Acc@1: {test_acc * 100:.6f} "
                  )

        # return results
        return results["train loss"], results["train acc"], results["test loss"], results["test acc"]


def main(args):
    if args.seed != 0:
        seed_all(args.seed)
        if args.verbose:
            print(
                f"Seed is set to {args.seed} to ensure reproducibility"
            )
    else:
        if args.verbose:
            print(
                f"No seed is manually set."
            )

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # Init Dataloaders
    train_dataloaders, test_dataloaders, _ = prepare_dataloaders(args)


    # Init model
    global_model = get_model(args).to(device)
    local_models = [copy.deepcopy(global_model).to(device) for _ in range(args.num_users)]

    # get the representation keys
    representation_keys = get_representation_keys(args, global_model)

    # Init Clients
    clients = [Client(idx, args, representation_keys, traindlr, testdlr, model, device) for idx, (model, traindlr, testdlr) in
               enumerate(zip(local_models, train_dataloaders, test_dataloaders))]

    # Init Server
    server = Server(args, global_model, representation_keys, clients)


    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    # Run experiment
    for epoch in range(args.epochs):
        train_loss, train_acc, test_loss, test_acc = server.step(epoch)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

    # return results
    return train_losses, train_accs, server.model.state_dict()




if __name__ == '__main__':
    args = args_parser()
    '''
    ####################################################################################################################
        If this is the main file, call <main> with "args" as it is.
    ####################################################################################################################    

    ####################################################################################################################
        If using ray.tune for hyper parameter tuning, <main> will be wrapped to produce <main_tune>.

            In <main_tune>, the first input is "config", which contains the hyper parameters to be tuned by ray.tune.
                1.  According to the "config" variable, the corresponding argument in "args" will be changed.
                2.  The procedure <main> will then be called with the altered "args".
                3.  The outputs (loss, accuracy) of <main> will be returned using ray.tune.report.
    ####################################################################################################################            
    '''
    loss, top1_acc, model_state = main(args)

    # '''
    # ####################################################################################################################
    #     If this is the main file, call <test_configuration> to test the trained model with "args" as it is.
    # ####################################################################################################################
    #
    #
    # ####################################################################################################################
    #     If using ray.tune for hyper parameter tuning, <test_configuration> will be wrapped to produce <test_best_model>.
    #
    #         In <test_best_model>, the input is "best_trial", which contains information about the best hyper parameters
    #         returned by ray.tune.
    #             1.  According to the "best_trial.checkpoint.value", the "model_state" will be loaded; "args" will be
    #                 altered according to the "best_trial.config".
    #             2.  The procedure <test_configuration> will be called, with the altered "args" and the loaded
    #                 "model_state".
    # ####################################################################################################################
    # '''
    # test_configuration(args, model_state)