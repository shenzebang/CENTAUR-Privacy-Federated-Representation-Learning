import warnings

from opacus.accountants import RDPAccountant
from torch import optim
from opacus.utils.batch_memory_manager import wrap_data_loader
from utils.data_utils import prepare_ft_dataloader

from utils.common_utils import *
from utils.ray_remote_worker import *
import copy

class Client:
    def __init__(self,
                 idx: int,
                 args,
                 global_keys: List[str],
                 local_keys: List[str],
                 fine_tune_keys: List[str],
                 train_dataloader: DataLoader,
                 validation_dataloader: DataLoader,
                 test_dataloader: DataLoader,
                 model: nn.Module,
                 device: torch.device
                 ):
        self.idx = idx
        self.args = args
        self.model = copy.deepcopy(model)
        self.global_keys = global_keys
        self.local_keys = local_keys
        self.fine_tune_keys = fine_tune_keys
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.test_dataloader = test_dataloader
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.PE = None
        self.noise_multiplier = -1
        if not args.disable_dp and args.dp_type == "local-level-DP":
            self.PE = PrivacyEngine(secure_mode=args.secure_rng)
            if self.args.noise_multiplier < 0:
                self.noise_multiplier = get_noise_multiplier(
                target_epsilon=args.epsilon,
                target_delta=args.delta,
                sample_rate=1. / len(train_dataloader),
                epochs=args.epochs * args.local_ep
                )
                if self.idx == 0:
                    print(
                        f"[ To achieve ({args.epsilon}, {args.delta}) local-level DP,"
                        f" the noise multiplier is automatically set to {self.noise_multiplier} ]"
                    )
            else:
                self.noise_multiplier = self.args.noise_multiplier
                if self.idx == 0:
                    print(
                        f"[ local-level DP. The noise multiplier is manually set to {self.noise_multiplier} ]"
                    )
            # self.noise_multiplier = 0

    def _train_over_keys(self, model: nn.Module, keys: List[str], regularization=None, test_freq_local=0):
        activate_in_keys(model, keys)

        model.train()
        optimizer = optim.SGD(model.parameters(),
                              lr=self.args.lr,
                              momentum=self.args.momentum,
                              weight_decay=self.args.weight_decay
                              )
        model, optimizer, train_loader = make_private(self.args, self.PE, model, optimizer, self.train_dataloader,
                                                      self.noise_multiplier)

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
                reg = regularization(model) if regularization is not None else 0
                loss = loss + reg
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
            if test_freq_local > 0 and rep_epoch % test_freq_local == 0:
                validation_loss, validation_acc, test_loss, test_acc = self.test(model)
                print(
                    f"After {rep_epoch} local epochs, on dataset {self.args.dataset}, {self.args.alg} achieves\t"
                    f"Validation Loss: {validation_loss:.2f} Validation Acc@1: {validation_acc * 100:.2f} \t"
                    f"Test loss: {test_loss:.2f} Test acc@1: {test_acc * 100:.2f} "
                )
        # del optimizer

        # Using PE to privitize the model will change the keys of model.state_dict()
        # This subroutine restores the keys to the non-DP model
        # self.model.load_state_dict(fix_DP_model_keys(self.args, model))

        return torch.tensor(np.mean(losses)), torch.tensor(np.mean(top1_acc))

    def _fine_tune_over_head(self, model: nn.Module, keys: List[str]):
        activate_in_keys(model, keys)
        model.train()

        optimizer = optim.SGD(model.parameters(),
                              lr=self.args.lr_head,
                              momentum=self.args.momentum,
                              weight_decay=self.args.weight_decay
                              )

        losses = []
        top1_acc = []
        ft_dataloader = prepare_ft_dataloader(self.args, self.device, model, self.train_dataloader.dataset.d_split)
        for head_epoch in range(self.args.local_head_ep):
            for _batch_idx, (data, target) in enumerate(ft_dataloader):
                data, target = data.to(self.device), target.to(self.device)
                output = model(data, head=True)
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

    def _eval(self, model: nn.Module, dataloader: DataLoader):
        model.eval()
        with torch.autograd.no_grad():
            losses = []
            top1_acc = []

            for _batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                loss = self.criterion(output, target)
                losses.append(loss.item())

                preds = np.argmax(output.detach().cpu().numpy(), axis=1)
                labels = target.detach().cpu().numpy()
                acc = accuracy(preds, labels)
                top1_acc.append(acc)

        return torch.tensor(np.mean(losses)), torch.tensor(np.mean(top1_acc))

    def test(self, model_test: nn.Module):
        validation_loss, validation_top1_acc = self._eval(model_test, self.validation_dataloader)
        test_loss, test_top1_acc = self._eval(model_test, self.test_dataloader)

        return validation_loss, validation_top1_acc, test_loss, test_top1_acc

    def report(self, model_old, model_new, train_loss, train_acc, validation_loss, validation_acc, test_loss, test_acc, loss_acc_only=False):
        if self.args.verbose:
            print(
                f"Client {self.idx} finished."
            )

        if self.args.use_ray:
            model_old.to("cpu")
            model_new.to("cpu")
        sd_old = model_old.state_dict()
        sd_new = model_new.state_dict()
        # updated head
        sd_local = {key: sd_new[key] for key in sd_new.keys() if key in self.local_keys} if not loss_acc_only else None
        # model difference
        sd_global_diff = {key: sd_new[key] - sd_old[key] for key in sd_new.keys() if key in self.global_keys} if not loss_acc_only else None


        result_dict = {
            "train loss": train_loss.cpu(),
            "train acc": train_acc.cpu(),
            "validation loss": validation_loss.cpu(),
            "validation acc": validation_acc.cpu(),
            "test loss": test_loss.cpu(),
            "test acc": test_acc.cpu(),
            "sd_local": sd_local,
            "sd_global_diff": sd_global_diff,
            "PE": self.PE if self.idx == 0 else None
        }
        return result_dict


    def step(self, step: int):
        raise NotImplementedError

class Server:
    def __init__(self,
                 args,
                 model: nn.Module,
                 global_keys: List[str],
                 local_keys: List[str],
                 fine_tune_keys: List[str],
                 clients: List[Client],
                 remote_workers: List[Worker],
                 logger: Logger,
                 device):
        self.args = args
        self.model = model
        self.global_keys = global_keys
        self.local_keys = local_keys
        self.fine_tune_keys = fine_tune_keys
        self.clients = clients
        self.remote_workers = remote_workers
        self.logger = logger
        self.device = device
        if not args.disable_dp and args.dp_type == "user-level-DP":
            self.accountant = RDPAccountant()
            if self.args.noise_multiplier < 0:
                self.noise_multiplier = get_noise_multiplier(
                    target_epsilon=args.epsilon,
                    target_delta=args.delta,
                    sample_rate=args.frac_participate,
                    epochs=args.epochs * int(1 / args.frac_participate)
                )
                print(
                    f"[ To achieve ({args.epsilon}, {args.delta}) user-level DP, the noise multiplier is set to {self.noise_multiplier} ]"
                )
            else:
                self.noise_multiplier = self.args.noise_multiplier
                print(
                    f"[ user-level DP. The noise multiplier is manually set to {self.noise_multiplier} ]"
                )
            self.clip_threshold = self.args.dp_clip
        else:
            self.noise_multiplier = 0
            self.accountant = None
            self.clip_threshold = -1

    def broadcast(self, clients: List[Client]):
        sd_server = self.model.state_dict()
        for client in clients:
            sd_client = client.model.state_dict()
            # for key in self.local_keys:
            #     sd_client[key] = sd_client[key].to(self.device)
            for key in self.global_keys:
                sd_client[key] = sd_server[key] if len(self.remote_workers) != 0 else copy.deepcopy(sd_server[key])
            client.model.load_state_dict(sd_client)

    def aggregate(self, sds_global_diff: List[OrderedDict]):
        '''
            Only the simplest average aggregation is implemented
        '''
        sd = self.model.state_dict()

        noise_level = self.args.dp_clip * self.noise_multiplier

        sd = server_update_with_clip(sd, sds_global_diff, self.global_keys, self.clip_threshold,
                                     self.args.global_lr, noise_level, self.args.aggr)

        self.model.load_state_dict(sd)

    def local_update(self, clients: List[Client], epoch: int):
        '''
            Server orchestrates the clients to perform local updates.
            The current implementation did not use ray backend.
        '''
        results = compute_with_remote_workers(self.remote_workers, clients, epoch)

        stats_dict_all = {
            "train loss": torch.stack([result["train loss"] for result in results]),
            "train acc": torch.stack([result["train acc"] for result in results]),
            "validation loss": torch.stack([result["validation loss"] for result in results]),
            "validation acc": torch.stack([result["validation acc"] for result in results]),
            "test loss": torch.stack([result["test loss"] for result in results]),
            "test acc": torch.stack([result["test acc"] for result in results]),
        }
        self.logger.log(stats_dict_all, epoch)

        result_dict = {
            "train loss": torch.mean(torch.stack([result["train loss"] for result in results])),
            "train acc": torch.mean(torch.stack([result["train acc"] for result in results])),
            "validation loss": torch.mean(torch.stack([result["validation loss"] for result in results])),
            "validation acc": torch.mean(torch.stack([result["validation acc"] for result in results])),
            "test loss": torch.mean(torch.stack([result["test loss"] for result in results])),
            "test acc": torch.mean(torch.stack([result["test acc"] for result in results])),
            "sds_local": [result["sd_local"] for result in results],
            "sds_global_diff": [result["sd_global_diff"] for result in results],
            "PEs": [result["PE"] for result in results]
        }
        return result_dict

    def step(self, epoch: int):
        raise NotImplementedError

    def report(self, epoch, results: Results):
        train_loss, train_acc, validation_loss, validation_acc, test_loss, test_acc = results.mean()
        if not self.args.disable_dp:
            for client in self.clients: # only client[0] maintains the accountant history
                if client.idx == 0:
                    accountant = self.accountant if self.args.dp_type == 'user-level-DP' else client.PE.accountant
                    noise_multiplier = self.noise_multiplier if self.args.dp_type == 'user-level-DP' else client.noise_multiplier
                    break

        if (epoch % self.args.print_freq == 0 or epoch > self.args.epochs - 5) and epoch >= 0:
            if not self.args.disable_dp and noise_multiplier > 0:
                epsilon, best_alpha = accountant.get_privacy_spent(
                    delta=self.args.delta
                )
                print(
                    f"On {self.args.dataset} using {self.args.alg} with {self.args.frac_participate * 100}\% par. rate, "
                    f"Epoch: {epoch} \t"
                    f"Loss: {train_loss:.2f} "
                    f"Acc@1: {train_acc * 100:.2f} "
                    f"(ε = {epsilon:.2f}, δ = {self.args.delta}) for α = {best_alpha}\t"
                    "[ TRAIN ]"
                )
                print(
                    f"On {self.args.dataset} using {self.args.alg} with {self.args.frac_participate * 100}\% par. rate, "
                    f"Epoch: {epoch} \t"
                    f"Loss: {validation_loss:.2f} "
                    f"Acc@1: {validation_acc * 100:.2f} "
                    f"(ε = {epsilon:.2f}, δ = {self.args.delta}) for α = {best_alpha}\t"
                    "[ VALIDATION ]"
                )
                print(
                    f"On {self.args.dataset} using {self.args.alg} with {self.args.frac_participate * 100}\% par. rate, "
                    f"Epoch: {epoch} \t"
                    f"loss: {test_loss:.2f} "
                    f"acc@1: {test_acc * 100:.2f} "
                    f"(ε = {epsilon:.2f}, δ = {self.args.delta}) for α = {best_alpha}\t"
                    "[ TEST ]"
                )
            else:
                print(
                    f"On {self.args.dataset} using {self.args.alg} with {self.args.frac_participate * 100}\% par. rate, "
                    f"Epoch: {epoch} \t"
                    f"Loss: {train_loss:.2f} "
                    f"Acc@1: {train_acc * 100:.2f} "
                    "[ TRAIN ]"
                      )
                print(
                    f"On {self.args.dataset} using {self.args.alg} with {self.args.frac_participate * 100}\% par. rate, "
                    f"Epoch: {epoch} \t"
                    f"Loss: {validation_loss:.2f} "
                    f"Acc@1: {validation_acc * 100:.2f} "
                    "[ VALIDATION ]"
                )
                print(
                    f"On {self.args.dataset} using {self.args.alg} with {self.args.frac_participate * 100}\% par. rate, "
                    f"Epoch: {epoch} \t"
                    f"loss: {test_loss:.2f} "
                    f"acc@1: {test_acc * 100:.2f} "
                    "[ TEST ]"
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

