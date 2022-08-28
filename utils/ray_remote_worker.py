import ray
import itertools

import torch.cuda

import gc

class Worker:
    def __init__(self, n_gpus: int, wid: int):
        self.wid = wid
        self.n_gpus = n_gpus

    def step(self, clients, epoch):
        result = []

        for client in clients:
            result.append(client.step(epoch))

        gc.collect()
        torch.cuda.empty_cache()
        return result

def create_remote_workers(args):
    if args.n_gpus > 0 and args.use_ray:
        RemoteWorker = ray.remote(num_gpus=args.ray_gpu_fraction)(Worker)
        n_remote_workers = int(1 / args.ray_gpu_fraction) * args.n_gpus
        print(
            f"[ Creating {n_remote_workers} remote workers altogether. ]"
        )
        remote_workers = [RemoteWorker.remote(args.n_gpus, wid) for wid in range(n_remote_workers)]
    else:
        print(
            f"[ No remote workers is created. Clients are evaluated sequentially. ]"
        )
        remote_workers = []
    return remote_workers

def compute_with_remote_workers(remote_workers, clients, epoch):
    if len(remote_workers) != 0:
        # Use remote workers to accelerate computation.
        num_remote_workers = len(remote_workers)
        num_clients = len(clients)


        # calculate how many clients a single worker should handle
        n_clients_per_worker = [int(num_clients/num_remote_workers)] * num_remote_workers
        for i in range(num_clients % num_remote_workers):
            n_clients_per_worker[i] += 1


        # assign clients to workers according to the above calculation.
        # IMPORTANT: the clients are assigned sequentially so that when aggregated, the order of the results will be
        #            the same as the order of the clients!
        jobs_clients = {
            wid: None for wid in range(num_remote_workers)
        }
        cid = 0 # client id
        for wid in range(num_remote_workers):
            jobs_clients[wid] = clients[cid: cid + n_clients_per_worker[wid]]
            cid += n_clients_per_worker[wid]
            ###### Uncomment for the purpose of DEBUG
            # client_ids = [client.idx for client in jobs_clients[wid]]
            # print(
            #     f"Worker {wid} handles clients {client_ids}."
            # )
            ###### Uncomment for the purpose of DEBUG

        ray_job_ids = [remote_worker.step.remote(jobs_clients[wid], epoch)
                  for wid, remote_worker in enumerate(remote_workers)]
        # Calling remote functions only creates job ids. Use ray.get() to actually carry out these jobs.
        results = ray.get(ray_job_ids)

        # IMPORTANT: "results" should have the same order as "clients" and "PEs"! Incorrect ordering will cause BUG!
        results = list(itertools.chain.from_iterable(results))
    else:
        # No remote workers are available. Simply evaluate sequentially.
        results = [client.step(epoch) for client in clients]

    return results