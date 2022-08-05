import ray
import itertools

class Worker:
    def step(self, clients, PEs):

        result = []

        for client, PE in zip(clients, PEs):
            result.append(client.step(PE))

        return result

def compute_with_remote_workers(remote_workers, clients, PEs):
    if remote_workers is not None:
        # Use remote workers to accelerate computation.
        num_remote_workers = len(remote_workers)
        num_clients = len(clients)
        n_clients_per_worker = [int(num_clients/num_remote_workers)] * num_remote_workers
        for i in range(num_clients % num_remote_workers):
            n_clients_per_worker[i] += 1

        print(n_clients_per_worker)
        jobs_clients = {
            wid: None for wid in range(num_remote_workers)
        }
        jobs_PEs = {
            wid: None for wid in range(num_remote_workers)
        }
        cid = 0 # client id
        for wid in range(num_remote_workers):
            jobs_clients[wid] = clients[cid: cid + n_clients_per_worker[wid]]
            jobs_PEs[wid] = PEs[cid: cid + n_clients_per_worker[wid]]
            cid += n_clients_per_worker[wid]


            # Uncomment for the purpose of DEBUG
            # client_ids = [client.idx for client in jobs_clients[wid]]
            # print(
            #     f"Worker {wid} handles clients {client_ids}."
            # )


        ray_job_ids = [remote_worker.step.remote(jobs_clients[wid], jobs_PEs[wid])
                  for wid, remote_worker in enumerate(remote_workers)]
        # Calling remote functions only creates job ids. Use ray.get() to actually carry out these jobs.
        results = ray.get(ray_job_ids)

        # IMPORTANT: "results" should have the same order as "clients" and "PEs"! Incorrect ordering will cause BUG!
        results = list(itertools.chain.from_iterable(results))
    else:
        # No remote workers are available. Simply evaluate sequentially.
        results = [client.step(PE) for client, PE in zip(clients, PEs)]

    return results