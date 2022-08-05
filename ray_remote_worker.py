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
        num_remote_workers = len(remote_workers)
        jobs_clients = {
            wid: [] for wid in range(num_remote_workers)
        }
        jobs_PEs = {
            wid: [] for wid in range(num_remote_workers)
        }
        wid = 0 # worker id
        for client, PE in zip(clients, PEs):
            jobs_clients[wid].append(client)
            jobs_PEs[wid].append(PE)

            wid = (wid + 1) % num_remote_workers

        job_ids = [remote_worker.step.remote(jobs_clients[idx], jobs_PEs[idx])
                  for idx, remote_worker in enumerate(remote_workers)]

        results = ray.get(job_ids)

        results = list(itertools.chain.from_iterable(results))
    else:
        results = [client.step(PE) for client, PE in zip(clients, PEs)]

    return results