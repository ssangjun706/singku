import os
import sys
import getpass
from setproctitle import *
from multiprocessing import get_context

import torch.nn as nn
import torch.cuda as cuda
import torch.distributed as dist

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from typing import Callable, Generator

from contextlib import contextmanager

@contextmanager
def suppress_output():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


class DistributedDataLoader:
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = False,
        drop_last: bool = False,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.device_count = cuda.device_count()

        assert (
            self.batch_size % self.device_count == 0
        ), f"(batch size) % (device count) != 0"
        self.batch_size //= self.device_count

        self.sampler = DistributedSampler(
            dataset=self.dataset,
            shuffle=self.shuffle,
            drop_last=self.drop_last,
        )

        self.loader = DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            sampler=self.sampler,
            shuffle=False,
            drop_last=False,
        )

    def __iter__(self):
        return iter(self.loader)

    def __len__(self):
        return len(self.loader)


class DistributedParallel(DistributedDataParallel):
    def __init__(
        self,
        module: nn.Module,
        device: int,
        find_unused_parameters: bool = False,
        name: str = '',
    ):
        setproctitle(f"{getpass.getuser()}/python/parallel/worker/{name}")
        cuda.set_device(device)
        module.to(device)
        super().__init__(
            module=module,
            device_ids=[device],
            find_unused_parameters=find_unused_parameters,
        )

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)


class DistributedTrainer:
    def __init__(self, 
        func: Callable, 
        port: int = 11111, 
        backend: str = "nccl", 
        device_ids: list[int] = None,
        gather: bool = False,
    ):
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, device_ids))
        self.backend = backend
        self.port = port
        self.func = func
        self.world_size = cuda.device_count()
        self.gather = gather

    def worker(self, ngpus_per_node: int, rank: int, queue, use_yield: bool):
        def runner():
            if use_yield:
                for item in self.func(rank):
                    queue.put(item)
                    dist.barrier()
            else:
                result = self.func(rank)
                queue.put(result)

        try:
            dist.init_process_group(
                backend=self.backend,
                init_method=f"tcp://localhost:{self.port}",
                world_size=ngpus_per_node,
                rank=rank,
            )
            if rank == 0:
                runner()
            else:
                with suppress_output():
                    runner()
        finally:
            dist.destroy_process_group()
            queue.put(None)


    def __iter__(self) -> Generator:
        context = get_context("spawn")
        queue = context.Queue()
        processes = []
        buffer = []
        terminate_counter = 0
        try:
            for rank in range(self.world_size):
                p = context.Process(
                    target=self.worker, 
                    args=(self.world_size, rank, queue, True)
                )
                p.start()
                processes.append(p)

            while terminate_counter < self.world_size:
                output = queue.get()
                if output is None:
                    terminate_counter += 1
                    continue
                if self.gather:
                    buffer.append(output)
                    if len(buffer) == self.world_size:
                        buffer = tuple(
                            sum(values) / len(values) for values in zip(*buffer)
                                    )
                        yield buffer
                        buffer = []
                else:
                    yield output
        finally:
            for p in processes:
                p.join()
            print("All processes have finished.")

    
    def __call__(self) -> list:
        context = get_context("spawn")
        queue = context.Queue()
        processes = []
        buffer = []
        terminate_counter = 0
        try:
            for rank in range(self.world_size):
                p = context.Process(
                    target=self.worker, 
                    args=(self.world_size, rank, queue, False)
                )
                p.start()
                processes.append(p)

            while terminate_counter < self.world_size:
                output = queue.get()
                if output is None:
                    terminate_counter += 1
                    continue
                buffer.append(output)

            if self.gather:
                buffer = tuple(sum(values) / len(values) for values in zip(*buffer))
            return buffer
        finally:
            for p in processes:
                p.join()
            print("All processes have finished.")