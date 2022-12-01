import logging
import os
import pickle
from typing import Any, Dict, List

import torch
from torch import Tensor

import torch.distributed as dist


__all__ = [
    "distributed_guard",
    "DistributedGuard",
    "all_gather",
    "broadcast_from_master",
    "get_rank",
    "get_world_size",
    "is_dist_avail_and_initialized",
    "is_main_process",
    "master_print",
    "reduce_dict_sum",
]

logger = logging.getLogger("DistributedGuard")


class DistributedGuard:
    def __init__(
        self,
        local_rank: int = os.environ.get("LOCAL_RANK", -1),
        world_size: int = os.environ.get("WORLD_SIZE", -1),
        visible_devices: List = os.environ.get("CUDA_VISIBLE_DEVICES", list(range(torch.cuda.device_count()))),
    ):
        self.local_rank = int(local_rank)
        self.world_size = int(world_size)
        self.visible_devices = visible_devices
        self.dist_is_available = torch.distributed.is_available()
        self.dist_is_initialized = torch.distributed.is_initialized()

        logger.info(
            f"Creating DistributedGuard with devices {self.visible_devices} {self.local_rank}/{self.world_size}"
        )

    def __enter__(self):
        if self.dist_is_available and self.world_size > 1:
            if self.dist_is_initialized:
                raise RuntimeError("Torch distributed is already initialized. This indicates an error.")

            device = torch.device(f"cuda:{self.local_rank}")
            torch.cuda.set_device(device)
            logger.info(f"Setting CUDA device %s for rank %d/%d", str(device), self.local_rank, self.world_size)
            torch.distributed.init_process_group(backend="nccl")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if self.dist_is_available and self.dist_is_initialized():
                torch.distributed.barrier()
                torch.distributed.destroy_process_group()
        except:
            pass


def distributed_guard(func):
    def inner1(*args, **kwargs):
        with DistributedGuard():
            return func(*args, **kwargs)

    return inner1


def is_dist_avail_and_initialized() -> bool:
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size() -> int:
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process() -> bool:
    return get_rank() == 0


def broadcast_from_master(data: Any) -> Any:
    """
    Broadcast data from master node to all other nodes. This may be required when you
    want to compute something only on master node (e.g computational-heavy metric) and
    don't want to vaste CPU of other nodes doing same work simultaneously.

    >>> if is_main_process():
    >>>    result = some_code_to_run(...)
    >>> else:
    >>>    result = None
    >>> # 'result' propagated to all nodes from master
    >>> result = broadcast_from_master(result)

    Args:
        data: Data to be broadcasted from master node (rank 0)

    Returns:
        Data from rank 0 node
    """
    world_size = get_world_size()
    if world_size == 1:
        return data

    local_rank = get_rank()
    storage: Tensor

    if local_rank == 0:
        buffer = pickle.dumps(data)
        storage = torch.ByteStorage.from_buffer(buffer)
        payload = torch.ByteTensor(storage).to("cuda")
        local_size = payload.numel()
    else:
        local_size = 0

    # Propagate target tensor size to all nodes
    local_size = max(all_gather(local_size))
    if local_rank != 0:
        payload = torch.empty((local_size,), dtype=torch.uint8, device="cuda")

    dist.broadcast(payload, 0)
    buffer = payload.cpu().numpy().tobytes()
    return pickle.loads(buffer)


def all_gather(data: Any) -> List[Any]:
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device="cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def reduce_dict_sum(input_dict: Dict[Any, Any]) -> Dict[Any, Any]:
    """
    Reduce the values in the dictionary from all processes so that all processes
    have the sum of the all values. Returns a dict with the same fields as
    input_dict, after reduction.

    Note: This function can work on defaultdict(list) and will effectively
    concatenate multiple dictionaries into single one, thanks to that summation
    of lists works as a concatenation.
    Args:
        input_dict (dict): all the values will be reduced
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict

    with torch.no_grad():
        reduced_dict = {}
        for x in all_gather(input_dict):
            for key, value in x.items():
                if key in reduced_dict:
                    reduced_dict[key] += value
                else:
                    reduced_dict[key] = value
    return reduced_dict


def master_print(*args, **kwargs) -> None:
    """
    Drop-in replacement for built-in `print` function that prints only on master node

    Args:
        *args:
        **kwargs:

    Returns:
        None
    """
    if is_main_process():
        print(*args, **kwargs)
