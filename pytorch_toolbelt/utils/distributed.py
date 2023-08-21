import gc
import inspect
import logging
import os
import pickle
import typing
import functools
from typing import Any, Dict, List, Optional
import numpy as np
import torch
from torch import Tensor

import torch.distributed as dist

from pytorch_toolbelt.utils.bucket_assignment import (
    naive_bucket_assignment,
    compute_bucket_imbalance_score,
    random_bucket_assignment,
    filler_bucket_assignment,
)

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
    "split_across_nodes",
    "master_node_only"
]

logger = logging.getLogger("pytorch_toolbelt.utils.distributed")


class DistributedGuard:
    def __init__(
        self,
        local_rank: int = os.environ.get("LOCAL_RANK", 0),
        world_size: int = os.environ.get("WORLD_SIZE", 1),
        visible_devices: List = os.environ.get("CUDA_VISIBLE_DEVICES", list(range(torch.cuda.device_count()))),
    ):
        self.local_rank = int(local_rank)
        self.world_size = int(world_size)
        self.visible_devices = visible_devices
        self.dist_is_available = torch.distributed.is_available()
        self.dist_is_initialized = torch.distributed.is_initialized()
        self.device = torch.device(f"cuda:{self.local_rank}")

        logger.info(
            f"Creating DistributedGuard with devices {self.visible_devices} {self.local_rank}/{self.world_size}"
        )

    def __enter__(self):
        if self.dist_is_available and self.world_size > 1:
            if self.dist_is_initialized:
                raise RuntimeError("Torch distributed is already initialized. This indicates an error.")

            torch.cuda.set_device(self.device)
            logger.info(f"Setting CUDA device {self.device} for rank {self.local_rank}/{self.world_size}")
            torch.distributed.init_process_group(backend="nccl", world_size=self.world_size, rank=self.local_rank)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if self.dist_is_available and self.dist_is_initialized:
                torch.distributed.barrier()
                torch.distributed.destroy_process_group()
        except Exception as e:
            logger.exception(e)
        finally:
            torch.cuda.empty_cache()
            gc.collect()


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
        storage = torch.UntypedStorage.from_buffer(buffer, dtype=torch.uint8)
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
    storage = torch.UntypedStorage.from_buffer(buffer, dtype=torch.uint8)
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


def split_across_nodes(
    collection: List,
    world_size: Optional[int] = None,
    local_rank: Optional[int] = None,
    cost: Optional[List] = None,
    method: str = "optimal",
) -> List:
    """
    Split input collection such that each node receives 1/N of the total collection elements to process, where
    N is the number of nodes.

    Example:

    >>> local_values = split_across_nodes([0,1,2,3,4,5,6,7,8,9])
    >>> print(local_values, get_rank())
    >>> # [0,1,2], 0
    >>> # [3,4,5], 1
    >>> # [6,7,8], 2
    >>> # [9], 3

    Args:
        collection: Initial collection of size N to split into K nodes
        world_size: World size (Number of nodes K)
        local_rank: Current node
        cost: A vector of size N that represents the cost of processing associated with each item.
              If present, it will affect the order of elements each node will receive to even the total cost each node
              will get.
        method: Bucket assignment method used to assign each sample of associated cost to specific node to minimze std of total cost per node.

                naive - Sort elements by cost then assing them in repeating patterm: [0, 1, 2, 3, 0, 1, 2, 3, 4, ...].
                Each node gets exactly the same number of samples, however cost per node may vary greatly.

                optimal - Iteratively assigns elements starting with the most costly ones to the least used bucket.
                This gives much better cost balance per node, but the number of items each node gets it not guaratneed to be equal at all.
    Returns:

    """
    if world_size is None:
        world_size = get_world_size()
    if local_rank is None:
        local_rank = get_rank()

    if world_size > 1:
        if cost is not None:
            if len(cost) != len(collection):
                raise RuntimeError()

            method_fn = {
                "optimal": filler_bucket_assignment,
                "naive": naive_bucket_assignment,
            }[method]
            assigned_indexes = method_fn(cost, world_size)

            rank_local_indexes = assigned_indexes == local_rank

            logger.debug(
                f"Node {local_rank} get {np.count_nonzero(rank_local_indexes)} items with total cost {sum(cost[rank_local_indexes])}"
            )

            if isinstance(collection, np.ndarray):
                rank_specific_subset = collection[rank_local_indexes]
            else:
                rank_specific_subset = [
                    collection[index] for index, should_pick in enumerate(rank_local_indexes) if should_pick
                ]

        else:
            indexes = np.linspace(0, len(collection), int(world_size + 1), dtype=int)
            rank_local_indexes = slice(indexes[local_rank], indexes[local_rank + 1])
            rank_specific_subset = collection[rank_local_indexes]

            logger.debug(
                f"split_across_nodes returning slice {rank_local_indexes} from collection of size {len(collection)} for rank {local_rank}"
            )
        return rank_specific_subset
    else:
        return collection

def master_node_only(func):
    """
    A decorator for making sure a function runs only in main process.
    If not in DDP mode (local_rank = -1), the function will run.
    If in DDP mode, the function will run only in the main process (local_rank = 0)
    This works only for functions with no return value
    """

    return_type = inspect.signature(func).return_annotation
    function_has_return_value = return_type is not None and return_type != inspect._empty
    if function_has_return_value:
        raise RuntimeError(f"Function {func} decorated with @master_node_only must not return any value. "
                           f"Function signature: {inspect.signature(func)}")

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if is_main_process():
            return func(*args, **kwargs)
        else:
            return None

    return wrapper