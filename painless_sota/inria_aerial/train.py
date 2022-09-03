import torch
from omegaconf import DictConfig

import pytorch_toolbelt.utils
from painless_sota.inria_aerial.hydra_utils import hydra_dpp_friendly_main, register_albumentations_resolver
from painless_sota.inria_aerial.pipeline import InriaAerialPipeline
from pytorch_toolbelt.utils.distributed import is_dist_avail_and_initialized


@hydra_dpp_friendly_main(config_path="configs", config_name="train", version_base="1.2")
def main(config: DictConfig) -> None:
    pytorch_toolbelt.utils.set_manual_seed(int(config.seed))

    torch.cuda.empty_cache()
    torch.set_anomaly_enabled(config.torch.detect_anomaly)

    torch.backends.cuda.matmul.allow_tf32 = config.torch.cuda_allow_tf32
    torch.backends.cudnn.deterministic = config.torch.deterministic
    torch.backends.cudnn.benchmark = config.torch.benchmark
    torch.backends.cudnn.allow_tf32 = config.torch.cudnn_allow_tf32

    register_albumentations_resolver()
    InriaAerialPipeline(config).train()

    # This is necessary if we are running DDP hydra sweep mode
    if is_dist_avail_and_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
