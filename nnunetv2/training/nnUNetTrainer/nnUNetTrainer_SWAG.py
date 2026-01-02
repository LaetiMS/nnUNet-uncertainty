#from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.lr_schedule.nnUNetTrainer_warmup import nnUNetTrainer_warmup
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p
import torch
import os


class nnUNetTrainer_warmup_SWAG(nnUNetTrainer_warmup):
    """
    Saves checkpoints during training to allow performing SWAG:) Other than that the training is not impacted at all
    """
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
    # def __init__(self, *args, swag_start_epoch: int = 800, swag_interval: int = 10, **kwargs) -> None:
    #     super().__init__(*args, **kwargs)
        super().__init__(plans,configuration,fold,dataset_json,device)
        self.swag_start_epoch = 800
        self.swag_interval = 10
        # todo: add those new parameters to the config file:)

    def on_epoch_end(self):
        # SWAG snapshots (inspired by save_checkpointing -> but as it is by default disabled -> didnt want to reuse fucntion)
        if self.current_epoch >= self.swag_start_epoch and (self.current_epoch - self.swag_start_epoch) % self.swag_interval == 0:

            # Ensure SWAG folder exists
            swag_dir = join(self.output_folder, "swag_snapshots")
            maybe_mkdir_p(swag_dir)

            # Define snapshot path
            snapshot_path = join(swag_dir, f"epoch_{self.current_epoch}.pth")
            #old: snapshot_path = join(self.output_folder, f'swag_snapshot_epoch_{self.current_epoch}.pth')

            if self.is_ddp:
                torch.save(self.network.module.state_dict(), snapshot_path)
            else:
                torch.save(self.network.state_dict(), snapshot_path)
            if self.local_rank == 0:
                self.print_to_log_file(f"[SWAG] Saved snapshot at epoch {self.current_epoch} -> {snapshot_path}")
        super().on_epoch_end()

class nnUNetTrainer_warmup_swag1000_tr1200_epochs(nnUNetTrainer_warmup_SWAG):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans,configuration,fold,dataset_json,device)
        self.swag_start_epoch = 1000
        self.swag_interval = 10
        self.total_epochs = 1200

class nnUNetTrainer_warmup_swag400_tr600_epochs(nnUNetTrainer_warmup_SWAG):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans,configuration,fold,dataset_json,device)
        self.swag_start_epoch = 500
        self.swag_interval = 5
        self.total_epochs = 600

class nnUNetTrainer_TEST_warmup_swag0_tr20_epochs(nnUNetTrainer_warmup_SWAG):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans,configuration,fold,dataset_json,device)
        self.warmup_duration_whole_net = 0
        self.swag_start_epoch = 0
        self.swag_interval = 1
        self.total_epochs = 20


