from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p
import torch
import os


class nnUNetTrainer_SWAG(nnUNetTrainer):
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

    def on_validation_epoch_start(self):
        #todo: is it better to implement MCDropout here or not? NB: mcdropout parameter not yet defined
        super().on_validation_epoch_start()
        if self.mcdropout:
            self.network.module.dropout.train()
