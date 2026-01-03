# wandb logging implementation inspired by : LuxImagingAI/nnUNet-wandb
# wandb figure implementation inspired by tomDag25 : https://github.com/MIC-DKFZ/nnUNet/issues/2733
from typing import Union

import torch
import yaml
import numpy as np

import matplotlib.pyplot as plt

from torch import autocast
from torch.optim import AdamW
from torch._dynamo import OptimizedModule
from torch.nn.parallel import DistributedDataParallel as DDP



from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.lr_scheduler.warmup import Lin_incr_LRScheduler, PolyLRScheduler_offset


from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

from nnunetv2.utilities.helpers import dummy_context, empty_cache

from nnunetv2.training.nnUNetTrainer.variants.WandbWrapper import WandbWrapper
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p

from importlib.resources import files


import torch.distributed as dist

# to check if main process is running -> for wandb logging
def is_main_process():
    return (
        not dist.is_available()
        or not dist.is_initialized()
        or dist.get_rank() == 0
    )

# This function creates a plot that we will send to WandB

def plot_slices_combined(combined, gt, pred, current_epoch, debug=False):
    """
    Plot the image, ground truth and prediction of the mid-sagittal axial slice
    The orientaion is assumed to RPI
    """

    mid_sagittal = combined.shape[2] // 2

    # plot X slices before and after the mid-sagittal slice in a grid
    fig, axs = plt.subplots(3, 6, figsize=(10, 6))
    fig.suptitle(f'Epoch: {current_epoch}: T2 Image --> Other contrast --> Ground Truth --> Prediction')
    if np.all(combined == 0):
        print("Array contains only zeros")
    for i in range(6):
        axs[0, i].imshow(combined[:, :, mid_sagittal - 3 + i].T, cmap='gray')
        axs[0, i].axis('off')
        axs[1, i].imshow(gt[:, :, mid_sagittal - 3 + i].T)
        axs[1, i].axis('off')
        axs[2, i].imshow(pred[:, :, mid_sagittal - 3 + i].T)
        axs[2, i].axis('off')

    plt.tight_layout()
    fig.show()
    return fig


class nnUNetTrainerWb_SWAG_warmup_optimizer(nnUNetTrainer):
    """
    Varies from the traditional nnUNetTrainer by:
    - Incorporating Wandb logging
    - Saving checkpoints for SWAG
    - Allowing to choose between the default or the adamw optimizer -> use_adamw_optimizer
    """
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        # run og nnUNetTrainer initialization as intended
        super().__init__(plans, configuration, fold, dataset_json, device = device)
        # swag parameters
        self.swag_start_epoch = 800
        self.swag_interval = 10

        self.use_adamw_optimizer = False

        #### hyperparameters for warmup
        self.warmup_duration_whole_net = 50  # lin increase whole network, set 0 if no warmup
        self.training_stage = None  # 'warmup_all', 'train'

        #self._wandb_init()
    def initialize(self):
        super().initialize()
        self._wandb_init()

    def _wandb_init(self) -> None:
        """
        used to set the configuration to initialize wandb
        """
        # Hyperparameters initialization
        yaml_config = dict()
        yaml_config['configuration_name'] = self.configuration_name
        yaml_config['fold'] = self.fold
        yaml_config['num_epochs'] = self.num_epochs
        yaml_config['initial_lr'] = self.initial_lr
        yaml_config['weight_decay'] = self.weight_decay
        yaml_config['num_iterations_per_epoch'] = self.num_iterations_per_epoch
        yaml_config['num_val_iterations_per_epoch'] = self.num_val_iterations_per_epoch
        yaml_config['oversample_foreground_percent'] = self.oversample_foreground_percent
        yaml_config['probabilistic_oversampling'] = self.probabilistic_oversampling
        yaml_config['enable_deep_supervision'] = self.enable_deep_supervision
        yaml_config['plans_manager'] = self.plans_manager
        yaml_config['configuration_manager'] = self.configuration_manager
        yaml_config['configuration_name'] = self.configuration_name
        yaml_config['dataset_json'] = self.dataset_json
        yaml_config['output_folder'] = self.output_folder
        yaml_config['network'] = self.network
        yaml_config['optimizer'] = self.optimizer

        yaml_config['swag_start_epoch'] = self.swag_start_epoch
        yaml_config['swag_interval'] = self.swag_interval
        yaml_config['warmup_duration_whole_net'] =self.warmup_duration_whole_net
        yaml_config['use_adamw_optimizer'] = self.use_adamw_optimizer

        # WandbWrapper initialization
        self.wandb = WandbWrapper(use_wandb=yaml_config['wandb_enabled'], config=yaml_config)
        # self.wandb.init()


    def on_train_start(self):
        super().on_train_start()
        self.wandb.init()


    def on_train_end(self):
        super().on_train_end()
        if is_main_process():
            self.wandb.finish()

    # train_step -> batch_id was added as  modified
    def train_step(self, batch: dict, batch_id: int) -> dict:
        # changes from original:
        # batch_id as parameter
        # at batch_id = 0 a figure gets plotted of middle slices and is saved to wandb.log
        # saves training_loss to wandb.log

        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            # del data
            l = self.loss(output, target)

            # only modification to train_stp to plot middle slices and save to wandb only for the first batch_id
            if is_main_process() and batch_id == 0 and self.configuration_name == '3d_fullres':
                train_image = data[0].detach().cpu().squeeze().float().numpy()
                train_gt = target[0].detach().cpu().squeeze().float().numpy()[0]
                train_pred = np.argmax(output[0].detach().cpu().squeeze().numpy(), axis=1)[0]

                fig = plot_slices_combined(combined=train_image,
                                           gt=train_gt,
                                           pred=train_pred,
                                           current_epoch = self.current_epoch)

                self.wandb.log({"training images": self.wandb.Image(fig)})
                plt.close(fig)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        self.wandb.log({"training_loss_per_it": l.detach().cpu().numpy()})
        return {'loss': l.detach().cpu().numpy()}


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

        #wandb logging
        if is_main_process():
            self.wandb.log({"epoch": self.current_epoch, "val_loss": self.logger.my_fantastic_logging['val_losses'][-1],
                            "training_loss": self.logger.my_fantastic_logging['train_losses'][-1],
                            "lr": self.optimizer.param_groups[0]['lr']})

            all_dice = [np.round(i, decimals=4) for i in self.logger.my_fantastic_logging['dice_per_class_or_region'][-1]]
            dice_val = np.average(all_dice)  # exclude background in the average dice
            self.wandb.log({"Average Dice": np.round(dice_val, decimals=4)})

            for label_name, label_idx in self.dataset_json['labels'].items():
                # Skip the 'background' or any label with index 0

                if label_idx == 0:
                    continue

                all_dice_idx = label_idx - 1

                if 0 <= all_dice_idx < len(all_dice):
                    dice_score = np.round(all_dice[all_dice_idx], decimals=4)

                    self.wandb.log({f"{label_name} Dice": dice_score})
            # handle 'best' checkpointing. ema_fg_dice is computed by the logger and can be accessed like this
            if self._best_ema is None or self.logger.my_fantastic_logging['ema_fg_dice'][-1] > self._best_ema:
                self.wandb.log({"best EMA pseudo Dice": self._best_ema})

        super().on_epoch_end() # added at the end, cause at end of on_epoch_end self.current_epoch += 1

    def validation_step(self, batch: dict) -> dict:
        logger_dict = super().validation_step(batch)
        if is_main_process():
            logger_dict_main = logger_dict.copy()
            #rename loss to validation loss, to avoid confusion with loss from training
            validation_loss = logger_dict_main['loss']
            logger_dict_main.pop('loss')
            logger_dict_main['val_loss'] = validation_loss

            self.wandb.log(logger_dict_main)
        return logger_dict

    def run_training(self):
        # only modification: self.train_step from Subclass now requires batch_id to plot slices to wandb
        self.on_train_start()

        for epoch in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start()

            self.on_train_epoch_start()
            train_outputs = []
            for batch_id in range(self.num_iterations_per_epoch):
                train_outputs.append(self.train_step(next(self.dataloader_train), batch_id)) # modified to included batch_id (to plot slices to wandb)
            self.on_train_epoch_end(train_outputs)

            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []
                for batch_id in range(self.num_val_iterations_per_epoch):
                    val_outputs.append(self.validation_step(next(self.dataloader_val)))
                self.on_validation_epoch_end(val_outputs)

            self.on_epoch_end()

        self.on_train_end()
    def configure_optimizers(self, stage: str = "warmup_all"):
        # adapted from nnUNetTrainer_warmup and integrating possibility for AdamW optimizer from nnUNetTrainerAdam
        if self.warmup_duration_whole_net != 0:
            assert stage in ["warmup_all", "train"]

            if self.training_stage == stage:
                return self.optimizer, self.lr_scheduler
            if isinstance(self.network, DDP):
                params = self.network.module.parameters()
            else:
                params = self.network.parameters()
            if stage == "warmup_all":
                self.print_to_log_file("train whole net, warmup")
                if self.use_adamw_optimizer:
                    optimizer = AdamW(params,
                                      lr=self.initial_lr,
                                      weight_decay=self.weight_decay,
                                      amsgrad=True)
                else:
                    optimizer = torch.optim.SGD(
                        params, self.initial_lr, weight_decay=self.weight_decay, momentum=0.99, nesterov=True
                    )
                lr_scheduler = Lin_incr_LRScheduler(optimizer, self.initial_lr, self.warmup_duration_whole_net)
                self.print_to_log_file(
                    f"Initialized warmup_all optimizer and lr_scheduler at epoch {self.current_epoch}")
            else:
                self.print_to_log_file("train whole net, default schedule")
                if self.training_stage == "warmup_all":
                    # we can keep the existing optimizer and don't need to create a new one. This will allow us to keep
                    # the accumulated momentum terms which already point in a useful driection
                    optimizer = self.optimizer
                else:
                    if self.use_adamw_optimizer:
                        optimizer = AdamW(params,
                                          lr=self.initial_lr,
                                          weight_decay=self.weight_decay,
                                          amsgrad=True)
                    else:
                        optimizer = torch.optim.SGD(
                            params, self.initial_lr, weight_decay=self.weight_decay, momentum=0.99, nesterov=True
                        )
                lr_scheduler = PolyLRScheduler_offset(
                    optimizer, self.initial_lr, self.num_epochs, self.warmup_duration_whole_net
                )
                self.print_to_log_file(f"Initialized train optimizer and lr_scheduler at epoch {self.current_epoch}")
            self.training_stage = stage
            empty_cache(self.device)
            return optimizer, lr_scheduler

        if self.use_adamw_optimizer:
            # copied from nnUNetTrainerAdam
            optimizer = AdamW(self.network.parameters(),
                              lr=self.initial_lr,
                              weight_decay=self.weight_decay,
                              amsgrad=True)
            # optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
            #                             momentum=0.99, nesterov=True)
            lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
            return optimizer, lr_scheduler
        else:
            return super().configure_optimizers()

    def load_checkpoint(self, filename_or_checkpoint: Union[dict, str]) -> None:
        if self.warmup_duration_whole_net != 0:
            """
            copied from nnUNetTrainer_warmup.py
            We need to overwrite that entire function because we need to fiddle the correct optimizer in between
            loading the checkpoint and applying the optimizer states. Yuck.
            """
            if not self.was_initialized:
                self.initialize()

            if isinstance(filename_or_checkpoint, str):
                checkpoint = torch.load(filename_or_checkpoint, map_location=self.device)
            # if state dict comes from nn.DataParallel but we use non-parallel model here then the state dict keys do not
            # match. Use heuristic to make it match
            new_state_dict = {}
            for k, value in checkpoint["network_weights"].items():
                key = k
                if key not in self.network.state_dict().keys() and key.startswith("module."):
                    key = key[7:]
                new_state_dict[key] = value

            self.my_init_kwargs = checkpoint["init_args"]
            self.current_epoch = checkpoint["current_epoch"]
            self.logger.load_checkpoint(checkpoint["logging"])
            self._best_ema = checkpoint["_best_ema"]
            self.inference_allowed_mirroring_axes = (
                checkpoint["inference_allowed_mirroring_axes"]
                if "inference_allowed_mirroring_axes" in checkpoint.keys()
                else self.inference_allowed_mirroring_axes
            )

            # messing with state dict naming schemes. Facepalm.
            if self.is_ddp:
                if isinstance(self.network.module, OptimizedModule):
                    self.network.module._orig_mod.load_state_dict(new_state_dict)
                else:
                    self.network.module.load_state_dict(new_state_dict)
            else:
                if isinstance(self.network, OptimizedModule):
                    self.network._orig_mod.load_state_dict(new_state_dict)
                else:
                    self.network.load_state_dict(new_state_dict)

            # it's fine to do this every time we load because configure_optimizers will be a no-op if the correct optimizer
            # and lr scheduler are already set up
            if self.current_epoch < self.warmup_duration_whole_net:
                self.optimizer, self.lr_scheduler = self.configure_optimizers("warmup_all")
            else:
                self.optimizer, self.lr_scheduler = self.configure_optimizers("train")

            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            if self.grad_scaler is not None:
                if checkpoint["grad_scaler_state"] is not None:
                    self.grad_scaler.load_state_dict(checkpoint["grad_scaler_state"])

        else:
            super().load_checkpoint()
    def on_train_epoch_start(self):
        # copied from nnUNetTrainer_warmup
        if self.current_epoch == 0:
            self.optimizer, self.lr_scheduler = self.configure_optimizers("warmup_all")
        elif self.current_epoch == self.warmup_duration_whole_net:
            self.optimizer, self.lr_scheduler = self.configure_optimizers("train")

        super().on_train_epoch_start()

class nnUNetTrainerWb_SWAG_warmup_optimizer_1200ep(nnUNetTrainerWb_SWAG_warmup_optimizer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device=device)
        self.num_epochs = 1200
        self.swag_start_epoch = 900

class nnUNetTrainerWb_SWAG_warmup_AdamW_1200ep(nnUNetTrainerWb_SWAG_warmup_AdamW):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device=device)
        self.num_epochs = 1200
        self.swag_start_epoch = 900


class nnUNetTrainerWb_SWAG_warmup_AdamW(nnUNetTrainerWb_SWAG_warmup_optimizer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device=device)

        self.use_adamw_optimizer = True

class nnUNetTrainerWb_SWAG_warmup_AdamW_1200ep(nnUNetTrainerWb_SWAG_warmup_AdamW):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device=device)
        self.num_epochs = 1200
        self.swag_start_epoch = 900


class nnUNetTrainerWb_SWAG_no_warmup_optimizer(nnUNetTrainerWb_SWAG_warmup_optimizer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device=device)
        self.warmup_duration_whole_net = 0

class nnUNetTrainerWb_SWAG_no_warmup_optimizer_1200ep(nnUNetTrainerWb_SWAG_no_warmup_optimizer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device=device)
        self.num_epochs = 1200
        self.swag_start_epoch = 900


class nnUNetTrainerWb_SWAG_no_warmup_AdamW(nnUNetTrainerWb_SWAG_warmup_optimizer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device=device)
        self.warmup_duration_whole_net = 0
        self.use_adamw_optimizer = True

class nnUNetTrainerWb_SWAG_no_warmup_AdamW_1200ep(nnUNetTrainerWb_SWAG_no_warmup_AdamW):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device=device)
        self.num_epochs = 1200
        self.swag_start_epoch = 900




















