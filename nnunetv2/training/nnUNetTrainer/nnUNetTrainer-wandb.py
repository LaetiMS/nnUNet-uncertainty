# wandb logging implementation inspired by : LuxImagingAI/nnUNet-wandb
# wandb figure implementation inspired by tomDag25 : https://github.com/MIC-DKFZ/nnUNet/issues/2733
import torch
import yaml
import numpy as np

import matplotlib.pyplot as plt

from torch import autocast
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.training.nnUNetTrainer.variants.WandbWrapper import WandbWrapper
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


class nnUNetTrainerCustom(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        # run og nnUNetTrainer initialization as intended
        super().__init__(plans, configuration, fold, dataset_json, device = device)

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













