# Taken from LuxImagingAI/nnUNet-wandb

import wandb
from datetime import datetime


class WandbWrapper():
    def __init__(self, use_wandb=True, config=None):
        self.use_wandb = use_wandb

        if self.use_wandb:
            self.config = config

    def init(self):
        if self.use_wandb:
            nnow = datetime.now()
            ndate_str = nnow.strftime("%Y.%m.%d")
            ntime_str = nnow.strftime("%H:%M:%S")

            wandb_name = f"{self.config['project_name']}_{self.config['architecture']}_fold_{self.config['fold']}_{ndate_str}_{ntime_str}"
            return wandb.init(project=self.config['project_name'], name=wandb_name, config=self.config)

    def log(self, *args, **kwargs):
        if self.use_wandb:
            return wandb.log(*args, **kwargs)

    def Image(self, *args, **kwargs):
        if self.use_wandb:
            return wandb.Image(*args, **kwargs)

    def finish(self):
        if self.use_wandb:
            return wandb.finish()