import lightning as L
from torchvision.transforms import v2


class ProgressiveResize(L.Callback):
    def __init__(self):
        self.resize_schedule = {
            0: {"batch_size": 16, "num_workers": 16, "size": 64},
            2: {"batch_size": 4, "num_workers": 4, "size": 128},
            4: {"batch_size": 1, "num_workers": 1, "size": 256},
        }

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch in self.resize_schedule:
            print(f"Rescheduling to: {params['size']}-{params['batch_size']}-{params['num_workers']}")
            params = self.resize_schedule[trainer.current_epoch]

            trainer.datamodule.size = params["size"]
            trainer.datamodule.batch_size = params["batch_size"]
            trainer.datamodule.num_workers = params["num_workers"]

            trainer.datamodule.setup(stage="fit")

    def on_validation_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch in self.resize_schedule:
            params = self.resize_schedule[trainer.current_epoch]

            trainer.datamodule.size = params["size"]
            trainer.datamodule.batch_size = params["batch_size"]
            trainer.datamodule.num_workers = params["num_workers"]

            trainer.datamodule.setup(stage="validate")
