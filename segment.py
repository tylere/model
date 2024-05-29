"""
Command line interface to run the neural network model!

From the project root directory, do:

    python segment.py fit

References:
- https://lightning.ai/docs/pytorch/2.1.0/cli/lightning_cli.html
- https://pytorch-lightning.medium.com/introducing-lightningcli-v2-supercharge-your-training-c070d43c7dd6
"""

from lightning.pytorch.cli import LightningCLI

from finetune.segment.chesapeake_datamodule import ChesapeakeDataModule  # noqa: F401
from finetune.segment.chesapeake_model import ChesapeakeSegmentor  # noqa: F401


# %%
def cli_main():
    """
    Command-line inteface to run ClayMAE with ClayDataModule.
    """
    cli = LightningCLI(ChesapeakeSegmentor, ChesapeakeDataModule)
    return cli


# %%
if __name__ == "__main__":
    cli_main()

    print("Done!")
