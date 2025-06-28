from omegaconf import OmegaConf
from model_pl import VideoClassificationPL
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

import torch
import lightning as L

def main(train_config, model_config, ckpt_path=None, wandb_id=None):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    model_config = OmegaConf.load(model_config)
    train_config = OmegaConf.load(train_config)

    model = VideoClassificationPL(model_config)

    wandb_logger = WandbLogger(
        entity="translomo", project="DeepLearningTeamProject", id=wandb_id
    )
    wandb_logger.watch(model)

    wandb_id = wandb_logger.experiment.id

    trainer = L.Trainer(
        **train_config,
        callbacks=[
            ModelCheckpoint(
                monitor="val/loss",
                dirpath="checkpoints/",
                filename=f"{wandb_id}_" + "{epoch}_{val/loss:.2f}_{val/acc:.2f}",
                save_top_k=5,
            )
        ],
        logger=wandb_logger,
    )
    trainer.fit(model, ckpt_path=ckpt_path)

if __name__ == "__main__":
    import fire
    fire.Fire(main)
