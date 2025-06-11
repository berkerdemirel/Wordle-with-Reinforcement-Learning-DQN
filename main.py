# train.py
import hydra
from omegaconf import DictConfig, OmegaConf
from pl_wordle import WordleLightning
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger


@hydra.main(config_path=".", config_name="config")
def main(cfg: DictConfig):
    seed_everything(cfg.seed, workers=True)

    model = WordleLightning(cfg)
    flat_cfg = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
    wandb_logger = WandbLogger(
        project="wordle-universal",
        config=flat_cfg,  # ← JSON-friendly
    )

    trainer = Trainer(
        max_steps=cfg.train.num_env_steps,
        logger=wandb_logger,
        log_every_n_steps=cfg.train.log_every,
        deterministic=True,
    )
    trainer.fit(model)


if __name__ == "__main__":
    main()
