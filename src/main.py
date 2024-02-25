import os
from pathlib import Path
import yaml
import sys
sys.path.append('../')
import shutil

import hydra
import torch
import wandb
from colorama import Fore
from jaxtyping import install_import_hook
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger


# Configure beartype and jaxtyping.
with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from src.config import load_typed_root_config
    from src.dataset.data_module import DataModule
    from src.global_cfg import set_cfg
    from src.loss import get_losses
    from src.misc.LocalLogger import LocalLogger
    from src.misc.step_tracker import StepTracker
    from src.misc.wandb_tools import update_checkpoint_path
    from src.model.decoder import get_decoder
    from src.model.encoder import get_encoder
    from src.model.model_wrapper import ModelWrapper
    from src.model.decoder.decoder_latent import DecoderLatent, DecoderLatentTiny


def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="main",
)
def train(cfg_dict: DictConfig):
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)
    torch.manual_seed(cfg_dict.seed)

    # Set up the output directory.
    output_dir = Path(
        hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
    )

    print(cyan(f"Saving outputs to {output_dir}."))
    # latest_run = output_dir.parents[1] / "latest-run"
    # os.system(f"rm {latest_run}")
    # os.system(f"ln -s {output_dir} {latest_run}")

    # Set up logging with wandb.
    callbacks = []
    if cfg_dict.wandb.mode != "disabled":
        logger = WandbLogger(
            project=cfg_dict.wandb.project,
            mode=cfg_dict.wandb.mode,
            name=f"{cfg_dict.wandb.name} ({output_dir.parent.name}/{output_dir.name})",
            tags=cfg_dict.wandb.get("tags", None),
            log_model="all",
            save_dir=output_dir,
            config=OmegaConf.to_container(cfg_dict),
        )
        callbacks.append(LearningRateMonitor("step", True))

        # On rank != 0, wandb.run is None.
        if wandb.run is not None:
            wandb.run.log_code("src")
    else:
        logger = LocalLogger()
        
        
    # Set up checkpointing.
    callbacks.append(
        ModelCheckpoint(
            output_dir / "checkpoints",
            every_n_train_steps=cfg.checkpointing.every_n_train_steps,
            save_top_k=cfg.checkpointing.save_top_k,
        )
    )

    # Prepare the checkpoint for loading.
    checkpoint_path = update_checkpoint_path(cfg.checkpointing.load, cfg.wandb)

    # This allows the current step to be shared with the data loader processes.
    step_tracker = StepTracker()

    trainer = Trainer(
        max_epochs=-1,
        logger=logger,
        devices=cfg.trainer.devices,
        num_nodes=cfg.trainer.num_nodes,
        accelerator="gpu",
        strategy="ddp_find_unused_parameters_true"
        if torch.cuda.device_count() > 1
        else "auto",
        callbacks=callbacks,
        val_check_interval=cfg.trainer.val_check_interval,
        enable_progress_bar=False,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        max_steps=cfg.trainer.max_steps,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        # profiler="advanced", # profiler, "simple", "advanced", "pytorch"
        # precision="bf16-mixed",
    )

    encoder, encoder_visualizer = get_encoder(cfg.model.encoder)
    
    decoder_latent_type = cfg.decoder_latent_type
    
    if decoder_latent_type is None:
        decoder_latent = None

    elif decoder_latent_type == "medium":
        config_path = "config/model/decoder/latent/config_vq-f4-noattn.yaml"
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        decoder_latent = DecoderLatent(
            **config['model']['params']['ddconfig'], **config['model']['params'])
    
    elif decoder_latent_type == "tiny":
        config_path = "config/model/decoder/latent/latent_tiny.yaml"
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        decoder_latent = DecoderLatentTiny(d_in=config['d_in'], d_out=config['d_out'])
    
    else:
        raise ValueError(f"Unknown decoder_latent_type: {decoder_latent_type}")


    model_wrapper = ModelWrapper(
        cfg.optimizer,
        cfg.test,
        cfg.train,
        encoder,
        encoder_visualizer,
        get_decoder(cfg.model.decoder, cfg.dataset),
        get_losses(cfg.loss),
        step_tracker, 
        decoder_latent=decoder_latent,
    )
    

    if cfg.load_pretrained_encoder is not None:
        if cfg.load_pretrained_encoder == 'encoder_latent':

            if encoder.encoder_latent_type  == 'tiny':
                encoder_init_path = "pretrained_models/encoder_latent_tiny.pth"
                encoder_init = torch.load(encoder_init_path)
                model_wrapper.encoder.load_state_dict(encoder_init)
            else:
                raise ValueError(f"Unknown encoder_latent_type or not implemented yet: {encoder.encoder_latent_type }")

        elif cfg.load_pretrained_encoder == 'encoder_and_encoder_latent':
            raise ValueError(f"Not implemented yet: {cfg.load_pretrained_encoder}")
        
        elif cfg.load_pretrained_encoder == 'encoder':
            raise ValueError(f"Not implemented yet: {cfg.load_pretrained_encoder}")
        
        else:
            raise ValueError(f"Unknown load_pretrained_encoder: {cfg.load_pretrained_encoder}")
    

    if cfg.load_pretrained_latent_decoder:
        if cfg.decoder_latent_type == 'tiny':
            decoder_latent_init_path = "pretrained_models/decoder_latent_tiny.pth"
            decoder_latent_init = torch.load(decoder_latent_init_path)
            model_wrapper.decoder_latent.load_state_dict(decoder_latent_init)
        else:
            raise ValueError(f"Unknown decoder_latent_type or not implemented yet: {decoder_latent_type}")

    # breakpoint()
    # len(model_wrapper.encoder.state_dict().keys())  # 632
    # len(model_wrapper.decoder.state_dict().keys())  # 0
    # len(model_wrapper.decoder_latent.state_dict().keys()) # 48
    # torch.save(model_wrapper.encoder.state_dict(), 'encoder_init.pth')
    # torch.save(model_wrapper.decoder_latent.state_dict(), 'decoder_latent_init.pth')


    if cfg.freeze_latent and decoder_latent_type is not None:
        print('==> Freeze latent encoder and decoder')
        # model_wrapper.decoder_latent.state_dict().keys()
        # model_wrapper.decoder_latent.state_dict()['quantize.embedding.weight'].requires_grad
        for param in model_wrapper.encoder.encoder_latent.parameters():
            param.requires_grad = False
        for param in model_wrapper.decoder_latent.parameters():
            param.requires_grad = False
       
    
    data_module = DataModule(cfg.dataset, cfg.data_loader, step_tracker)
    
    if cfg.mode == "train":
        trainer.fit(model_wrapper, datamodule=data_module,
                    ckpt_path=checkpoint_path,
                    )
    else:
        trainer.test(
            model_wrapper,
            datamodule=data_module,
            ckpt_path=checkpoint_path,
        )


if __name__ == "__main__":
    train()
