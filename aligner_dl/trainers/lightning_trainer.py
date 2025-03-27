import torch
import logging
import yaml
from torch.utils.data import DataLoader, RandomSampler
from pytorch_lightning.loggers import CometLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from datetime import datetime
import os
import argparse
import lightning as L
from models.utils.collate import custom_collate_fn
from models.utils.misc import build_object, flatten_dict


# Argument parser setup
def parse_args():
    parser = argparse.ArgumentParser(description="Train a model with configurable arguments.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--log_dir", type=str, default="logs/learnable_softbbs", help="Directory to save logs.")
    parser.add_argument("--seed", type=int, default=41, help="Random seed for reproducibility.")
    parser.add_argument("--device", type=str, choices=["gpu", "cpu"], default=None, help="Device to use for training.")
    parser.add_argument("--validate_only", action="store_true", help="Run only validation.")
    return parser.parse_args()

# Main script
def main():
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)

    # Configure logging
    start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(filename=os.path.join(log_dir, start_time + ".log"), level=logging.INFO, format='%(message)s')

    logger = logging.getLogger(__name__)

    # Load configuration
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Setup datasets and data loaders
    train_dataset = build_object(config['dataset']['train'], 'datasets')
    valid_dataset = build_object(config['dataset']['validation'], 'datasets')
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['dataloader']['train_batch_size'], 
        collate_fn=custom_collate_fn, 
        num_workers=config['dataloader']['n_workers'],
        shuffle=True,
        pin_memory=True,
        # sampler=RandomSampler(train_dataset, num_samples=10000, replacement=True)
    )
    val_loader = DataLoader(
        valid_dataset, 
        batch_size=config['dataloader']['valid_batch_size'], 
        collate_fn=custom_collate_fn, 
        num_workers=config['dataloader']['n_workers'],
        pin_memory=True
    )

    # Build model
    model = build_object(config['model'], 'models')

    # Setup logging with Comet
    log_exp = "delete" not in config['trainer']['exp_name']
    comet_logger = None
    if log_exp:
        comet_logger = CometLogger(
            api_key="9ydBzigeK75Z6RhAiX63xGdsg",
            workspace="hagairavid18",
            project_name="pocket_aligner",
            experiment_name=config['trainer']['exp_name']
        )
        comet_logger.experiment.add_tags(config['trainer'].get('tags', []))
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/{config['trainer']['exp_name']}",
        save_top_k=-1,
        every_n_epochs=1,
        verbose=True,
        save_on_train_epoch_end=True
    )

    device = args.device or ('gpu' if torch.cuda.is_available() else 'cpu')

    # Setup trainer
    trainer = L.Trainer(
        logger=comet_logger if log_exp else None,
        max_epochs=config['trainer']['max_epochs'],
        check_val_every_n_epoch=config['trainer']['check_val_every_n_epoch'],
        callbacks=[checkpoint_callback],
        gradient_clip_val=config['trainer'].pop('gradient_clipping', None),
        log_every_n_steps=100,
        accelerator=device,
        profiler="advanced" if config['trainer'].get('profiler', False) else None,
        # detect_anomaly=True,
    )

    # Log hyperparameters
    if trainer.logger:
        trainer.logger.log_hyperparams(flatten_dict(config))

    # Train or validate
    if config['trainer']['validate_only']:
        trainer.validate(model, val_loader, ckpt_path=config['trainer']['ckpt_path'])
    else:
        trainer.fit(model, train_loader, val_dataloaders=val_loader, ckpt_path=config['trainer'].pop('ckpt_path', None))

if __name__ == "__main__":
    main()
