import glob
import torch
import logging
import yaml
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import CometLogger
from comet_ml import API, ExistingExperiment
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

    # Load configuration
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Setup datasets and data loaders
    if 'train' in config['dataset']:
        train_dataset = build_object(config['dataset']['train'], 'datasets')
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config['dataloader']['train_batch_size'], 
            collate_fn=custom_collate_fn, 
            num_workers=config['dataloader']['n_workers'],
            shuffle=True,
            pin_memory=True,
            # sampler=RandomSampler(train_dataset, num_samples=10000, replacement=True)
        )
    
    valid_dataset = build_object(config['dataset']['validation'], 'datasets')
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
    if log_exp:
        api = API(api_key="9ydBzigeK75Z6RhAiX63xGdsg")  # Or omit if using env var

        workspace = "hagairavid18"
        project = "pocket-aligner"
        experiment_name = config['trainer']['exp_name']

        # Search for existing experiment by name
        experiment_id = None
        for exp in api.get_experiments(workspace, project):
            if exp.name == experiment_name:
                experiment_id = exp.id
                print(f"Resuming existing experiment: {experiment_name} (ID: {experiment_id})")
                break

        found_checkpoint = False
        ckpt_dir = os.path.join("checkpoints", experiment_name)
        if os.path.isdir(ckpt_dir):
            ckpts = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
            if ckpts:
                # Get latest by modification time
                resume_ckpt = max(ckpts, key=os.path.getmtime)
                config['trainer']['ckpt_path'] = resume_ckpt
                print(f"✅ Resuming from checkpoint: {resume_ckpt}")
                found_checkpoint = True
            else:
                print(f"⚠️ No .ckpt files found in {ckpt_dir}")
        else:
            print(f"⚠️ Checkpoint directory not found: {ckpt_dir}")
        if experiment_id  and found_checkpoint:
            # Create a dummy logger and replace its experiment with ExistingExperiment
            comet_logger = CometLogger(
                api_key="9ydBzigeK75Z6RhAiX63xGdsg",
                workspace=workspace,
                project_name=project,
                experiment_name=experiment_name,
            )
            # Manually replace internal experiment object
            comet_logger._experiment = ExistingExperiment(
                api_key="9ydBzigeK75Z6RhAiX63xGdsg",
                previous_experiment=experiment_id,
                workspace=workspace,
                project_name=project,
            )

        else:
            # Create a new experiment via CometLogger
            comet_logger = CometLogger(
                api_key="9ydBzigeK75Z6RhAiX63xGdsg",
                workspace=workspace,
                project_name=project,
                experiment_name=experiment_name,
            )
            

    # Add tags
        comet_logger.experiment.add_tags(config['trainer'].get('tags', []))
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/{config['trainer']['exp_name']}",
        save_top_k=-1,
        every_n_epochs=1,
        verbose=True,
        save_on_train_epoch_end=True
    )
    # Ensure checkpoint directory exists
    os.makedirs(f"checkpoints/{config['trainer']['exp_name']}", exist_ok=True)

    # Save model config alongside checkpoints as YAML
    model_config_path = os.path.join(f"checkpoints/{config['trainer']['exp_name']}", "model_config.yaml")
    with open(model_config_path, 'w') as f:
        yaml.dump(config['model'], f)

    print(f"Saved model config to {model_config_path}")

    device = args.device or ('gpu' if torch.cuda.is_available() else 'cpu')

    # Setup trainer
    trainer = L.Trainer(
        logger=comet_logger if log_exp else None,
        max_epochs=config['trainer']['max_epochs'],
        check_val_every_n_epoch=config['trainer'].pop('check_val_every_n_epoch', 1),
        callbacks=[checkpoint_callback],
        gradient_clip_val=config['trainer'].pop('gradient_clipping', None),
        log_every_n_steps=100,
        accelerator=device,
        
        # precision="bf16-mixed" if device == "gpu" else 32,
        profiler="advanced" if config['trainer'].get('profiler', False) else None,
        # detect_anomaly=True,
    )

    # Log hyperparameters
    if trainer.logger:
        trainer.logger.log_hyperparams(flatten_dict(config))

    # Train or validate
    if config['trainer']['validate_only']:
        trainer.validate(model, val_loader, ckpt_path=config['trainer'].get('ckpt_path', None))
    else:
        trainer.fit(model, train_loader, val_dataloaders=val_loader, ckpt_path=config['trainer'].pop('ckpt_path', None))

if __name__ == "__main__":
    main()
