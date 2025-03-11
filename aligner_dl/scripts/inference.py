import torch
import yaml
from torch.utils.data import DataLoader
import argparse
import lightning as L
from models.utils.collate import custom_collate_fn
from models.utils.misc import build_object


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

    # Load configuration
    with open(args.config) as f:
        config = yaml.safe_load(f)

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

    device = args.device or ('gpu' if torch.cuda.is_available() else 'cpu')

    # Setup trainer
    trainer = L.Trainer(accelerator=device)

    trainer.validate(model, val_loader, ckpt_path=config['trainer'].pop('ckpt_path', None))

if __name__ == "__main__":
    main()
