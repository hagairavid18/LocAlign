#! /bin/sh

#SBATCH --job-name=baseline
#SBATCH --output=/home/iscb/wolfson/hagairavid/baseline.out # redirect stdout
#SBATCH --error=/home/iscb/wolfson/hagairavid/baseline.err # redirect stderr
#SBATCH --partition=killable
#SBATCH --time=400 # max time (minutes)
#SBATCH --signal=USR1@120 # how to end job when time’s up
#SBATCH --nodes=1 # number of machines
#SBATCH --ntasks=1 # number of processes
#SBATCH --mem=50000 # CPU memory (MB)
#SBATCH --cpus-per-task=16 # CPU cores per process
#SBATCH --gpus=1 # GPUs in total


# /home/iscb/wolfson/hagairavid/miniforge3/envs/aligner/bin/python lightning_trainer.py --config configs/soft_bb.yaml
/home/iscb/wolfson/hagairavid/miniforge3/envs/miner/bin/python miners/scripts/baseline.py --config miners/configs/baseline_config.json --save_path datasets/csv_files/homology_0507_05/val_baseline_softalign.csv
# /home/iscb/wolfson/hagairavid/miniforge3/envs/aligner/bin/python models/learnable_soft_bb.py