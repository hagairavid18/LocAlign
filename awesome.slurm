#! /bin/sh

#SBATCH --job-name=softbbs
#SBATCH --output=/home/iscb/wolfson/hagairavid/test_print9.out # redirect stdout
#SBATCH --error=/home/iscb/wolfson/hagairavid/awesome.err # redirect stderr
#SBATCH --partition=gpu-h100-killable # partition
#SBATCH --time=400 # max time (minutes)
#SBATCH --signal=USR1@120 # how to end job when time’s up
#SBATCH --nodes=1 # number of machines
#SBATCH --ntasks=1 # number of processes
#SBATCH --mem=50000 # CPU memory (MB)
#SBATCH --cpus-per-task=16 # CPU cores per process
#SBATCH --gpus=1 # GPUs in total


# /home/iscb/wolfson/hagairavid/miniforge3/envs/aligner/bin/python lightning_trainer.py --config configs/soft_bb.yaml
srun /home/iscb/wolfson/hagairavid/miniforge3/envs/localign_infer3/bin/python aligner_dl/trainers/lightning_trainer.py --config aligner_dl/configs/loc_align.yaml
# /home/iscb/wolfson/hagairavid/miniforge3/envs/aligner_dl2/bin/python aligner_dl/trainers/lightning_trainer.py --config aligner_dl/configs/tm_align.yaml
