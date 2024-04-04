#!/bin/bash
#SBATCH --job-name=debug                           # name of job
##SBATCH -C v100-32g 							   # reserving 16 GB GPUs only if commented
##SBATCH --partition=gpu_p2                        # uncomment for partition gpu_p2
##SBATCH -C a100                                   # uncomment for partition a100
#SBATCH --ntasks=1					 			   # total number of processes (= number of GPUs here)
##SBATCH --ntasks-per-node=1
#SBATCH --nodes=1                                  # reserving 1 node
#SBATCH --gres=gpu:1                 			   # number of GPUs
#SBATCH --cpus-per-task=10           			   # number of cores per task (1/4 of the 4-GPUs node)
# /!\ Caution, "multithread" in Slurm vocabulary refers to hyperthreading.
#SBATCH --hint=nomultithread         			   # hyperthreading is deactivated
##SBATCH --time=20:00:00             			   # maximum execution time requested (HH:MM:SS)
#SBATCH --time=00:10:00             			   # maximum execution time requested (HH:MM:SS)
#SBATCH --output=slurm_logs/debug_%j.output        # name of output file
#SBATCH --error=slurm_logs/debug_%j.error          # name of error file (here, in common with the output file)
##SBATCH --qos=qos_gpu-t4                          # for running (max 100h)
##SBATCH --qos=qos_gpu-t3                          # for running (max 20h)
#SBATCH --qos=qos_gpu-dev                          # for verifying that the code is running.


# Cleans out the modules loaded in interactive and inherited by default
module purge

# Loading of modules
module load python/3.11.5
module load cuda/12.1.0
conda activate psplat

# Echo of launched commands
set -x

# Code execution
python3 -m src.main +experiment=acid data_loader.train.batch_size=1 checkpointing.every_n_train_steps=2000 trainer.max_steps=10000
