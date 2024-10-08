#!/bin/bash
#SBATCH --job-name=acid_tiny_rnd_6l_ggs256                   # name of job
##SBATCH -C v100-32g 							   # reserving 16 GB GPUs only if commented
##SBATCH --partition=gpu_p2                        # uncomment for gpu_p2 partition gpu_p2
##SBATCH -C a100                                   # uncomment for partition a100
##SBATCH --ntasks=4					 			   # total number of processes (= number of GPUs here)
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=1                                  # nb reserved nodes
#SBATCH --gres=gpu:4                 			   # number of GPUs
#SBATCH --cpus-per-task=10           			   # number of cores per task (1/4 of the 4-GPUs node)
# /!\ Caution, "multithread" in Slurm vocabulary refers to hyperthreading.
#SBATCH --hint=nomultithread         			   # hyperthreading is deactivated
#SBATCH --time=20:00:00             			   # maximum execution time requested (HH:MM:SS)
##SBATCH --time=00:10:00             			   # maximum execution time requested (HH:MM:SS)
#SBATCH --output=slurm_logs/acid_tiny_rnd_6l_ggs256_%j.output   # name of output file
#SBATCH --error=slurm_logs/acid_tiny_rnd_6l_ggs256_%j.error     # name of error file (here, in common with the output file)
##SBATCH --qos=qos_gpu-t4                          # for running (max 100h)
#SBATCH --qos=qos_gpu-t3                           # for running (max 20h)
##SBATCH --qos=qos_gpu-dev                          # for verifying that the code is running.

EXP_NAME="acid_tiny_rnd_6l_ggs256"
RUN_DIR="./outputs/${EXP_NAME}"

# Cleans out the modules loaded in interactive and inherited by default
module purge

# Loading of modules
module load python/3.11.5
module load cuda/12.1.0
conda activate psplat

# Echo of launched commands
set -x

# Code execution
srun python3 -m src.main +experiment=acid exp_name=${EXP_NAME} hydra.run.dir=${RUN_DIR} decoder_latent_dim=6 model.encoder.d_latent=6 model.decoder.d_latent=6 trainer.devices=4 trainer.num_nodes=1 data_loader.train.batch_size=1 wandb.mode=offline wandb.tags=[acid,256x256,d_latent,ggs256] loss.lpips.apply_after_step=40000 trainer.max_steps=80000 checkpointing.every_n_train_steps=10000
