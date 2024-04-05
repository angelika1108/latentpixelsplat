#!/bin/bash
#SBATCH --job-name=test_acid_tiny_enc_lat_ed_80k  # name of job
##SBATCH -C v100-32g 							   # reserving 16 GB GPUs only if commented
##SBATCH --partition=gpu_p2                        # uncomment for gpu_p2 partition gpu_p2
##SBATCH -C a100                                   # uncomment for partition a100
#SBATCH --ntasks=1					 			   # total number of processes (= number of GPUs here)
##SBATCH --ntasks-per-node=1
#SBATCH --nodes=1                                  # reserving 1 node
#SBATCH --gres=gpu:1                 			   # number of GPUs
#SBATCH --cpus-per-task=10           			   # number of cores per task (1/4 of the 4-GPUs node)
# /!\ Caution, "multithread" in Slurm vocabulary refers to hyperthreading.
#SBATCH --hint=nomultithread         			   # hyperthreading is deactivated
#SBATCH --time=00:15:00             			   # maximum execution time requested (HH:MM:SS)
##SBATCH --time=00:10:00             			   # maximum execution time requested (HH:MM:SS)
#SBATCH --output=slurm_logs/test_acid_tiny_enc_lat_ed_80k_%j.output   # name of output file
#SBATCH --error=slurm_logs/test_acid_tiny_enc_lat_ed_80k_%j.error     # name of error file (here, in common with the output file)
##SBATCH --qos=qos_gpu-t4                          # for running (max 100h)
#SBATCH --qos=qos_gpu-t3                          # for running (max 20h)
##SBATCH --qos=qos_gpu-dev                          # for veryfuing that the code is running (max 10min)


EXP_NAME ="acid_tiny_enc_lat_ed_80k"
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
# Test
python3 -m src.main +experiment=acid mode=test exp_name=${EXP_NAME} hydra.run.dir=${RUN_DIR} dataset/view_sampler=evaluation dataset.view_sampler.index_path=assets/evaluation_index_acid.json test.output_path=outputs/${EXP_NAME}/test checkpointing.load=outputs/${EXP_NAME}/checkpoints/epoch7_step80000.ckpt load_pretrained_encoder=encoder_and_encoder_latent load_pretrained_latent_decoder=true
# Metrics
python3 -m src.scripts.compute_metrics +experiment=acid +evaluation=acid output_metrics_path=outputs/${EXP_NAME}/test/acid/evaluation_metrics.json evaluation.methods.0.path=outputs/${EXP_NAME}/test/acid
