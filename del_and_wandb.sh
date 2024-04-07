
module load python/3.11.5
module load cuda/12.1.0
conda activate psplat

cd /gpfswork/rech/dpe/uwp78ya/latentpixelsplat/outputs


# cd acid_tiny_rnd_4g
# cd checkpoints
# rm 'epoch=0-step=10000.ckpt' 'epoch=1-step=20000.ckpt' 'epoch=2-step=30000.ckpt' 'epoch=3-step=40000.ckpt'  'epoch=4-step=50000.ckpt' 'epoch=5-step=60000.ckpt'  'epoch=6-step=70000.ckpt'
# mv 'epoch=7-step=80000.ckpt' epoch7_step80000.ckpt
# cd ../../

# cd acid_tiny_rnd_1g_1ms
# cd checkpoints
# rm 'epoch=0-step=10000.ckpt' 'epoch=1-step=20000.ckpt' 'epoch=2-step=30000.ckpt' 'epoch=3-step=40000.ckpt'  'epoch=4-step=50000.ckpt' 'epoch=5-step=60000.ckpt'  'epoch=6-step=70000.ckpt'
# mv 'epoch=7-step=80000.ckpt' epoch7_step80000.ckpt
# cd ../../

# cd acid_tiny_rnd_1g_64ms
# cd checkpoints
# rm 'epoch=0-step=10000.ckpt' 'epoch=1-step=20000.ckpt' 'epoch=2-step=30000.ckpt' 'epoch=3-step=40000.ckpt'  'epoch=4-step=50000.ckpt' 'epoch=5-step=60000.ckpt'  'epoch=6-step=70000.ckpt'
# mv 'epoch=7-step=80000.ckpt' epoch7_step80000.ckpt
# cd ../../


# cd acid_tiny_rnd_4g
# wandb sync --id=0407_3 --sync-all
# cd ../

# cd acid_tiny_rnd_1g_1ms
# wandb sync --id=0407_4 --sync-all
# cd ../

# cd acid_tiny_rnd_1g_64ms
# wandb sync --id=0407_5 --sync-all
# cd ../



