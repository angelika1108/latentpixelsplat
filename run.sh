
# Test PixelSplat on ACID with pre-trained weights
python3 -m src.main +experiment=acid mode=test dataset/view_sampler=evaluation checkpointing.load=pretrained_models/acid.ckpt dataset.view_sampler.index_path=assets/evaluation_index_acid.json

# python3 -m src.main \
# +experiment=acid \
# mode=test \
# dataset/view_sampler=evaluation \
# checkpointing.load=pretrained_models/acid.ckpt \
# dataset.view_sampler.index_path=assets/evaluation_index_acid.json


# Test on ACID with latent encoder and decoder with pre-trained weights
python3 -m src.main +experiment=acid mode=test wandb.mode=offline dataset/view_sampler=evaluation checkpointing.load=pretrained_models/acid_latent_d3_f4_noattn.ckpt dataset.view_sampler.index_path=assets/evaluation_index_acid.json

# tiny_rnd_bs4_20000
python3 -m src.main +experiment=acid mode=test dataset/view_sampler=evaluation checkpointing.load=outputs/2024-02-22/11-51-54_tiny_rnd_bs4/checkpoints/epoch_7-step_20000.ckpt dataset.view_sampler.index_path=assets/evaluation_index_acid.json


# Test without checkpoint
python3 -m src.main +experiment=acid mode=test wandb.mode=disabled dataset/view_sampler=evaluation dataset.view_sampler.index_path=assets/evaluation_index_acid.json


# Train
# trainer.val_check_interval=30
# trainer.log_every_n_steps=30 ####################
# optimizer.warm_up_steps=1000
# checkpointing.load=pretrained_models/acid_latent_d3_f4_noattn.ckpt
# checkpointing.every_n_train_steps=2000
# checkpointing.save_top_k=3
# checkpointing.every_n_epochs=5000


# Train without checkpoint
python3 -m src.main +experiment=acid hydra.run.dir='outputs/exp' wandb.mode=disabled data_loader.train.batch_size=1 trainer.val_check_interval=30 optimizer.warm_up_steps=1000 checkpointing.every_n_train_steps=2000

# Train with checkpoint
python3 -m src.main +experiment=acid data_loader.train.batch_size=1 wandb.mode=disabled trainer.val_check_interval=30 optimizer.warm_up_steps=1000 
checkpointing.load=pretrained_models/acid_latent_d3_f4_noattn.ckpt 
checkpointing.every_n_train_steps=2000 trainer.max_steps=350000




# Tiny random
python3 -m src.main +experiment=acid exp_name='exp' hydra.run.dir='outputs/exp' data_loader.train.batch_size=1 checkpointing.every_n_train_steps=10000 trainer.val_check_interval=30 optimizer.warm_up_steps=1000
# model.encoder.epipolar_transformer.upscale=4

# Tiny load encoder and latent encoder and decoder
python3 -m src.main +experiment=acid exp_name='acid_tiny_enc_lat_ed' hydra.run.dir='outputs/acid_tiny_enc_lat_ed' data_loader.train.batch_size=1 load_pretrained_encoder=encoder_and_encoder_latent load_pretrained_latent_decoder=true checkpointing.every_n_train_steps=10000 trainer.val_check_interval=10 optimizer.warm_up_steps=1000

# Tiny load encoder
python3 -m src.main +experiment=acid exp_name='acid_tiny_enc' hydra.run.dir='outputs/acid_tiny_enc' data_loader.train.batch_size=1 load_pretrained_encoder=encoder load_pretrained_latent_decoder=false checkpointing.every_n_train_steps=10000 trainer.val_check_interval=10 optimizer.warm_up_steps=1000

# Tiny load latent encoder and decoder
python3 -m src.main +experiment=acid exp_name='acid_tiny_lat_ed' hydra.run.dir='outputs/acid_tiny_lat_ed' data_loader.train.batch_size=1 load_pretrained_encoder=encoder_latent load_pretrained_latent_decoder=true checkpointing.every_n_train_steps=10000 trainer.val_check_interval=30 optimizer.warm_up_steps=1000

# Tiny load latent encoder and decoder + freeze
python3 -m src.main +experiment=acid exp_name='acid_tiny_lat_ed_freeze' hydra.run.dir='outputs/acid_tiny_lat_ed_freeze' data_loader.train.batch_size=1 freeze_latent=true load_pretrained_encoder=encoder_latent load_pretrained_latent_decoder=true checkpointing.every_n_train_steps=10000 trainer.val_check_interval=30 optimizer.warm_up_steps=1000


# Tiny random lpips from 40k, max steps 80k
python3 -m src.main +experiment=acid exp_name='exp' hydra.run.dir='outputs/exp' loss.lpips.apply_after_step=40000 trainer.max_steps=80000 data_loader.train.batch_size=1 checkpointing.every_n_train_steps=10000 trainer.val_check_interval=30 optimizer.warm_up_steps=1000




# Tiny random no latent encoder
python3 -m src.main +experiment=acid exp_name='exp' hydra.run.dir='outputs/exp' model.encoder.encoder_latent_type=null model.encoder.epipolar_transformer.upscale=4 data_loader.train.batch_size=1 checkpointing.every_n_train_steps=10000 trainer.val_check_interval=30 optimizer.warm_up_steps=1000
# model.encoder.encoder_latent_type=null
# model.encoder.epipolar_transformer.upscale=4



# pixelsplat load encoder 
python3 -m src.main +experiment=acid exp_name='acid_enc' hydra.run.dir='outputs/acid_enc' data_loader.train.batch_size=1 load_pretrained_encoder=encoder checkpointing.every_n_train_steps=10000 trainer.val_check_interval=10 optimizer.warm_up_steps=1000

