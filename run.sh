
### Test

### Breaking down the command:
### Possible options
# python3 -m src.main 
# +experiment=acid 
# mode=test 
# exp_name='exp' 
# hydra.run.dir='outputs/exp' 
# dataset/view_sampler=evaluation checkpointing.load=pretrained_models/acid.ckpt 
# dataset.view_sampler.index_path=assets/evaluation_index_acid.json 
# model.encoder.epipolar_transformer.upscale=4 
# test.output_path=outputs/test/acid_pretrained_pixelsplat


# Test on ACID with latent encoder and decoder with pre-trained weights
python3 -m src.main +experiment=acid mode=test exp_name='exp' hydra.run.dir='outputs/exp' wandb.mode=disabled dataset/view_sampler=evaluation model.encoder.epipolar_transformer.upscale=4 checkpointing.load=pretrained_models/acid_latent_d3_f4_noattn.ckpt dataset.view_sampler.index_path=assets/evaluation_index_acid.json

# tiny_rnd_bs4_20000
python3 -m src.main +experiment=acid mode=test dataset/view_sampler=evaluation checkpointing.load=outputs/2024-02-22/11-51-54_tiny_rnd_bs4/checkpoints/epoch_7-step_20000.ckpt dataset.view_sampler.index_path=assets/evaluation_index_acid.json test.output_path=outputs/test/tiny_rnd_bs4_20000

# Test without checkpoint
python3 -m src.main +experiment=acid mode=test exp_name='exp' hydra.run.dir='outputs/exp' wandb.mode=disabled dataset/view_sampler=evaluation dataset.view_sampler.index_path=assets/evaluation_index_acid.json test.output_path=outputs/test/random

# acid_tiny_enc_lat_ed_80k
python3 -m src.main +experiment=acid mode=test exp_name='acid_tiny_enc_lat_ed_80k' hydra.run.dir='outputs/acid_tiny_enc_lat_ed_80k' dataset/view_sampler=evaluation dataset.view_sampler.index_path=assets/evaluation_index_acid.json test.output_path=outputs/acid_tiny_enc_lat_ed_80k/test checkpointing.load=outputs/acid_tiny_enc_lat_ed_80k/checkpoints/epoch7_step80000.ckpt load_pretrained_encoder=encoder_and_encoder_latent load_pretrained_latent_decoder=true
# test.output_path=outputs/acid_tiny_enc_lat_ed_80k/test 
# checkpointing.load=outputs/acid_tiny_enc_lat_ed_80k/checkpoints/epoch7_step80000.ckpt 
# load_pretrained_encoder=encoder_and_encoder_latent 
# load_pretrained_latent_decoder=true

# acid_tiny_enc_80k
python3 -m src.main +experiment=acid mode=test exp_name='acid_tiny_enc_80k' hydra.run.dir='outputs/acid_tiny_enc_80k' dataset/view_sampler=evaluation dataset.view_sampler.index_path=assets/evaluation_index_acid.json test.output_path=outputs/acid_tiny_enc_80k/test checkpointing.load=outputs/acid_tiny_enc_80k/checkpoints/epoch7_step80000.ckpt load_pretrained_encoder=encoder load_pretrained_latent_decoder=false

# acid_tiny_lat_ed_80k
python3 -m src.main +experiment=acid mode=test exp_name='acid_tiny_lat_ed_80k' hydra.run.dir='outputs/acid_tiny_lat_ed_80k' dataset/view_sampler=evaluation dataset.view_sampler.index_path=assets/evaluation_index_acid.json test.output_path=outputs/acid_tiny_lat_ed_80k/test checkpointing.load=outputs/acid_tiny_lat_ed_80k/checkpoints/epoch7_step80000.ckpt load_pretrained_encoder=encoder_latent load_pretrained_latent_decoder=true

# acid_tiny_rnd_80k
python3 -m src.main +experiment=acid mode=test exp_name='acid_tiny_rnd_80k' hydra.run.dir='outputs/acid_tiny_rnd_80k' dataset/view_sampler=evaluation dataset.view_sampler.index_path=assets/evaluation_index_acid.json test.output_path=outputs/acid_tiny_rnd_80k/test checkpointing.load=outputs/acid_tiny_rnd_80k/checkpoints/epoch7_step80000.ckpt


### Train

# !!! trainer.val_check_interval=30 optimizer.warm_up_steps=1000 options are only added for debug !!!
# !!! These are not added when training for real. !!!

# trainer.val_check_interval=30
# optimizer.warm_up_steps=1000

# model.encoder.epipolar_transformer.upscale=4  ## not used
# wandb.mode=disabled, offline
# checkpointing.load=pretrained_models/acid_latent_d3_f4_noattn.ckpt
# checkpointing.every_n_train_steps=2000
# trainer.max_steps=80000
# loss.lpips.apply_after_step=40000
# exp_name='exp' hydra.run.dir='outputs/exp'

# model.encoder.encoder_latent_type=encoder, encoder_and_encoder_latent, null
# load_pretrained_latent_decoder=true

# model.encoder.gaussian_grid_size=[64,64] 

# Train without checkpoint
python3 -m src.main +experiment=acid exp_name='exp' hydra.run.dir='outputs/exp' model.encoder.gaussian_grid_size=[64,64] data_loader.train.batch_size=1 trainer.val_check_interval=30 optimizer.warm_up_steps=1000 checkpointing.every_n_train_steps=2000

# Train with checkpoint
python3 -m src.main +experiment=acid exp_name='exp' hydra.run.dir='outputs/exp' data_loader.train.batch_size=1 trainer.val_check_interval=30 optimizer.warm_up_steps=1000 checkpointing.load=pretrained_models/acid_latent_d3_f4_noattn.ckpt checkpointing.every_n_train_steps=2000

# Tiny random
python3 -m src.main +experiment=acid exp_name='exp' hydra.run.dir='outputs/exp' data_loader.train.batch_size=1 checkpointing.every_n_train_steps=10000 trainer.val_check_interval=30 optimizer.warm_up_steps=1000

# Tiny load encoder and latent encoder and decoder
python3 -m src.main +experiment=acid exp_name='acid_tiny_enc_lat_ed' hydra.run.dir='outputs/acid_tiny_enc_lat_ed' data_loader.train.batch_size=1 load_pretrained_encoder=encoder_and_encoder_latent load_pretrained_latent_decoder=true checkpointing.every_n_train_steps=10000 trainer.val_check_interval=10 optimizer.warm_up_steps=1000

# Tiny load encoder
python3 -m src.main +experiment=acid exp_name='acid_tiny_enc' hydra.run.dir='outputs/acid_tiny_enc' data_loader.train.batch_size=1 load_pretrained_encoder=encoder load_pretrained_latent_decoder=false checkpointing.every_n_train_steps=10000 trainer.val_check_interval=10 optimizer.warm_up_steps=1000

# Tiny load latent encoder and decoder
python3 -m src.main +experiment=acid exp_name='acid_tiny_lat_ed' hydra.run.dir='outputs/acid_tiny_lat_ed' data_loader.train.batch_size=1 load_pretrained_encoder=encoder_latent load_pretrained_latent_decoder=true checkpointing.every_n_train_steps=10000 trainer.val_check_interval=30 optimizer.warm_up_steps=1000


# Tiny random no latent encoder
python3 -m src.main +experiment=acid exp_name='exp' hydra.run.dir='outputs/exp' model.encoder.encoder_latent_type=null data_loader.train.batch_size=1 checkpointing.every_n_train_steps=10000 trainer.val_check_interval=30 optimizer.warm_up_steps=1000


# gaussians_per_pixel=4
python3 -m src.main +experiment=acid exp_name='exp' hydra.run.dir='outputs/exp' data_loader.train.batch_size=1 checkpointing.every_n_train_steps=10000 trainer.val_check_interval=30 optimizer.warm_up_steps=1000 model.encoder.gaussians_per_pixel=4

# gaussians_per_pixel=1, num_monocular_samples=1
python3 -m src.main +experiment=acid wandb.tags=[acid,256x256,num_monocular_samples] exp_name='exp' hydra.run.dir='outputs/exp' data_loader.train.batch_size=1 checkpointing.every_n_train_steps=10000 trainer.val_check_interval=30 optimizer.warm_up_steps=1000 model.encoder.gaussians_per_pixel=1 model.encoder.num_monocular_samples=1


# latent_channels = 6
python3 -m src.main +experiment=acid exp_name='exp' hydra.run.dir='outputs/exp' decoder_latent_dim=6 model.encoder.d_latent=6 model.decoder.d_latent=6 data_loader.train.batch_size=1 checkpointing.every_n_train_steps=10000 trainer.val_check_interval=30 optimizer.warm_up_steps=1000

# latent_channels = 3
python3 -m src.main +experiment=acid exp_name='exp' hydra.run.dir='outputs/exp' decoder_latent_dim=3 model.encoder.d_latent=3 model.decoder.d_latent=3 data_loader.train.batch_size=1 checkpointing.every_n_train_steps=10000 trainer.val_check_interval=30 optimizer.warm_up_steps=1000


# Train without checkpoint gaussian grid size 256x256
python3 -m src.main +experiment=acid exp_name='exp' hydra.run.dir='outputs/exp' data_loader.train.batch_size=1 trainer.val_check_interval=30 optimizer.warm_up_steps=1000 checkpointing.every_n_train_steps=2000

# Train without checkpoint img size 300x300
python3 -m src.main +experiment=acid exp_name='exp' hydra.run.dir='outputs/exp' data_loader.train.batch_size=1 trainer.val_check_interval=30 optimizer.warm_up_steps=1000 checkpointing.every_n_train_steps=2000 dataset.image_shape=[300,300]


# Train tiny_norm
python3 -m src.main +experiment=acid exp_name='exp' hydra.run.dir='outputs/exp' data_loader.train.batch_size=1 trainer.val_check_interval=30 optimizer.warm_up_steps=1000 checkpointing.every_n_train_steps=2000 decoder_latent_type=tiny_norm model.encoder.encoder_latent_type=tiny_norm

# Train tiny enc tiny_norm dec
python3 -m src.main +experiment=acid exp_name='exp' hydra.run.dir='outputs/exp' data_loader.train.batch_size=1 trainer.val_check_interval=30 optimizer.warm_up_steps=1000 checkpointing.every_n_train_steps=2000 decoder_latent_type=tiny_norm model.encoder.encoder_latent_type=tiny


# decoder_latent_dim=3 model.encoder.d_latent=3 model.decoder.d_latent=3
# Train no latent enc-dec
python3 -m src.main +experiment=acid exp_name='exp' hydra.run.dir='outputs/exp' data_loader.train.batch_size=1 trainer.val_check_interval=30 optimizer.warm_up_steps=1000 checkpointing.every_n_train_steps=2000 decoder_latent_type=null model.encoder.encoder_latent_type=null model.encoder.epipolar_transformer.upscale=4 decoder_latent_dim=3 model.encoder.d_latent=3 model.decoder.d_latent=3

# Train no latent enc
python3 -m src.main +experiment=acid exp_name='exp' hydra.run.dir='outputs/exp' data_loader.train.batch_size=1 trainer.val_check_interval=30 optimizer.warm_up_steps=1000 checkpointing.every_n_train_steps=2000 model.encoder.encoder_latent_type=null model.encoder.epipolar_transformer.upscale=4 decoder_latent_dim=3 model.encoder.d_latent=3 model.decoder.d_latent=3

# Train no latent enc no upsample 4
python3 -m src.main +experiment=acid exp_name='exp' hydra.run.dir='outputs/exp' data_loader.train.batch_size=1 trainer.val_check_interval=30 optimizer.warm_up_steps=1000 checkpointing.every_n_train_steps=2000 model.encoder.encoder_latent_type=null model.encoder.epipolar_transformer.upscale=0 decoder_latent_dim=3 model.encoder.d_latent=3 model.decoder.d_latent=3





### Metrics

# output_metrics_path=outputs/test/tiny_rnd_bs4_20000/acid/evaluation_metrics.json 
# evaluation.methods.0.path=outputs/test/tiny_rnd_bs4_20000/acid

# PixelSplat on ACID
python3 -m src.scripts.compute_metrics +experiment=acid +evaluation=acid output_metrics_path=outputs/test/pixelsplat/acid/evaluation_metrics.json evaluation.methods.0.path=outputs/test/pixelsplat/acid



# LatentPixelSplat

# tiny_rnd_bs4_20000
python3 -m src.scripts.compute_metrics +experiment=acid +evaluation=acid output_metrics_path=outputs/test/tiny_rnd_bs4_20000/acid/evaluation_metrics.json evaluation.methods.0.path=outputs/test/tiny_rnd_bs4_20000/acid


# acid_tiny_enc_lat_ed_80k
python3 -m src.scripts.compute_metrics +experiment=acid +evaluation=acid output_metrics_path=outputs/acid_tiny_enc_lat_ed_80k/test/acid/evaluation_metrics.json evaluation.methods.0.path=outputs/acid_tiny_enc_lat_ed_80k/test/acid
# output_metrics_path=outputs/acid_tiny_enc_lat_ed_80k/test/acid/evaluation_metrics.json 
# evaluation.methods.0.path=outputs/acid_tiny_enc_lat_ed_80k/test/acid

# acid_tiny_enc_80k
python3 -m src.scripts.compute_metrics +experiment=acid +evaluation=acid output_metrics_path=outputs/acid_tiny_enc_80k/test/acid/evaluation_metrics.json evaluation.methods.0.path=outputs/acid_tiny_enc_80k/test/acid

# acid_tiny_lat_ed_80k
python3 -m src.scripts.compute_metrics +experiment=acid +evaluation=acid output_metrics_path=outputs/acid_tiny_lat_ed_80k/test/acid/evaluation_metrics.json evaluation.methods.0.path=outputs/acid_tiny_lat_ed_80k/test/acid

# acid_tiny_rnd_80k
python3 -m src.scripts.compute_metrics +experiment=acid +evaluation=acid output_metrics_path=outputs/acid_tiny_rnd_80k/test/acid/evaluation_metrics.json evaluation.methods.0.path=outputs/acid_tiny_rnd_80k/test/acid



