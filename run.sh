
# Test PixelSplat on ACID with pre-trained weights
python3 -m src.main +experiment=acid mode=test dataset/view_sampler=evaluation checkpointing.load=pretrained_models/acid.ckpt dataset.view_sampler.index_path=assets/evaluation_index_acid.json

# python3 -m src.main \
# +experiment=acid \
# mode=test \
# dataset/view_sampler=evaluation \
# checkpointing.load=pretrained_models/acid.ckpt \
# dataset.view_sampler.index_path=assets/evaluation_index_acid.json


# Test on ACID with latent encoder and decoder with pre-trained weights
python3 -m src.main +experiment=acid mode=test dataset/view_sampler=evaluation checkpointing.load=pretrained_models/acid_latent_d3_f4_noattn.ckpt dataset.view_sampler.index_path=assets/evaluation_index_acid.json

# Test without checkpoint
python3 -m src.main +experiment=acid mode=test dataset/view_sampler=evaluation dataset.view_sampler.index_path=assets/evaluation_index_acid.json


# Train
# trainer.val_check_interval=30
# trainer.log_every_n_steps=30 ####################
# optimizer.warm_up_steps=1000
# checkpointing.load=pretrained_models/acid_latent_d3_f4_noattn.ckpt
# checkpointing.every_n_train_steps=2000

# Train without checkpoint
python3 -m src.main +experiment=acid data_loader.train.batch_size=1 trainer.val_check_interval=30 optimizer.warm_up_steps=1000 checkpointing.every_n_train_steps=2000

# Train with checkpoint
python3 -m src.main +experiment=acid data_loader.train.batch_size=1 trainer.val_check_interval=30 optimizer.warm_up_steps=1000 checkpointing.load=pretrained_models/acid_latent_d3_f4_noattn.ckpt checkpointing.every_n_train_steps=2000 trainer.max_steps=350000






