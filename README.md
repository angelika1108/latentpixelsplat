# LatentPixelSplat

This project is based on **[pixelSplat:](https://github.com/dcharatan/pixelsplat) 3D Gaussian Splats from Image Pairs for Scalable Generalizable 3D Reconstruction** by David Charatan, Sizhe Li, Andrea Tagliasacchi, and Vincent Sitzmann.

The goal is to integrate a tiny VAE-based encoder-decoder in the pixelSplat architecture. The input of the model are two context images and multiple target viewing directions. The output of the model are target images that correspond to the input viewing directions.

The latent encoder takes the 256x256 context images as input and outputs 64x64 latent feature maps. These feature maps are added to the output feature maps of pixelSplat's encoder. Consequently, Gaussians are predicted in a smaller latent space and then they are projected into 64x64 image frames by rasterization, which are the target images. The latent decoder then upsamples the target images to have spatial dimensions 256x256.

This architectural change accelerates pixelSplat's inference and training time, while maintaining a comparable visual quality.

For the necessary library installations and information about datasets, training, evaluation, etc. please visit the [GitHub page](https://github.com/dcharatan/pixelsplat) of pixelSplat.


## BibTeX

```
@inproceedings{charatan23pixelsplat,
      title={pixelSplat: 3D Gaussian Splats from Image Pairs for Scalable Generalizable 3D Reconstruction},
      author={David Charatan and Sizhe Li and Andrea Tagliasacchi and Vincent Sitzmann},
      year={2023},
      booktitle={arXiv},
}
```
