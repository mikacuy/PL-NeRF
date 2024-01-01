# Introduction 

This is the official PL-MipNeRF implementation using NeRFStudio. 

## Set-up
PL-MipNeRF is tested on Blender Synthetic dataset. Please follow the main repo's data preparation to setup data. 

### Installation 
PL-MipNeRF relies on [NeRFStudio](https://github.com/nerfstudio-project/nerfstudio/tree/main). Please install NeRFStudio and its dependencies following the instructions from their repo. We tested our model on version 0.3.4 with CUDA 11.8. Refer to its website for installation details.  

After installing NeRFStudio, run 
```
pip install -e .
```
to install PL-MipNeRF related files. 

# Training
To train PL-MipNeRF on Blender synthetic dataset, run 
```
ns-train linear-mipnerf --vis wandb --pipeline.model.use_same_field=False  --pipeline.model.use_constant_importance_sampling=True --pipeline.model.concat_walls=False --pipeline.model.loss_coefficients.rgb_loss_coarse=1.0 --experiment_name=pl_mipnerf_SCENE blender-data --data data/nerf_synthetic/SCENE
```
where SCENE should be replaced with one of the scene names. 

To train vanilla MipNeRF, run 
```
ns-train mipnerf  --pipeline.model.collider_params near_plane 2.0 far_plane 6.0  --max_num_iterations=1000001 --experiment_name=mipnerf_SCENE blender-data --data data/nerf_synthetic/SCENE
```

# Evaluating 
We provide pretrained model weights for PL-MipNeRF on Blender Synthetic dataset. Please download the weights [here](http://download.cs.stanford.edu/orion/pl-nerf/linear_mip_weights/pretrained.zip). Unzip the file under this directory by 
```
unzip [PATH TO pretrained.zip] -d ./
```
and run 
```
ns-eval --load-config pretrained/linear_SCENE/config.yml
```
to obtain rendered test views and reproduce the reported scores. 

Refer to [NerfStudio documentation](https://docs.nerf.studio/) for a detailed evaluation script usage. 