# Introduction 

This is the official PL-MipNeRF implementation using NeRFStudio. 

## Set-up
PL-MipNeRF is tested on Blender Synthetic dataset. Please follow the main repo's data preparation to setup data. 

### Installation 
PL-MipNeRF relies on [NeRFStudio](https://github.com/nerfstudio-project/nerfstudio/tree/main). Please install NeRFStudio and its dependencies following the instructions from their repo. We tested our model on version 0.3.4. 

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