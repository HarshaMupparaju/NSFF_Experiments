from pathlib import Path
import shutil
import os

database_dirpath = Path('../nerf_data/dynerf/')
out_put_dirpath = Path('../nerf_data/dynerf_processed/')
out_put_dirpath.mkdir(exist_ok=True, parents=True)
scenes = ['coffee_martini', 'cook_spinach', 'cut_roasted_beef', 'flame_salmon_1', 'flame_steak', 'sear_steak']

for scene in scenes:
    scene_dirpath = database_dirpath / scene
    
    image_folders = [folder for folder in scene_dirpath.iterdir() if folder.is_dir()]
    image_folders.sort()
    for i, folder_path in enumerate(image_folders):
        print(f'Processing {folder_path}')
        image_folder = folder_path / 'images'
        out_folder_path = out_put_dirpath / scene / f'{i:02d}/images/'
        out_folder_path.mkdir(exist_ok=True, parents=True)
        image_files = [file for file in image_folder.iterdir() if file.is_file()]

        for image_file in image_files:
            shutil.copy(image_file, out_folder_path / image_file.name)
        print(f'Copied {len(image_files)} images to {out_folder_path}') 
    
    poses_bounds_filepath = scene_dirpath / 'poses_bounds.npy'

    poses_bounds_output_filepath = out_put_dirpath / scene / 'poses_bounds.npy'
    shutil.copy(poses_bounds_filepath, poses_bounds_output_filepath)
