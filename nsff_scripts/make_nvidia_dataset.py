from pathlib import Path
import pandas as pd
import numpy as np
import shutil
import glob

scene_names = ['Balloon1-2', 'Balloon2-2', 'DynamicFace-2', 'Jumping', 'Playground', 'Skating-2', 'Truck-2', 'Umbrella']

for scene_name in scene_names:
    data_dir = Path(f'/mnt/2tb-hdd/Harsha/NSFF/nerf_data/nvidia_data_full/{scene_name}/dense')
    images_data_dir = data_dir / 'mv_images'
    poses_bounds_filepath = data_dir / 'poses_bounds.npy'

    output_dir = Path('../nerf_data/Nvidia')
    output_dir.mkdir(exist_ok=True, parents=True)

    output_scene_dir = output_dir / f'{scene_name}' 
    output_scene_dir.mkdir(exist_ok=True, parents=True)


    poses_bounds = np.load(poses_bounds_filepath)   
    # print(poses_bounds)
    # print(poses_bounds[0] - poses_bounds[12])
    # print(1/0)
    image_dir_paths = glob.glob(str(images_data_dir / '*'))
    image_dir_paths.sort()
    for i, image_dir_path in enumerate(image_dir_paths):
        image_dir = Path(image_dir_path)
        image_files = list(image_dir.glob('*.jpg'))
        image_files.sort()
        for j, image_file in enumerate(image_files):
            output_image_folder = output_scene_dir / f'{j:02d}/images'
            output_image_folder.mkdir(exist_ok=True, parents=True)
            output_image_file = output_image_folder / f'{i:05d}.jpg'
            shutil.copy(image_file, output_image_file)
        

    poses_bounds_processed = poses_bounds[:12,:]
    np.save(output_scene_dir / 'poses_bounds.npy', poses_bounds_processed)


