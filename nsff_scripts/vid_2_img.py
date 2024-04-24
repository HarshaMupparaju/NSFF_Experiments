import numpy as np
import skvideo.io
import skimage
from pathlib import Path
import shutil
from tqdm import tqdm

unzipped_dirpath = Path('../nerf_data/dynerf_zipped/')
output_dirpath = Path('../nerf_data/dynerf/')

scene_names = ['coffee_martini', 'cook_spinach', 'cut_roasted_beef', 'flame_salmon_1', 'flame_steak', 'sear_steak']

for scene_name in scene_names:
    scene_dirpath = unzipped_dirpath / scene_name
    output_scene_dirpath = output_dirpath / scene_name
    # output_scene_dirpath.mkdir(exist_ok=False, parents=True)

    video_files = [file for file in scene_dirpath.iterdir() if file.is_file()]
    video_files.remove(scene_dirpath / 'poses_bounds.npy')
    video_files.sort()
    for i, video_file in enumerate(video_files):
        print(f'Processing {video_file}')
        output_images_dirpath = output_scene_dirpath / f'{i:02d}/images/'
        output_images_dirpath.mkdir(exist_ok=False, parents=True)
        video = skvideo.io.vread(str(video_file))
        for j, frame in tqdm(enumerate(video)):
            skimage.io.imsave(output_images_dirpath / f'{j:05d}.png', frame)
        print(f'Extracted frames to {output_images_dirpath}')
    
    poses_bounds_filepath = scene_dirpath / 'poses_bounds.npy'
    output_poses_bounds_filepath = output_scene_dirpath / 'poses_bounds.npy'
    shutil.copy(poses_bounds_filepath, output_poses_bounds_filepath)

    