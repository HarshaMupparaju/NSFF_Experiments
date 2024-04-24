from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import time
import datetime

def preprocess_dynerf(scene_names, set_num):
    for scene_name in scene_names:
        output_dirpath = Path(f'../nerf_data/N3DV/set{set_num:02d}/{scene_name}/')
        output_images_dirpath = output_dirpath / 'images/'
        output_images_dirpath.mkdir(exist_ok=False, parents=True)
        root_dirpath = Path('../nerf_data/')
        database_dirpath = root_dirpath / 'dynerf/'
        scene_dirpath = database_dirpath / scene_name
        train_test_set_dirpath = database_dirpath / f'train_test_sets/set{set_num:02d}/'
        train_set_csv_filepath = train_test_set_dirpath /'TrainVideosData.csv'
        test_set_csv_filepath = train_test_set_dirpath /'TestVideosData.csv'
        poses_bounds_filepath = scene_dirpath / 'poses_bounds.npy'
        poses_bounds = np.load(poses_bounds_filepath)
        train_set = pd.read_csv(train_set_csv_filepath)
        scene_train_set = train_set[train_set['scene_name'] == scene_name]
        train_nums = list(scene_train_set['pred_video_num'])

        #Pick frames
        frames = list(range(0, 100, 10))
        poses_bounds_processed = []
        for i, train_num in enumerate(train_nums):
            for j, frame in enumerate(frames):
                image_filepath = scene_dirpath / f'{train_num:02d}/images/{frame:05d}.png'
                output_filepath = output_dirpath / f'images/{(i * 10 + j):05d}.png'
                shutil.copy(image_filepath, output_filepath)
            
            poses_bounds_required = poses_bounds[i]
            poses_bounds_required_tiled = np.tile(poses_bounds_required, (10, 1))
            poses_bounds_processed.append(poses_bounds_required_tiled)

        #Copy poses_bounds.npy
        output_poses_bounds_filepath = output_dirpath / 'poses_bounds.npy'
        #TODO: Don't hardcode
        poses_bounds_processed = np.array(poses_bounds_processed).reshape(30,17)
        np.save(output_poses_bounds_filepath, poses_bounds_processed)

    return




def demo1():
    scene_names = ['coffee_martini']
    set_num = 4
    preprocess_dynerf(scene_names, set_num)
    return

def main():
    demo1()
    return

if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print(f'Elapsed time: {datetime.timedelta(seconds=end-start)}')
