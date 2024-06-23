import numpy as np 
import pandas as pd
from pathlib import Path
import shutil

def make_train_test_set(set_num, scene_names, num_train_videos, num_test_videos):
    #TODO: Implement the Uniform Sampling for sparse views
    train_df = pd.DataFrame(columns=['scene_name', 'pred_video_num'])
    test_df = pd.DataFrame(columns=['scene_name', 'pred_video_num'])
    for scene_name in scene_names:
        data_dir = Path(f'../nerf_data/Nvidia/{scene_name}/')
        cameras_list = list(data_dir.glob('*/'))
        cameras_list.sort()
        cameras = []
        for camera in cameras_list:
            camera_name = camera.stem
            cameras.append(camera_name)
        cameras = cameras[:-1]
        cameras = [int(camera) for camera in cameras]
        if(num_test_videos == 1):
            test_cameras = [cameras[0]]
        
        if(num_train_videos == -1):
            train_cameras = cameras[1:]
        
        for camera in train_cameras:
            train_df = train_df.append({'scene_name': scene_name, 'pred_video_num': camera}, ignore_index = True)
        
        for camera in test_cameras:
            test_df = test_df.append({'scene_name': scene_name, 'pred_video_num': camera}, ignore_index = True)


    train_csv_output_path = Path(f'../nerf_data/Nvidia/train_test_sets/set{set_num:02d}/TrainVideosData.csv')
    test_csv_output_path = Path(f'../nerf_data/Nvidia/train_test_sets/set{set_num:02d}/TestVideosData.csv')

    train_df.to_csv(train_csv_output_path, index=False)
    test_df.to_csv(test_csv_output_path, index=False)

def demo1():
    set_num = 1
    scene_names = ['Balloon1-2', 'Balloon2-2', 'DynamicFace-2', 'Jumping', 'Playground', 'Skating-2', 'Truck-2', 'Umbrella']
    num_test_videos = 1
    num_train_videos = -1
    make_train_test_set(set_num, scene_names, num_train_videos, num_test_videos)

def main():
    demo1()

    return

if __name__ == '__main__':
    main()