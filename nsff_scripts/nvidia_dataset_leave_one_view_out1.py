from pathlib import Path
import numpy as np 
import shutil

scene_name = 'Balloon1-2'

original_dataset_path = Path(f'../nerf_data/Nvidia/{scene_name}/')

output_dataset_path = Path(f'../nerf_data/nvidia_monocular_leave_one_view_out/{scene_name}/dense/')
output_dataset_path.mkdir(exist_ok=True, parents=True)

output_train_images_dirpath = output_dataset_path / 'images'
output_test_images_dirpath = output_dataset_path / 'test_images'

output_train_images_dirpath.mkdir(exist_ok=True, parents=True)
output_test_images_dirpath.mkdir(exist_ok=True, parents=True)

poses_bounds = np.load(original_dataset_path / 'poses_bounds.npy')

test_cameras = [0]
train_cameras = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
test_poses_bounds = poses_bounds[test_cameras]
train_poses_bounds = poses_bounds[train_cameras]
num_time_stamps = 24
train_poses_bounds_final = []
test_poses_bounds_final = []

for i in range(num_time_stamps):
    j = i % len(train_cameras)
    src_image_filepath = original_dataset_path / f'{train_cameras[j]:02d}/images/{i:05d}.jpg'
    dst_image_filepath = output_train_images_dirpath / f'{i:05d}.png'
    shutil.copy(src_image_filepath, dst_image_filepath)
    train_poses_bounds_final.append(train_poses_bounds[j])

    src_test_image_filepath = original_dataset_path / f'00/images/{i:05d}.jpg'
    dst_test_image_filepath = output_test_images_dirpath / f'{i:05d}.png'
    shutil.copy(src_test_image_filepath, dst_test_image_filepath)
    test_poses_bounds_final.append(test_poses_bounds[0])


train_poses_bounds_final = np.array(train_poses_bounds_final)
test_poses_bounds_final = np.array(test_poses_bounds_final)


np.save(output_dataset_path / 'poses_bounds.npy', train_poses_bounds_final)
np.save(output_dataset_path / 'test_poses_bounds.npy', test_poses_bounds_final)
