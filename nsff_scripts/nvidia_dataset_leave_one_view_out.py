from pathlib import Path
import numpy as np 
import shutil

scene_name = 'Balloon1-2'

original_dataset_path = Path(f'../nerf_data/nvidia_data_full/{scene_name}/dense/')

output_dataset_path = Path(f'../nerf_data/nvidia_monocular_leave_one_view_out/{scene_name}/dense/')
output_dataset_path.mkdir(exist_ok=True, parents=True)

poses_bounds = np.load(original_dataset_path / 'poses_bounds.npy')
output_train_poses_bounds = np.concatenate([poses_bounds[1:12,:] , poses_bounds[13:, :]], axis=0)
output_test_poses_bounds = np.concatenate([poses_bounds[0:1,:] , poses_bounds[12:13, :]], axis=0)

images_dirpath = original_dataset_path / 'images'
output_train_images_dirpath = output_dataset_path / 'images'
output_test_images_dirpath = output_dataset_path / 'test_images'

output_train_images_dirpath.mkdir(exist_ok=True, parents=True)
output_test_images_dirpath.mkdir(exist_ok=True, parents=True)

image_files = list(images_dirpath.glob('*.jpg'))
image_files.sort()

train_image_files = image_files[1:12] + image_files[13:]
test_image_files = image_files[0:1] + image_files[12:13]

for i, image_file in enumerate(train_image_files):
    output_image_file = output_train_images_dirpath / f'{i:05d}.jpg'
    shutil.copy(image_file, output_image_file)

for i, image_file in enumerate(test_image_files):
    output_image_file = output_test_images_dirpath / f'{i:05d}.jpg'
    shutil.copy(image_file, output_image_file)

np.save(output_dataset_path / 'poses_bounds.npy', output_train_poses_bounds)
np.save(output_dataset_path / 'test_poses_bounds.npy', output_test_poses_bounds)




