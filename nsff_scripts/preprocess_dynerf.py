from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import time
import datetime
import os
import skimage
from scipy.spatial.transform import Rotation
from database import COLMAPDatabase, array_to_blob
from colmapUtils.get_inliers import get_inliers

def save_tmp_data(sparse_dirpath: Path, images_dirpath:Path, camera_path: Path, points_path: Path, images: np.ndarray, intrinsics: np.ndarray):
    # TODO: handle differing intrinsics as well
    for intrinsic in intrinsics:
        assert np.allclose(intrinsic, intrinsics[0])
    intrinsic = intrinsics[0]
    camera_id = 1

    sparse_dirpath.mkdir(parents=True, exist_ok=True)

    # for frame_num, image in enumerate(images):
    #     tgt_image_path = images_dirpath / f'{frame_num:04}.jpg'
    #     save_image(tgt_image_path, image)

    # Create cameras.txt file
    h, w = images[0].shape[:2]
    camera_data = f'{camera_id} FULL_OPENCV {w} {h} {intrinsic[0,0]} {intrinsic[1,1]} {intrinsic[0,2]} {intrinsic[1,2]} 0 0 0 0 0 0 0 0 \n'
    with open(camera_path.as_posix(), 'w') as camera_file:
        camera_file.writelines(camera_data)
    camera_data = {
        camera_id: intrinsic
    }

    # Create points3D.txt file
    os.system(f'touch {points_path.as_posix()}')
    return camera_data

@staticmethod
def get_quaternions_and_translations(trans_mat: np.ndarray):
    rot_mat = trans_mat[:3, :3]
    rotation = Rotation.from_matrix(rot_mat)
    assert type(rotation) == Rotation
    quaternions = rotation.as_quat()
    quaternions = np.roll(quaternions, 1)
    quaternions_str = ' '.join(quaternions.astype('str'))
    translations = trans_mat[:3, 3]
    translations_str = ' '.join(translations.astype('str'))
    return quaternions_str, translations_str

def run_colmap(db_path:Path, images_dirpath:Path, sparse_dirpath:Path, images_path:Path, camera_data, extrinsics: np.ndarray):
    cmd = f'colmap feature_extractor --database_path {db_path.as_posix()} --image_path {images_dirpath.as_posix()} --ImageReader.single_camera 1'
    print(cmd)
    os.system(cmd)

    # Reset camera params
    db = COLMAPDatabase.connect(db_path.as_posix())
    # TODO: handle different intrinsics
    camera_id, intrinsic = next(iter(camera_data.items()))
    params = [intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]]
    params = np.asarray(params, np.float64)
    params = array_to_blob(params)
    # db.execute("UPDATE cameras SET params=? WHERE camera_id=?", (params, camera_id))
    db.execute("UPDATE cameras SET model=6, params=? WHERE camera_id=?", (params, camera_id))

    # Create images.txt file
    db_cursor = db.cursor()
    images_data = []
    for frame_num, trans_mat in enumerate(extrinsics):
        quaternions, translations = get_quaternions_and_translations(trans_mat)
        db_cursor.execute(f"SELECT image_id FROM images WHERE name='{frame_num:05}.png'")
        db_image_data = db_cursor.fetchall()
        assert len(db_image_data) == 1
        images_data.append(f'{db_image_data[0][0]} {quaternions} {translations} {camera_id} {frame_num:05}.png\n')
        images_data.append(f'\n')
    with open(images_path.as_posix(), 'w') as images_file:
        images_file.writelines(images_data)
    db.close()

    cmd = f'colmap exhaustive_matcher --database_path {db_path.as_posix()}'
    print(cmd)
    os.system(cmd)

    return

@staticmethod
def save_image(path: Path, image: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    skimage.io.imsave(path.as_posix(), image)
    return

def run_colmap_wrapper(dataset_name, scene_names, set_num):
    root_dirpath = Path('../nerf_data/')
    output_dirpath = root_dirpath / dataset_name / f'set{set_num:02d}'
    database_dirpath = root_dirpath / 'dynerf/'
    for scene_name in scene_names:
        output_scene_dirpath = output_dirpath / scene_name
        output_sparse_dirpath = output_scene_dirpath / 'dense/sparse'
        images_dirpath = output_scene_dirpath / 'dense/images'
        output_sparse_dirpath.mkdir(exist_ok=True, parents=True)
        database_scene_dirpath = database_dirpath / scene_name
        cameras_path = output_sparse_dirpath / 'cameras.txt'
        images_path = output_sparse_dirpath / 'images.txt'
        points_path = output_sparse_dirpath / 'points3D.txt'
        db_path = output_sparse_dirpath / 'database.db'

        train_set_csv_filepath = database_dirpath / 'train_test_sets/set04/TrainVideosData.csv'
        train_set = pd.read_csv(train_set_csv_filepath)
        scene_train_set = train_set[train_set['scene_name'] == scene_name]
        train_nums = list(scene_train_set['pred_video_num'])

        poses_bounds_filepath = output_scene_dirpath / 'dense/poses_bounds.npy'
        poses_bounds = np.load(poses_bounds_filepath)

        poses = poses_bounds[:, :15]
        bounds = poses_bounds[:, 15:]

        extrinsics = poses[:, :12].reshape(-1, 3, 4)
        hwf = poses[0, 12:].reshape(3, 1)

        #Convert poses to colmap format (-y, x, z) to (x, -y, -z)
        #TODO: Check if this is correct, should be changed column-wise or row-wise?
        extrinsics_colmap = np.concatenate([extrinsics[:, 1, :].reshape(-1, 1, 4), extrinsics[:, 0, :].reshape(-1, 1, 4), -extrinsics[:, 2, :].reshape(-1, 1, 4)], axis=1)
        



        intrinsics = np.eye(3)
        intrinsics[0, 0] = hwf[2, 0]
        intrinsics[1, 1] = hwf[2, 0]
        intrinsics[0, 2] = hwf[0, 0] / 2
        intrinsics[1, 2] = hwf[1, 0] / 2
        intrinsics = intrinsics.reshape(1, 3, 3)
        intrinsics = intrinsics.repeat(30, axis=0)
        #Read images
        images = []
        for image_path in images_dirpath.iterdir():
            image = skimage.io.imread(image_path)
            images.append(image)
        
        images = np.array(images)
        camera_data = save_tmp_data(output_sparse_dirpath, images_dirpath, cameras_path, points_path, images, intrinsics)
        run_colmap(db_path, images_dirpath, output_sparse_dirpath, images_path, camera_data, extrinsics_colmap)




def preprocess_dynerf(dataset_name, scene_names, set_num):
    for scene_name in scene_names:
        output_dirpath = Path(f'../nerf_data/{dataset_name}/set{set_num:02d}/{scene_name}/dense')
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
    dataset_name = 'N3DV'
    scene_names = ['coffee_martini', 'cook_spinach', 'cut_roasted_beef', 'flame_steak', 'sear_steak']
    set_num = 4
    preprocess_dynerf(dataset_name, scene_names, set_num)
    run_colmap_wrapper(dataset_name, scene_names, set_num)
    return

def main():
    demo1()
    return

if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print(f'Elapsed time: {datetime.timedelta(seconds=end-start)}')
