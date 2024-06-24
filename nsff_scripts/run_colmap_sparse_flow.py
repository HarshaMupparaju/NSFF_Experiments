import numpy as np
import pandas as pd
from pathlib import Path
import shutil
import os
import glob

from colmapUtils.get_inliers import get_matches

def estimate_sparse_flow_dynerf():
    set_num = 1
    scene_names = ['coffee_martini']
    for scene_name in scene_names:
        data_dirpath = Path(f'../nerf_data/N3DV/set{set_num:02}/{scene_name}/dense')
        images_dirpath = data_dirpath / 'images_1352x1014'
        poses_bounds_filepath = data_dirpath / 'poses_bounds.npy'

        tmp_dirpath = Path('../tmp')

        tmp_images_dirpath = tmp_dirpath / 'images'


        tmp_db_dirpath = tmp_dirpath / 'database.db'
        sparse_dirpath = data_dirpath / 'sparse'
        sparse_dirpath1 = sparse_dirpath / '0'


        output_dirpath = data_dirpath / 'sparse_flow_colmap'
        output_dirpath.mkdir(exist_ok=True)

        #Make l1 and l2 dynamically
        num_cams = len(glob.glob(str(images_dirpath / '*.png'))) / 10
        l1 = []
        l2 = []
        for i in range(int(num_cams)):
            l1 += list(np.arange(i * 10, i * 10 + 9)) + list(np.arange(i * 10 + 1, i * 10 + 10))
            l2 += list((np.arange(i * 10, i * 10 + 9) + 11) % (int(num_cams) * 10)) + list((np.arange(i * 10 + 1, i * 10 + 10) + 9) % (int(num_cams) * 10))
        
        
        for i, j in zip(l1, l2):
            image1_filepath = images_dirpath / f'{i:05d}.png'
            image2_filepath = images_dirpath / f'{j:05d}.png'
            tmp_images_dirpath.mkdir(exist_ok=True, parents=True)
            shutil.copy(image1_filepath, tmp_images_dirpath / '0000.png')
            shutil.copy(image2_filepath, tmp_images_dirpath / '0001.png')

            #Run Colmap
            cmd = f'colmap feature_extractor --database_path {tmp_db_dirpath} --image_path {tmp_images_dirpath} --ImageReader.single_camera 1 --SiftExtraction.estimate_affine_shape=true --SiftExtraction.domain_size_pooling=true'
            print(cmd)
            os.system(cmd)

            cmd = f'colmap exhaustive_matcher --database_path {tmp_db_dirpath}'
            print(cmd)
            os.system(cmd)

            matches_data = get_matches(str(tmp_db_dirpath))

            output_filepath = output_dirpath / f'{i:05d}_{j:05d}.csv'
            matches_data = pd.DataFrame(matches_data, columns=['x1', 'y1', 'x2', 'y2'])
            matches_data.to_csv(output_filepath, index=False)

            #Clean tmp directory
            shutil.rmtree(tmp_dirpath)
    return

def estimate_sparse_flow_nvidia():
    set_num = 1
    scene_names = ['Balloon1-2']
    for scene_name in scene_names:
        data_dirpath = Path(f'../nerf_data/Nvidia_processed/set{set_num:02}/{scene_name}/dense')
        images_dirpath = data_dirpath / 'images_540x288'
        poses_bounds_filepath = data_dirpath / 'poses_bounds.npy'

        tmp_dirpath = Path('../tmp')

        tmp_images_dirpath = tmp_dirpath / 'images'


        tmp_db_dirpath = tmp_dirpath / 'database.db'
        sparse_dirpath = data_dirpath / 'sparse'
        sparse_dirpath1 = sparse_dirpath / '0'


        output_dirpath = data_dirpath / 'sparse_flow_colmap'
        output_dirpath.mkdir(exist_ok=True)

        #Make l1 and l2 dynamically
        num_cams = len(glob.glob(str(images_dirpath / '*.png'))) / 10
        l1 = []
        l2 = []
        for i in range(int(num_cams)):
            l1 += list(np.arange(i * 10, i * 10 + 9)) + list(np.arange(i * 10 + 1, i * 10 + 10))
            l2 += list((np.arange(i * 10, i * 10 + 9) + 11) % (int(num_cams) * 10)) + list((np.arange(i * 10 + 1, i * 10 + 10) + 9) % (int(num_cams) * 10))
        
        
        for i, j in zip(l1, l2):
            image1_filepath = images_dirpath / f'{i:05d}.png'
            image2_filepath = images_dirpath / f'{j:05d}.png'
            tmp_images_dirpath.mkdir(exist_ok=True, parents=True)
            shutil.copy(image1_filepath, tmp_images_dirpath / '0000.png')
            shutil.copy(image2_filepath, tmp_images_dirpath / '0001.png')

            #Run Colmap
            cmd = f'colmap feature_extractor --database_path {tmp_db_dirpath} --image_path {tmp_images_dirpath} --ImageReader.single_camera 1 --SiftExtraction.estimate_affine_shape=true --SiftExtraction.domain_size_pooling=true'
            print(cmd)
            os.system(cmd)

            cmd = f'colmap exhaustive_matcher --database_path {tmp_db_dirpath}'
            print(cmd)
            os.system(cmd)

            matches_data = get_matches(str(tmp_db_dirpath))

            output_filepath = output_dirpath / f'{i:05d}_{j:05d}.csv'
            matches_data = pd.DataFrame(matches_data, columns=['x1', 'y1', 'x2', 'y2'])
            matches_data.to_csv(output_filepath, index=False)

            #Clean tmp directory
            shutil.rmtree(tmp_dirpath)
    return

def main():
    # estimate_sparse_flow_dynerf()
    estimate_sparse_flow_nvidia()
    return

if __name__ == '__main__':
    main()