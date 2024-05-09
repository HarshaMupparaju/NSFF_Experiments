from pathlib import Path
import shutil
import pandas as pd
from glob import glob

def sparse_flow_aggregator(set_num, dataset_name, scene_names):
    for scene_name in scene_names:
        sparse_flows_dirpath = Path('../nerf_data/N3DV/set04/coffee_martini/dense/sparse_flow_colmap/')
        sparse_flows_csvs_filepaths = glob(str(sparse_flows_dirpath / '*.csv'))

        sparse_flows = pd.DataFrame()
        for sparse_flows_csv_filepath in sparse_flows_csvs_filepaths:
            filename = Path(sparse_flows_csv_filepath).stem
            frame1_num, frame2_num = filename.split('_')
            sparse_flow = pd.read_csv(sparse_flows_csv_filepath)
            sparse_flow['frame1_num'] = int(frame1_num)
            sparse_flow['frame2_num'] = int(frame2_num)
            sparse_flows = pd.concat([sparse_flows, sparse_flow])

        sparse_flows = sparse_flows.reset_index(drop=True)
        #Reaarange columns
        sparse_flows = sparse_flows[['frame1_num', 'frame2_num', 'x1', 'y1', 'x2', 'y2']]
        sparse_flows = sparse_flows.sort_values(by=['frame1_num', 'frame2_num'])
        sparse_flows.to_csv(sparse_flows_dirpath / 'matched_pixels.csv', index=False)
    return


def main():
    set_num = 4
    dataset_name = 'N3DV'
    scene_names =['coffee_martini']
    sparse_flow_aggregator(set_num, dataset_name, scene_names)
    return

if __name__ == '__main__':
    main()
