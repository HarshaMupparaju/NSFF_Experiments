from pathlib import Path
import shutil
import pandas as pd
from glob import glob

def sparse_flow_aggregator(set_num, dataset_name, scene_names):
    for scene_name in scene_names:
        sparse_flows_dirpath = Path(f'../nerf_data/{dataset_name}/set{set_num:02}/{scene_name}/dense/sparse_flow_colmap/')
        sparse_flows_csvs_filepaths = glob(str(sparse_flows_dirpath / '*.csv'))

        sparse_flows = pd.DataFrame()
        for sparse_flows_csv_filepath in sparse_flows_csvs_filepaths:
            filename = Path(sparse_flows_csv_filepath).stem
            frame1_num, frame2_num = filename.split('_')
            print(filename)
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


def demo1():
    set_num = 1
    dataset_name = 'N3DV'
    scene_names =['coffee_martini']
    sparse_flow_aggregator(set_num, dataset_name, scene_names)
    return

def demo2():
    set_num = 1
    dataset_name = 'Nvidia_processed'
    scene_names =['Balloon1-2']
    sparse_flow_aggregator(set_num, dataset_name, scene_names)
    return

def main():
    # demo1()
    demo2()

if __name__ == '__main__':
    main()
