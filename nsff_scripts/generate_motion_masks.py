from pathlib import Path
import numpy as np
import skimage
import glob
import matplotlib.pyplot as plt

def generation_motion_masks(dataset_name, scene_names, set_num, num_frames):
    data_dirpath = Path(f'../nerf_data/{dataset_name}/set{set_num:02d}/')
    for scene_name in scene_names:
        scene_dirpath = data_dirpath / scene_name / 'dense'
        output_motion_masks_dirpath = scene_dirpath / 'motion_masks'
        output_motion_masks_dirpath.mkdir(exist_ok=True, parents=True)
        flow_dirpath = scene_dirpath / 'flow_i1'
        for frame in range(num_frames):
            if(frame == 0 or frame == 10 or frame == 20):
                flow_paths = glob.glob(str(flow_dirpath / f'{frame:05d}_fwd.npz'))
            elif(frame == 9 or frame == 19 or frame == 29):
                flow_paths = glob.glob(str(flow_dirpath / f'{frame:05d}_bwd.npz'))
            else:
                flow_paths = glob.glob(str(flow_dirpath / f'{frame:05d}_*.npz'))
            print(flow_paths)
            if(len(flow_paths) == 2):
                flow1 = np.load(flow_paths[0])
                flow2 = np.load(flow_paths[1])

                flow1 = flow1['flow']
                flow1 = np.sqrt(np.sum(flow1**2, axis=2))

                flow2 = flow2['flow']
                flow2 = np.sqrt(np.sum(flow2**2, axis=2))

                threshold = 2
                flow1_mask = flow1 > threshold
                flow2_mask = flow2 > threshold
                motion_mask = flow1_mask | flow2_mask

            elif(len(flow_paths) == 1):
                flow = np.load(flow_paths[0])
                flow = flow['flow']
                flow = np.sqrt(np.sum(flow**2, axis=2))
                threshold = 2
                motion_mask = flow > threshold

            #Visualize motion mask
            # plt.imshow(motion_mask)
            # plt.show()
            skimage.io.imsave(output_motion_masks_dirpath / f'{frame:05d}.png', motion_mask.astype(np.uint8)*255)


def demo1():
    scene_names = ['coffee_martini']
    set_num = 4
    dataset_name = 'N3DV'
    num_frames = 30
    generation_motion_masks(dataset_name, scene_names, set_num, num_frames)
    return

def main():
    demo1()
    return

if __name__ == '__main__':
    main()