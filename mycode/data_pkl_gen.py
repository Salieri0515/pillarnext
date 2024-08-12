import pickle
import glob
import os
import argparse
from tqdm import tqdm


# info_path (.pkl) should be organized as: [{'token': frame_token, 'lidar_path': frame_absolute_path
# , 'sweeps': [sweep_pc_absolute_path1, sweep_pc_absolute_path2, ...]}, ...]
# TODO: add sweep
def main(args):
    data_path = args.data_path
    filelist = glob.glob(os.path.join(data_path, '*.pcd'))
    res = []
    for file in tqdm(filelist):
        frame_token = file.split('/')[-1].split('.')[0]
        lidar_path = file
        res.append({'token': frame_token, 'lidar_path': lidar_path})
    with open('debug_data.pkl', 'wb') as a:
        pickle.dump(res, a)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='absolute path of data folder')
    args = parser.parse_args()
    main(args)