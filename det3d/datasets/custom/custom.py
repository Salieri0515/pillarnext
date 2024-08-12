import os
import numpy as np
import pickle
import open3d as o3d
import itertools
from torch.utils.data import Dataset


# This dataset is only for evaluation
# info_path (.pkl) should be organized as: [{'token': frame_token, 'lidar_path': frame_absolute_path}, ...]
class CustomDatasetVal(Dataset):
    def __init__(self, info_path, nsweeps, class_names, loading_pipelines=None):
        self.info_path = info_path
        self.loading_pipelines = loading_pipelines
        self.nsweeps = nsweeps
        self.class_names = list(itertools.chain(*[t for t in class_names]))
        self.info = None
        self.load_info()

    def load_info(self):
        with open(self.info_path, 'rb') as a:
            self.info = pickle.load(a)

    def read_pcd_with_intensity(self, file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()

        # 找到 DATA ascii 行的索引
        data_index = lines.index("DATA ascii\n") + 1

        # 解析点云数据
        point_data = []
        for line in lines[data_index:]:
            parts = line.split()
            x, y, z, intensity = map(float, parts)
            point_data.append([x, y, z, intensity])
        point_data = np.concatenate((point_data, np.zeros((len(point_data), 1))), axis=1)
        print(point_data.shape)
        return np.array(point_data)

    def load_pointcloud(self, res, info):
        # TODO: concat times with points, points should be xyz, intensity, timestamp(each sweep)
        pc_path = info['lidar_path']
        token = info['token']

        pc = self.read_pcd_with_intensity(pc_path)

        res['points'] = pc.astype(np.float32)
        res['token'] = token
        return res

    def __len__(self):
        return len(self.info)

    def __getitem__(self, idx):
        info = self.info[idx]
        res = {}
        if self.loading_pipelines is not None:
            for lp in self.loading_pipelines:
                res = getattr(self, lp)(res, info)
        return res
