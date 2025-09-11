import argparse
import glob
from pathlib import Path

try:
    import open3d
    from visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V
    OPEN3D_FLAG = False

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from collections import defaultdict

from scipy.io import savemat

from utils_sw import SpatialTransform

class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        elif self.ext == '.pcd':
            points = self.read_pcd_ascii(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        print(points)
        print(points.shape)
        # input_dict = {
        #     'points': points,
        # }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict
    
    def read_pcd_ascii(self, filename):
        with open(filename, 'r') as f:
            lines = f.readlines()

        # 헤더 끝 지점 찾기
        data_start = 0
        for i, line in enumerate(lines):
            if line.strip() == 'DATA ascii':
                data_start = i + 1
                break

        # 데이터 부분 읽기
        data = []
        for line in lines[data_start:]:
            vals = line.strip().split()
            if len(vals) == 4:
                # if vals[0] == 'nan' or vals[1] == 'nan' or vals[2] == 'nan':
                #     continue
                try:
                    x, y, z, intensity = map(np.float64, vals)
                    data.append([x, y, z, 0])
                except ValueError:
                    continue

        return np.array(data)
    
def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg

def filter_predictions_by_threshold(pred_dicts, pred_threshold):
    """
    Filters predictions in pred_dicts based on class-specific threshold in pred_threshold.

    Args:
        pred_dicts (dict): {
            'pred_boxes': Tensor(N, 7),
            'pred_scores': Tensor(N),
            'pred_labels': Tensor(N)
        }
        pred_threshold (dict): {class_id: threshold_score}

    Returns:
        dict: filtered pred_dicts
    """
    # 각 label별 threshold에 따른 mask 생성
    scores = pred_dicts['pred_scores']
    labels = pred_dicts['pred_labels']

    keep_mask = torch.zeros_like(scores, dtype=torch.bool)
    for class_id, threshold in pred_threshold.items():
        class_mask = (labels == class_id) & (scores >= threshold)
        keep_mask |= class_mask

    # mask 적용
    filtered_pred_dicts = {
        'pred_boxes': pred_dicts['pred_boxes'][keep_mask],
        'pred_scores': pred_dicts['pred_scores'][keep_mask],
        'pred_labels': pred_dicts['pred_labels'][keep_mask]
    }

    return filtered_pred_dicts

def organize_boxes_by_class_for_matlab(data_dict, num_classes):
    class_to_boxes = defaultdict(list)

    for idx in range(len(data_dict['pred_labels'])):
        label = data_dict['pred_labels'][idx].item()
        box = data_dict['pred_boxes'][idx].cpu().numpy()

        xyz_wlh = box[:6]
        heading = np.rad2deg(box[-1])
        rpy = np.array([0., 0., heading])

        _box = np.concatenate([xyz_wlh, rpy])

        # print(_box)
        class_to_boxes[label].append(_box)

    row = []
    for class_id in range(1, num_classes + 1):
        if class_id in class_to_boxes:
            boxes = np.array(class_to_boxes[class_id])
        else:
            boxes = np.empty((0, 0), dtype=np.uint8)
        row.append(boxes)

    return row

def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()

    label_array = np.empty((len(demo_dataset), len(cfg.CLASS_NAMES)), dtype=object)

    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

            # print(pred_dicts)

            # ['Vehicle', 'Pedestrian', 'Cyclist']
            filtered = filter_predictions_by_threshold(pred_dicts[0], {1:0.65, 2:0.45, 3:0.5})

            # ['car','truck', 'construction_vehicle', 'bus', 'trailer',
            #'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
            # filtered = filter_predictions_by_threshold(pred_dicts[0], {1:0.7, 2:0.7, 3:0.7, 4:0.7, 5:0.6, 6:0.6, 7:0.4, 8:0.4, 9:0.35, 10:0.6})

            converted = organize_boxes_by_class_for_matlab(filtered, len(cfg.CLASS_NAMES))

            label_array[idx] = converted
            # V.draw_scenes(
            #     points=data_dict['points'][:, 1:], ref_boxes=filtered['pred_boxes'],
            #     ref_scores=filtered['pred_scores'], ref_labels=filtered['pred_labels']
            # )

            # if not OPEN3D_FLAG:
            #     mlab.show(stop=True)

    # logger.info('Demo done.')
    savemat("label_array_from_python.mat", {"label_array": label_array})

if __name__ == '__main__':
    main()
