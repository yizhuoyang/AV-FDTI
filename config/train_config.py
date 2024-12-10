import os
import torch

base_dir = '/home/kemove/yyz/'
data_dir = os.path.join(base_dir, 'Data')

CONFIG = {
    "dark_aug": 0, # Simulate pure dark environment
    "audio_seq": 1,
    "workers": 8,
    "kernel_num": 32,
    "feature_dim": 512,
    "checkpoint_path": 'output/best_epoch.pth',
    "base_dir": base_dir,
    "annotation_lines_train": os.path.join(data_dir, 'anotation_split_50/train.txt'),
    "annotation_lines_val": os.path.join(data_dir, 'anotation_split_50/val.txt'),
    "audio_path": os.path.join(data_dir, 'np_data_align'),
    "image_path": os.path.join(data_dir, 'image'),
    "detect_path": os.path.join(data_dir, 'Detection_new'),
    "gt_path": os.path.join(data_dir, 'label'),
    "num_class": 6,
    "device": torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
    "confusion_matrix_path": 'confusion_matrix_d.png'
}
