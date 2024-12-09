import os

base_dir = '/home/kemove/yyz/'
data_dir = os.path.join(base_dir, 'Data')

CONFIG = {
    "dark_aug": 1,
    "audio_seq": 1,
    "workers": 4,
    "batchsize": 32,
    "dropout_rate": 0.3,
    "kernel_num": 16,
    "num_class": 6,
    "feature_dim": 256,
    "epochs": 200,
    "learning_rate": 0.0001,
    "checkpoint_path": '',
    "base_dir": base_dir,
    "annotation_lines_train": os.path.join(data_dir, 'anotation_split_50/train.txt'),
    "annotation_lines_val": os.path.join(data_dir, 'anotation_split_50/val.txt'),
    "audio_path": os.path.join(data_dir, 'np_data_align'),
    "image_path": os.path.join(data_dir, 'image'),
    "detect_path": os.path.join(data_dir, 'Detection_new'),
    "gt_path": os.path.join(data_dir, 'label'),
    "save_path": 'output/',
}
os.makedirs(CONFIG["save_path"], exist_ok=True)
