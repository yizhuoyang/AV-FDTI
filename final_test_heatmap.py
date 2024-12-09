import os
import torch
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torchvision import transforms as trans
from preprocess.audio_process import *
from utils.metrics import calculate_result
from utils.test_loader import load_test,load_annotations,calculate_metrics,evaluate_model,plot_confusion_matrix
from nets.AVFDTI import AVFDTI


# Configuration
base_dir = '/home/kemove/yyz/'
data_dir = os.path.join(base_dir, 'Data')
CONFIG = {
    "dark_aug": 100, # Simulate pure dark environment
    "audio_seq": 1,
    "workers": 8,
    "kernel_num": 16,
    "feature_dim": 256,
    "checkpoint_path": 'output/last_epoch.pth',
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

def main():
    config = CONFIG
    val_lines = load_annotations(config["annotation_lines_val"])

    model = AVFDTI(kernel_num=config["kernel_num"],feature_dim=config["feature_dim"],num_class=config["num_class"])
    model = model.to(config["device"])
    model.eval()
    if config["checkpoint_path"]:
        model.load_state_dict(torch.load(config["checkpoint_path"]))

    gt_class, predict_class, real_positions, predicted_positions, class_right = evaluate_model(model, val_lines, config)
    calculate_metrics(gt_class, predict_class, real_positions, predicted_positions, config)
    plot_confusion_matrix(gt_class, predict_class, config)

if __name__ == "__main__":
    main()
